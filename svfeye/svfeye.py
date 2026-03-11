from svfeye.svfeye_model import SVFEYModel
from svfeye.utils import *
from svfeye.qwen2_5_methods import rel_attention_qwen2_5
from PIL import Image, ImageDraw
from copy import deepcopy
import os
import matplotlib.cm as cm
import torch
import gc
import numpy as np

def get_response_with_attention(
        svfeye_model: SVFEYModel,
        annotation,
        ic_examples,
        image_folder: str = None,
        conf_threshold: float = 1.00
    ):
    input_image = annotation['input_image']
    if image_folder is not None:
        input_image = os.path.join(image_folder, input_image)
    question = annotation['question']
    options = annotation.get('options', None)

    targets = svfeye_model.generate_visual_cues_using_ic(ic_examples, question)
    targets = [t for t in targets if not include_pronouns(t)]
    answer_type = annotation.get('answer_type', 'free_form')
    image_pil = Image.open(input_image).convert('RGB')
    
    if answer_type == "option_list":
        CONFIDENCE_THRESHOLD = conf_threshold
        initial_images_list = [image_pil] 
        answers = []
        first_option_str = options[0]

        question_input_first = format_question(question, first_option_str)
        first_avg_conf, _, _, _ = svfeye_model.free_form_using_nodes(image_pil, question_input_first, initial_images_list,calculate_confidence=True)
        torch.cuda.empty_cache()
        gc.collect()

        entered_crop_branch = False
        images_list = []
        attention_maps = None
        if first_avg_conf < CONFIDENCE_THRESHOLD:
            print(f"Confidence ({first_avg_conf:.4f}) is below threshold ({CONFIDENCE_THRESHOLD}). Triggering zoom-in...")
            images_list, attention_maps, bboxes, _ = process_attention_to_image_list(svfeye_model, image_pil, targets)
            entered_crop_branch = True
        else:
            print(f"Confidence ({first_avg_conf:.4f}) is sufficient. Using original image.")
            images_list = initial_images_list
            attention_maps = None 

        for option_str in options:
            question_input = format_question(question, option_str)
            final_output = svfeye_model.free_form_using_nodes(image_pil, question_input, images_list, calculate_confidence=False)
            answers.append(final_output)
            torch.cuda.empty_cache()
            gc.collect()
        print(answers)
        return answers,images_list, attention_maps, entered_crop_branch
    else:
        raise NotImplementedError(f"Unsupported answer_type: {answer_type}")


def process_attention_to_image_list(svfeye_model: SVFEYModel, image_pil, targets, include_original=False):
    if not targets:
        images_list = [image_pil]
        attention_maps = []
        bboxes = []
        union_bbox = None
        return images_list, attention_maps, bboxes, union_bbox
    
    bbox_size = 224*2
    question_tpl_3 = "where is the {}"
    general_question = 'Write a general description of the image.'
    general_prompt = f'{general_question} Answer the question using a single word or phrase.'
    
    bboxes = []
    attention_maps = []

    for target in targets:
        question = question_tpl_3.format(target)
        prompt = f'{question} Answer the question using a single word or phrase.'
        att_map = rel_attention_qwen2_5(image_pil, prompt, general_prompt, svfeye_model.model, svfeye_model.processor, target)

        if target.startswith("all "):
            bbox, _ = bbox_from_att_image_nms(att_map, image_pil.size, bbox_size, sum_threshold_ratio=0.7, nms_iou_threshold=0.2)
            bboxes.append(bbox)
        else:
            bbox = bbox_from_att_image_adaptive(att_map, image_pil.size, bbox_size)
            bboxes.append(bbox)

        # 热力图生成
        att = att_map.astype(np.float32)
        att = (att - att.min()) / (att.max() - att.min() + 1e-8)
        cmap = cm.get_cmap("viridis")
        att_color = cmap(att)[:, :, :3]
        att_img = (att_color * 255).astype(np.uint8)
        
        att_pil_low_res = Image.fromarray(att_img)
        grid_heatmap = att_pil_low_res.resize(image_pil.size, Image.NEAREST)
        attention_maps.append(grid_heatmap)

        smooth_heatmap = att_pil_low_res.resize(image_pil.size, Image.BILINEAR)
        alpha = 0.55
        image_for_blending = image_pil.convert('RGB')
        overlay_image = Image.blend(image_for_blending, smooth_heatmap, alpha=alpha)

        draw_overlay = ImageDraw.Draw(overlay_image)
        draw_overlay.rectangle(bbox, outline="red", width=max(1, overlay_image.width // 200))
        attention_maps.append(overlay_image)

    annotated_image = deepcopy(image_pil)
    draw = ImageDraw.Draw(annotated_image)
    for bbox in bboxes:
        draw.rectangle(bbox, outline="red", width=max(1, annotated_image.width // 200))

    images_list = []
    if include_original:
        img0 = image_pil
    else:
        img0 = annotated_image
    images_list.append(img0)

    # 合并bbox并裁剪图像
    union_bbox = union_all_bboxes(bboxes)
    if union_bbox:
        crop_image = image_pil.crop(union_bbox)
        crop_image = svfeye_model.resize_image(crop_image)
        images_list.append(crop_image)
        
    return images_list, attention_maps, bboxes, union_bbox

def format_question(question, option_str):
    return question + '\n' + option_str + 'Answer the option letter directly.'

def format_question_multichoice(question, options):
    ret = question
    for o in options:
        ret += '\n'
        ret += o
    # This prompt is copied from the original paper of MME-RealWorld
    ret += '\nSelect the best answer to the above multiple-choice question based on the image. Respond with only the letter (A, B, C, D, or E) of the correct option.\nThe best answer is:'
    return ret
def format_question_text_match(question):
    """
    为 AOKVQA 数据集格式化问题
    参考 mllms_know 中 Qwen2.5 的简洁风格：不包含选项，只包含问题和指令
    """
    # 参考 mllms_know run.py 第153行
    return f'{question} Answer the question using a single word or phrase.'

def get_direct_response(
        svfeye_model: SVFEYModel,
        annotation,
        image_folder
    ):
    input_image = annotation['input_image']
    if image_folder is not None:
        input_image = os.path.join(image_folder, input_image)
    question = annotation['question']
    options = annotation.get('options', None)

    image_pil = Image.open(input_image).convert('RGB')
    image_list = [image_pil]
    
    answers = []
    for option_str in options:
        question_input = format_question(question, option_str)
        answers.append(svfeye_model.free_form_using_nodes(image_pil, question_input, image_list))
    return answers
