import re
from PIL import Image, ImageDraw
import numpy as np
import spacy
import torchvision.transforms.functional as TF
from scipy.ndimage import median_filter
from skimage.measure import block_reduce
from qwen_vl_utils import process_vision_info
from io import BytesIO
from copy import deepcopy
import base64
nlp = spacy.load("en_core_web_sm")

def extract_targets_from_tags(response: str):
    """
    使用正则表达式从 <target> 标签中提取目标列表。
    这种方法对格式错误（如多余的空格或文本）有更好的容错性。
    """
    # 使用 re.search 查找被 <target> 和 </target> 包围的内容
    # re.IGNORECASE 使得 <target> 或 <Target> 都能被匹配
    # (.*?) 是一个非贪婪捕获组，它会捕获两个标签之间的所有文本
    match = re.search(r"<target>(.*?)</target>", response, re.IGNORECASE | re.DOTALL)
    
    if match:
        # group(1) 获取捕获组的内容 (即标签内的文本)
        targets_str = match.group(1)
        # 按逗号分割字符串，并使用 strip() 清除每个目标周围可能存在的空格
        targets_list = [target.strip() for target in targets_str.split(',')]
        # 过滤掉可能因多余逗号产生的空字符串
        return [target for target in targets_list if target]
    else:
        # 如果在响应中没有找到 <target> 标签，则返回一个空列表
        print("Warning: Could not find <target> tag in the response.")
        return []

def merge_bboxes(bbox1, bbox2):
    return (
        min(bbox1[0], bbox2[0]),
        min(bbox1[1], bbox2[1]),
        max(bbox1[2], bbox2[2]),
        max(bbox1[3], bbox2[3])
    )

def union_all_bboxes(bboxes):
    if len(bboxes) == 0:
        return None
    ret = bboxes[0]
    for bbox in bboxes[1:]:
        ret = merge_bboxes(ret, bbox)
    return ret

# For the visual cues like "man and his bag", we should remove the pronoun "his bag"
def include_pronouns(text):
    doc = nlp(text)
    for token in doc:
        if token.pos_ == 'PRON':
            return True
    return False


def encode_base64(image):
    """
    Encodes a PIL image to a base64 string.
    """
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def prepare_qwen2_5_input(messages, processor):

    """
    Prepare the input for Qwen2.5VL.
    """

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")

    return inputs


def calculate_iou(box1, box2):
    """计算两个边界框的IoU"""
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    
    iou = inter_area / union_area if union_area > 0 else 0
    return iou

def non_max_suppression(boxes, scores, iou_threshold):
    """执行非极大值抑制 (NMS)"""
    if len(boxes) == 0:
        return []

    sorted_indices = np.argsort(scores)[::-1]
    
    keep_boxes = []
    while len(sorted_indices) > 0:
        current_index = sorted_indices[0]
        keep_boxes.append(boxes[current_index])
        
        remaining_indices = sorted_indices[1:]
        
        ious = np.array([calculate_iou(boxes[current_index], boxes[i]) for i in remaining_indices])
        
        low_iou_indices = np.where(ious < iou_threshold)[0]
        sorted_indices = remaining_indices[low_iou_indices]
        
    return keep_boxes



def bbox_from_att_image_nms(att_map, image_size, bbox_size, sum_threshold_ratio, nms_iou_threshold):
    """
    寻找、去重并合并注意力图中的多个边界框。
    """
    
    # 步骤 1: 滑动窗口计算注意力总和
    block_size = image_size[0] / att_map.shape[1], image_size[1] / att_map.shape[0]
    block_num = min(int(bbox_size/block_size[0]), att_map.shape[1]), min(int(bbox_size/block_size[1]), att_map.shape[0])

    if att_map.shape[1] < block_num[0] or att_map.shape[0] < block_num[1]:
        return 0, 0, image_size[0], image_size[1]

    sliding_att = np.zeros((att_map.shape[0]-block_num[1]+1, att_map.shape[1]-block_num[0]+1))
    for x in range(att_map.shape[1]-block_num[0]+1): 
        for y in range(att_map.shape[0]-block_num[1]+1): 
            sliding_att[y, x] = att_map[y:y+block_num[1], x:x+block_num[0]].sum()

    # 步骤 2: 根据阈值筛选高注意力候选框
    max_att = np.max(sliding_att)
    threshold = max_att * sum_threshold_ratio
    
    candidate_indices = np.where(sliding_att >= threshold)
    
    candidate_boxes = []
    candidate_scores = []
    
    for y, x in zip(*candidate_indices):
        x1 = int(x * block_size[0])
        y1 = int(y * block_size[1])
        x2 = int((x + block_num[0]) * block_size[0])
        y2 = int((y + block_num[1]) * block_size[1])
        candidate_boxes.append((x1, y1, x2, y2))
        candidate_scores.append(sliding_att[y, x])
        
    if not candidate_boxes:
        return 0, 0, 0, 0

    # 步骤 3: 执行NMS去除重叠框
    boxes_after_nms = non_max_suppression(candidate_boxes, candidate_scores, nms_iou_threshold)

    if not boxes_after_nms:
        return 0, 0, 0, 0

    # 步骤 4: 合并所有保留的框
    x1, y1, x2, y2 = boxes_after_nms[0]

    for box in boxes_after_nms[1:]:
        x1 = min(x1, box[0])
        y1 = min(y1, box[1])
        x2 = max(x2, box[2])
        y2 = max(y2, box[3])

    merged_box = (x1, y1, x2, y2)
    return merged_box,boxes_after_nms


def bbox_from_att_image_adaptive(att_map, image_size, bbox_size=336):
    """
    Generates an adaptive bounding box for original image from an attention map.
    
    This function finds the region with the highest attention in the attention map
    and creates a bounding box around it. It tries different crop ratios and selects
    the one that produces the sharpest attention difference.
    
    Args:
        att_map: A 2D numpy array representing the attention map (e.g., 24x24 for LLaVA or 16x16 for BLIP)
        image_size: Tuple of (width, height) of the original image
        bbox_size: Base size for the bounding box (default: 336)
        
    Returns:
        tuple: (x1, y1, x2, y2) coordinates of the bounding box in the original image
    """

    # the ratios corresponds to the bounding box we are going to crop the image
    #ratios = [1, 1.2, 1.4, 1.6, 1.8, 2]
    ratios = [1, 1.2, 1.4, 1.6, 1.8, 2, 4, 6]
    max_att_poses = []
    differences = []
    block_nums = []

    for ratio in ratios:
        # perform a bbox_size*r width and bbox_size*r height crop, where bbox_size is the size of the model's original image input resolution. (336 for LLaVA, 224 for BLIP)

        # the size of each block in the attention map, in the original image
        block_size = image_size[0] / att_map.shape[1], image_size[1] / att_map.shape[0]

        # if I want a bbox_size*r width and bbox_size*r height crop from the original image, the number of blocks I need (x, y)
        block_num = min(int(bbox_size*ratio/block_size[0]), att_map.shape[1]), min(int(bbox_size*ratio/block_size[1]), att_map.shape[0])
        if att_map.shape[1]-block_num[0] < 1 and att_map.shape[0]-block_num[1] < 1:
            if ratio == 1:
                return 0, 0, image_size[0], image_size[1]
            else:
                continue
        block_nums.append((block_num[0], block_num[1]))
        
        # attention aggregation map
        sliding_att = np.zeros((att_map.shape[0]-block_num[1]+1, att_map.shape[1]-block_num[0]+1))
        max_att = -np.inf
        max_att_pos = (0, 0)

        # sliding window to find the block with the highest attention
        for x in range(att_map.shape[1]-block_num[0]+1): 
            for y in range(att_map.shape[0]-block_num[1]+1): 
                att = att_map[y:y+block_num[1], x:x+block_num[0]].sum()
                sliding_att[y, x] = att
                if att > max_att:
                    max_att = att
                    max_att_pos = (x, y)
        
        # we have the position of max attention, we can calculate the difference between the max attention and the average of its adjacent attentions, to see if it is sharp enough, the more difference, the sharper
        # we choose the best ratio r according to their attention difference
        adjcent_atts = []
        if max_att_pos[0] > 0:
            adjcent_atts.append(sliding_att[max_att_pos[1], max_att_pos[0]-1])
        if max_att_pos[0] < sliding_att.shape[1]-1:
            adjcent_atts.append(sliding_att[max_att_pos[1], max_att_pos[0]+1])
        if max_att_pos[1] > 0:
            adjcent_atts.append(sliding_att[max_att_pos[1]-1, max_att_pos[0]])
        if max_att_pos[1] < sliding_att.shape[0]-1:
            adjcent_atts.append(sliding_att[max_att_pos[1]+1, max_att_pos[0]])
        difference = (max_att - np.mean(adjcent_atts)) / (block_num[0] * block_num[1])
        differences.append(difference)
        max_att_poses.append(max_att_pos)
    max_att_pos = max_att_poses[np.argmax(differences)]
    block_num = block_nums[np.argmax(differences)]
    selected_bbox_size = bbox_size * ratios[np.argmax(differences)]
    
    x_center = int(max_att_pos[0] * block_size[0] + block_size[0] * block_num[0] / 2)
    y_center = int(max_att_pos[1] * block_size[1] + block_size[1] * block_num[1] / 2)
    
    x_center = selected_bbox_size//2 if x_center < selected_bbox_size//2 else x_center
    y_center = selected_bbox_size//2 if y_center < selected_bbox_size//2 else y_center
    x_center = image_size[0] - selected_bbox_size//2 if x_center > image_size[0] - selected_bbox_size//2 else x_center
    y_center = image_size[1] - selected_bbox_size//2 if y_center > image_size[1] - selected_bbox_size//2 else y_center

    x1 = max(0, x_center - selected_bbox_size//2)
    y1 = max(0, y_center - selected_bbox_size//2)
    x2 = min(image_size[0], x_center + selected_bbox_size//2)
    y2 = min(image_size[1], y_center + selected_bbox_size//2)

    return x1, y1, x2, y2
