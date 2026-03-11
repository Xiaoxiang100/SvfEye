import torch
from torch.nn import CrossEntropyLoss
from typing import List
from PIL import Image
import numpy as np

import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from svfeye.utils import *
BOX_COLOR = "red"

class SVFEYModelQwenVL:
    def __init__(self, model_path: str, device: str = "cuda:0", torch_dtype=torch.bfloat16, **kwargs) -> None:
        self.device = device
        self.dtype = torch_dtype
        load_kwargs = {}
        load_in_8bit = kwargs.get("load_in_8bit", False)
        if load_in_8bit:
            device_map = "auto"
        else:
            device_map = self.device

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=self.dtype,
            device_map=device_map, 
            **load_kwargs
        )

        max_pixels = 448*4*448*4
        print("max_pixels:", max_pixels)
        self.processor = AutoProcessor.from_pretrained(model_path, max_pixels=max_pixels)
        self.processor.image_processor.size["longest_edge"] = max_pixels
        self.tokenizer = self.processor.tokenizer
        self.bias_value = kwargs.get("bias_value", 0.6)
        print("max_pixels:", max_pixels)
        print("size:", self.processor.image_processor.size)

        self.background_color = tuple(int(x*255) for x in self.processor.image_processor.image_mean)
        self.patch_scale = kwargs.get("patch_scale", None)

        self.init_prompts()

    def init_prompts(self):
        self.prompts = {
            "global":{
                "pre_information": "<image>\n",
            },
            "zoom":{
                "pre_information": f"<image>\nThis is the main image, and the section enclosed by the red rectangle is the focus region.\n<image>\nThis is the zoomed-in view of the focus region.\n",
            },
        }
    
    @torch.no_grad()
    def generate_visual_cues_using_ic(self, ic_examples, question: str, split_tag = r' and |, '):
        ic_question_template = ic_examples["question_template"]
        ic_question_list = ic_examples["question_list"]
        ic_response_list = ic_examples["response_list"]
        
        message = []
        for q, a in zip(ic_question_list, ic_response_list):
            message.extend([
                {"role": "user", "content": [{"type": "text", "text": ic_question_template.format(q)}]},
                {"role": "assistant", "content": [{"type": "text", "text": a}]}
            ])
        message.extend([
            {"role": "user", "content": [{"type": "text", "text": ic_question_template.format(question)}]},
        ])
        texts = [self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)]
        model_inputs = self.processor(
            text=texts,
            images=None,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False)
        model_inputs = model_inputs.to(self.device)
        generated_ids = self.model.generate(**model_inputs, use_cache=True, max_new_tokens=256, do_sample=False)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        targets_list = extract_targets_from_tags(response)
        return targets_list


    def get_prompt_from_qs(self, qs, response=None, show_prompt=False):
        message = []
        message.append({"role": "user", "content": []})
        while "<image>\n" in qs:
            index = qs.find("<image>\n")
            if index == 0:
                message[0]["content"].append({"type": "image"})
                qs = qs[len("<image>\n"):]
            else:
                message[0]["content"].append({"type": "text", "text": qs[:index]})
                qs = qs[index:]
        if len(qs) > 0:
            message[0]["content"].append({"type": "text", "text": qs})
        if response is not None:
            message.append({"role": "assistant", "content": [{"type": "text", "text": response}]})
        texts = [self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True if response is None else False)]
        return texts[0]

    def resize_image(self, image_pil):
        img = deepcopy(image_pil)
        input_size = 448*3
        if img.width > img.height:
            img = img.resize((input_size, int(img.height * input_size / img.width)))
        else:
            img = img.resize((int(img.width * input_size / img.height), input_size))
        return img

    def get_prompt_tag(self, image_list):
        if len(image_list) == 1:
            prompt_tag = "global"
        elif len(image_list) == 2:
            prompt_tag = "zoom"
        else:
            raise ValueError
        return prompt_tag

    @torch.inference_mode()
    def free_form_using_nodes(self, image_pil, question, image_list, return_zoomed_view=False, calculate_confidence=False):
        # --- 这部分代码保持不变 ---
        prompt_tag = self.get_prompt_tag(image_list)
        qs = self.prompts[prompt_tag]["pre_information"] + question
        prompt = self.get_prompt_from_qs(qs)
        model_inputs = self.processor(
            text=[prompt],
            images=image_list,
            return_tensors="pt",
            padding=True,
            padding_side="left",
        )
        model_inputs = model_inputs.to(self.device)
        
        generation_kwargs = {
            "use_cache": True,
            "max_new_tokens": 256,
            "do_sample": False
        }
        if calculate_confidence:
            generation_kwargs['output_scores'] = True
            generation_kwargs['return_dict_in_generate'] = True

        generate_output = self.model.generate(**model_inputs, **generation_kwargs)
        
        if calculate_confidence:
            generated_ids = generate_output.sequences
        else:
            generated_ids = generate_output

        input_ids_len = model_inputs.input_ids.shape[1]
        generated_ids_trimmed = generated_ids[0][input_ids_len:]
        
        outputs_text = self.processor.decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        # --- 代码修改从这里开始 ---

        final_output = outputs_text
            
        # 如果需要计算置信度，则计算并返回四元组
        if calculate_confidence:
            logits_per_step = generate_output.scores
            avg_conf, token_confs, first_conf = self.calculate_token_confidence(logits_per_step, generated_ids_trimmed)
            
            return avg_conf, token_confs, first_conf, final_output
        # 否则，只返回基础输出
        return final_output

    @torch.inference_mode()
    def calculate_token_confidence(self, logits_per_step, generated_tokens):
        token_confs = []
        for logit, token_id in zip(logits_per_step, generated_tokens):
            probs = torch.softmax(logit[0], dim=-1)
            token_confs.append(probs[token_id].item())
        avg_conf = sum(token_confs) / len(token_confs) if token_confs else 0.0
        first_conf = token_confs[0] if token_confs else 0.0
        return avg_conf, token_confs, first_conf