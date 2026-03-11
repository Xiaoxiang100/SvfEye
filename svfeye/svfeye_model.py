import torch
from torch.nn import CrossEntropyLoss
from abc import ABC, abstractmethod
from typing import List
from PIL import Image
import numpy as np

from transformers import AutoTokenizer
from llava.model.language_model.llava_qwen import LlavaQwenForCausalLM
from llava.utils import disable_torch_init
from svfeye.utils import *

class SVFEYModel(ABC):
    def __init__(self, model_path: str, conv_type: str = "qwen_1_5", device: str = "cuda:0", torch_dtype=torch.float16, attn_implementation="flash_attention_2", padding_side="left", **kwargs) -> None:
        disable_torch_init()
        self.device = device
        self.dtype = torch_dtype
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, padding_size=padding_side)
        self.tokenizer.padding_side = padding_side
        print("padding side:", self.tokenizer.padding_side)
        
        if "qwen" in model_path:
            self.model = LlavaQwenForCausalLM.from_pretrained(
                model_path,
                low_cpu_mem_usage=True,
                torch_dtype=self.dtype,
                attn_implementation=attn_implementation,
            )
        else:
            raise ValueError("This class is specifically designed for Qwen models. Only models with 'qwen' in the name are supported.")
        print("model:", type(self.model))
        self.model.config.tokenizer_padding_side = padding_side
        self.model.to(self.device)

        vision_tower = self.model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model(device_map=self.device)
            vision_tower.to(device=self.device, dtype=torch.float16)
        self.image_processor = vision_tower.image_processor

        self.conv_type = conv_type
        self.bias_value = kwargs.get("bias_value", 0.2)
        print("bias_value:", self.bias_value)

        self.input_size = (self.image_processor.crop_size['width'], self.image_processor.crop_size['height'])
        self.background_color = tuple(int(x*255) for x in self.image_processor.image_mean)
        print("input size:", self.input_size)
        print("background color:", self.background_color)

        self.patch_scale = kwargs.get("patch_scale", None)
        print("patch scale:", self.patch_scale)

        self.init_prompts()

    @abstractmethod
    def init_prompts(self):
        pass
    
    @abstractmethod
    def get_prompt_tag(self, image_list: List[Image.Image]):
        pass

    @abstractmethod
    def process_image_list_to_tensor(self, image_list: List[Image.Image]):
        pass
    
    @abstractmethod
    def get_prompt_from_qs(self, qs, response=None, show_prompt=False):
        pass
    
    @abstractmethod
    def free_form_using_nodes(self, image_pil, question, image_list, return_zoomed_view=False, calculate_confidence=False):
        pass
    
    @abstractmethod
    def generate_visual_cues_using_ic(self, ic_examples, question: str, split_tag = r' and |, '):
        pass

