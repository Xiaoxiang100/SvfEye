import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from skimage.measure import block_reduce
from svfeye.utils import *

# currently select 22 but feel free to try other layers
ATT_LAYER = 22

def rel_attention_qwen2_5(image, prompt, general_prompt, model, processor, target=None):
    """
    计算相对注意力图。
    - 'att' 使用 target 最后一个 token 作为 query。
    - 'general_att' 使用 general_prompt 的最后一个 token 作为 query。
    """
    image_str = encode_base64(image)

    messages = [{"role": "user", "content": [
        {"type": "image", "image": f'data:image;base64,{image_str}'},
        {"type": "text", "text": prompt}
    ]}]

    general_messages = [{"role": "user", "content": [
        {"type": "image", "image": f'data:image;base64,{image_str}'},
        {"type": "text", "text": general_prompt}
    ]}]
    
    inputs = prepare_qwen2_5_input(messages, processor).to(model.device, torch.bfloat16)
    #general_inputs = prepare_qwen2_5_input(general_messages, processor).to(model.device, torch.bfloat16)

    att_shape = (inputs['image_grid_thw'][0, 1:] / 2).cpu().numpy().astype(int).tolist()
    vision_start_token_id = processor.tokenizer.convert_tokens_to_ids('<|vision_start|>')
    vision_end_token_id = processor.tokenizer.convert_tokens_to_ids('<|vision_end|>')

    input_ids = inputs['input_ids'][0].tolist()
    pos = input_ids.index(vision_start_token_id) + 1
    pos_end = input_ids.index(vision_end_token_id)

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
        #general_outputs = model(**general_inputs, output_attentions=True)

    # === 查找 target 对应的 token，并默认选择最后一个 ===
    q_indices_for_att = [-1]  # 默认回退到整个 prompt 的最后一个 token
    if target is not None:
        
        def find_subseq(seq, subseq):
            L, M = len(seq), len(subseq)
            if M == 0: return -1
            for i in range(L - M + 1):
                if seq[i:i+M] == subseq:
                    return i
            return -1

        # 尝试 1: 直接对 target 分词并查找
        target_ids = processor.tokenizer(target, add_special_tokens=False).input_ids
        start_idx = find_subseq(input_ids, target_ids)
        
        # 尝试 2: 如果找不到，在 target 前面加上一个空格再试
        if start_idx == -1:
            target_with_space = " " + target
            target_ids_with_space = processor.tokenizer(target_with_space, add_special_tokens=False).input_ids
            start_idx = find_subseq(input_ids, target_ids_with_space)
            if start_idx != -1:
                target_ids = target_ids_with_space
        
        # 如果找到了
        if start_idx != -1:
            all_indices = list(range(start_idx, start_idx + len(target_ids)))
            # 直接选择最后一个 token 的索引
            q_indices_for_att = [all_indices[-1]]
            #print(f"Found target '{target}'. Using LAST token at index: {q_indices_for_att}")
        else:
            print(f"Warning: Target '{target}' not found. Falling back to the last token of the prompt.")

    # --- 计算注意力 --
    att_tensor = outputs['attentions'][ATT_LAYER][0]
    att = att_tensor[:, q_indices_for_att, pos:pos_end].mean(dim=(0, 1)).to(torch.float32).detach().cpu().numpy()

    att_map = att
    att_map = att_map.reshape(att_shape)
    return att_map
