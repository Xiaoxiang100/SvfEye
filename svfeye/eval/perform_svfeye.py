import sys
import os

# 将项目根目录加入环境变量，确保能导入 svfeye 模块
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

from PIL import Image
import argparse
import json
import warnings
warnings.filterwarnings("ignore")
import torch
import time
from tqdm import tqdm

from svfeye.svfeye_model_qwenvl import SVFEYModelQwenVL
from svfeye.svfeye import get_direct_response, get_response_with_attention
import random

import numpy as np
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def get_chunk(lst, n, k):
    """将列表切分为 n 份，返回第 k 份"""
    subarrays = [[] for _ in range(n)]
    for i in range(n):
        subarrays[i] = lst[i::n]
    return subarrays[k]
    
if __name__ == "__main__":
    
    # --- 1. 参数解析 ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--answers-file", type=str, default=None)
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help="Qwen2.5-VL model path")
    parser.add_argument("--annotation_path", type=str, default="svfeye/svfeye_data", help="Path to benchmark annotations")
    parser.add_argument("--benchmark",  type=str, choices=["vstar", "hr-bench_4k", "hr-bench_8k", "mme-realworld", "aokvqa", "docvqa", "pope", "textvqa"], default="aokvqa")
    parser.add_argument("--direct-answer", action="store_true")
    parser.add_argument("--conf-threshold", type=float, default=1.00)
    args = parser.parse_args()

    model_path = args.model_path
    annotation_path = args.annotation_path
    benchmark = args.benchmark

    # 设置输出文件路径
    if args.answers_file is None:
        answers_dir = f"svfeye/eval/answers/{benchmark}"
        answers_dir = os.path.join(answers_dir, os.path.basename(args.model_path))
        os.makedirs(answers_dir, exist_ok=True)
        answer_tag = "svfeye" if not args.direct_answer else "direct_answer"
        args.answers_file = os.path.join(answers_dir, f"{answer_tag}.jsonl")
        print(f"Output file: {args.answers_file}")

    # --- 2. 模型初始化（仅支持 Qwen2.5-VL）---
    print("Initializing Model...")
    if "32b" in model_path.lower():
        kwargs = {"load_in_8bit": True}
    else:
        kwargs = {}
    svfeye_model = SVFEYModelQwenVL(model_path=model_path, device="cuda:0", torch_dtype=torch.bfloat16, patch_scale=1.2, **kwargs)

    # --- 3. 数据加载 --
    ic_examples_path = f"svfeye/ic_examples/{benchmark}.json"
    data_path = os.path.join(annotation_path, f"{benchmark}/annotation_{benchmark}.json")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Annotation file not found: {data_path}")

    with open(data_path, "r", encoding="utf-8") as f:
        m = json.load(f)
    m = get_chunk(m, args.num_chunks, args.chunk_idx)

    # 打开结果文件
    results_file = open(args.answers_file, 'w')

    # --- 4. 初始化统计信息 ---
    total_start_time = time.time()
    time_stats = {
        'total_samples': len(m),
        'total_time': 0.0,
        'avg_time_per_sample': 0.0,
        'sample_times': []
    }
    svfeye_called_count = 0
    skip_crop_after_filter = 0
    crop_count = 0
    no_cropping_count=0
    svfeye_called_count = 0
    # 新增：option_list 分支的裁剪统计
    option_list_crop_count = 0
    option_list_no_crop_count = 0
    print(f"Start inference on {len(m)} samples...")
    with open(ic_examples_path, "r", encoding="utf-8") as f:
        ic_examples = json.load(f)
    
    # --- 5. 主循环 ---
    for idx, annotation in enumerate(tqdm(m), 1):  # idx 从 1 开始编号
        # 记录单个样本开始时间
        sample_start_time = time.time()
        try:
            # svfeye 模式（阈值比较在 get_response_with_attention / svfeye.py 内完成）
            if not args.direct_answer:
                svfeye_called_count += 1
                response, image_list, attention_maps, entered_crop = get_response_with_attention(
                    svfeye_model=svfeye_model,
                    annotation=annotation,
                    ic_examples=ic_examples,
                    image_folder=os.path.join(annotation_path, f"{benchmark}"),
                    conf_threshold=args.conf_threshold
                )
                print(response)

                answer_type = annotation.get('answer_type', 'free_form')
                # 统计 option_list 分支的裁剪情况
                if answer_type == "option_list" and entered_crop is not None:
                    if entered_crop is True:
                        option_list_crop_count += 1
                    elif entered_crop is False:
                        option_list_no_crop_count += 1

                if len(image_list) == 1:
                    no_cropping_count += 1

                if isinstance(response, dict) and response.get("skip_crop"):
                    skip_crop_after_filter += 1
                else:
                    crop_count += 1

            else:
                response = get_direct_response(
                    svfeye_model=svfeye_model,
                    annotation=annotation,
                    image_folder=os.path.join(annotation_path, f"{benchmark}"),
                )

            # B. 记录结果
            sample_time = time.time() - sample_start_time
            
            annotation['output'] = response
            
            # 写入结果
            results_file.write(json.dumps(annotation) + "\n")
            results_file.flush() # 强制刷新，防止中断丢失数据

            # 更新统计
            time_stats['sample_times'].append({
                'idx': idx,
                'time': round(sample_time, 3),
                'image': annotation.get('input_image', '')
            })

        # C. 异常处理
        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            print(f"\n[OOM Error] Sample {idx} failed. Image: {annotation.get('input_image', 'N/A')}")
            
            annotation['output'] = "[OOM_ERROR] Out of memory"
            annotation['error'] = "OOM"
            results_file.write(json.dumps(annotation) + "\n")
            results_file.flush()
            continue

        except Exception as e:
            torch.cuda.empty_cache()
            print(f"\n[Error] Sample {idx} failed. Error: {str(e)}")
            
            annotation['output'] = f"[ERROR] {type(e).__name__}: {str(e)}"
            annotation['error'] = type(e).__name__
            results_file.write(json.dumps(annotation) + "\n")
            results_file.flush()
            continue

    # --- 6. 结束清理与统计 ---
    results_file.close()

    # 计算总耗时
    total_time = time.time() - total_start_time
    time_stats['total_time'] = round(total_time, 3)
    time_stats['avg_time_per_sample'] = round(total_time / len(m), 3) if len(m) > 0 else 0.0

    print("\n" + "="*60)
    print("Inference Statistics Summary:")
    print("="*60)
    print(f"Total Samples: {time_stats['total_samples']}")
    print(f"Total Time   : {time_stats['total_time']:.3f} s ({time_stats['total_time']/60:.2f} min)")
    print(f"Avg Time/Item: {time_stats['avg_time_per_sample']:.3f} s")
    print(f"Entered svfeye pipeline : {svfeye_called_count}")
    print(f"Skipped crop after filter: {skip_crop_after_filter}")
    print(f"Crop pipeline runs    : {crop_count}")
    if time_stats['sample_times']:
        times = [s['time'] for s in time_stats['sample_times']]
        print(f"Fastest      : {min(times):.3f} s")
        print(f"Slowest      : {max(times):.3f} s")
    print("="*60)
    print("no_cropping_count",no_cropping_count)