# SvfEye: A Semantic–Visual Fusion Framework with Multi-Scale Visual Context for Multimodal Reasoning

[![Paper](https://img.shields.io/badge/paper-arXiv-red.svg)](https://arxiv.org/abs/YOUR_PAPER_ID)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**SvfEye** is a novel, training-free framework for adaptive visual–semantic fusion. It addresses two critical limitations of existing training-free approaches: (1) indiscriminate extraction and fusion of local visual regions causing computational redundancy and perceptual noise, and (2) drift between semantic intent and visual attention preventing accurate localization of user-focused regions. SvfEye follows a two-stage pipeline: a **Confidence-based Decision** module determines whether additional local visual information is needed, and a **Semantic-attention Fusion** module identifies where to extract informative local regions. This design enables adaptive global-local visual information fusion without additional training.

---

## 📖 Core Idea

Multimodal Large Language Models (MLLMs) often struggle to accurately perceive fine-grained visual details, particularly in scenarios involving tiny or visually subtle targets. This challenge can be addressed through **semantic–visual information fusion**, which integrates global image context with fine-grained local evidence to achieve effective multi-scale visual understanding.

The "Thinking with Images" paradigm enables models to actively acquire high-resolution visual evidence by zooming or cropping image regions and fusing these local details with global context during reasoning. While training-based approaches have demonstrated effectiveness, they typically require extensive computational resources and large-scale task-specific data. Consequently, lightweight training-free methods have been proposed as a practical alternative.

However, existing training-free approaches still suffer from two critical limitations:

1.  **Computational Redundancy:** They indiscriminately extract and fuse local visual regions for all input instances regardless of actual necessity, introducing unnecessary computational overhead and perceptual noise.
2.  **Semantic–Attention Drift:** They exhibit a drift between semantic intent and visual attention, which prevents accurate localization of user-focused regions.

SvfEye addresses these challenges with a two-stage pipeline:

1.  **Confidence-based Decision:** Dynamically determines whether additional local visual information is needed based on the model's initial confidence.
2.  **Semantic-attention Fusion:** Uses semantic-guided target extraction combined with attention-based localization to accurately identify where to extract informative local regions.

*   **[Insert SvfEye Core Concept Figure Here (Corresponding to Paper Figure 1)]*
    *(Caption: SvfEye enables MLLMs to intelligently determine when and where more careful observation of image details is needed based on task requirements.)*

## ✨ Features

*   **Training-Free, Plug-and-Play:** As a lightweight module, easily integrable into various mainstream MLLMs (e.g., LLaVA, Qwen) without any additional training.
*   **Excellent Performance:** Achieves significant performance improvements on multiple visual reasoning benchmarks (especially on high-resolution datasets V*-Bench and HR-Bench).
*   **Efficient Inference:** Achieves approximately **4.0×** inference speedup compared to SOTA methods like ZoomEye, while ensuring higher accuracy.
*   **Strong Generalization:** Demonstrates strong generalization ability and stability across different MLLM architectures.

## 📊 Main Results

We conducted comprehensive evaluations of SvfEye on multiple benchmarks. Below is the performance comparison with state-of-the-art methods on LLaVA-1.5-7B and Qwen2.5VL-3B models:

| Model | Method | AOKVQA | POPE | V* Bench | HR-Bench 4K | HR-Bench 8K |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| LLaVA-1.5-7B | Baseline | 71.00 | 86.98 | 48.68 | 36.13 | 32.13 |
| | **Ours (SvfEye)** | **72.90** | 87.37 | **62.80** | **47.38** | **42.00** |
| | *Improvement (Δ)* | *+1.90* | *+0.39* | *+14.12* | *+11.25* | *+9.87* |
| Qwen2.5VL-3B | Baseline | 71.44 | 87.20 | 75.90 | 67.50 | 58.88 |
| | **Ours (SvfEye)** | **73.10** | 89.12 | **85.86** | **71.75** | **70.00** |
| | *Improvement (Δ)* | *+1.66* | *+1.92* | *+9.96* | *+4.25* | *+11.12* |

*   **[Insert SvfEye Workflow Figure Here (Corresponding to Paper Figure 4)]*
    *(Caption: The overall workflow of SvfEye.)*

## 🚀 Getting Started

### 1. Environment Installation

Clone the repository and install required dependencies. We recommend using conda to create a virtual environment.

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/SvfEye.git
cd SvfEye

# Install dependencies
pip install -r requirements.txt

### 2. Run Evaluation

**HR-Bench 4K Evaluation:**
```bash
# Run SvfEye mode (with cropping)
bash perform_svfeye_4k.sh

# Or run Baseline mode (direct answer, no cropping)
bash perform_svfeye_4k.sh direct
```

**HR-Bench 8K Evaluation:**
```bash
# Run SvfEye mode (with cropping)
bash perform_svfeye_8k.sh

# Or run Baseline mode (direct answer, no cropping)
bash perform_svfeye_8k.sh direct
```

**Set Confidence Threshold (Optional):**
```bash
# Default threshold is 1.00, can be modified via environment variable
CONF_THRESHOLD=0.95 bash perform_svfeye_4k.sh
```

### 3. Get Evaluation Results

After evaluation completes, use the evaluation script to calculate scores:

```bash
# HR-Bench 4K evaluation
python svfeye/eval/eval_results_hr-bench.py --answers-file svfeye/eval/answers/hr-bench_4k/Qwen2.5-VL-3B-Instruct/

# HR-Bench 8K evaluation
python svfeye/eval/eval_results_hr-bench.py --answers-file svfeye/eval/answers/hr-bench_8k/Qwen2.5-VL-3B-Instruct/

```

## 📁 Project Structure

```
svfeye/
├── svfeye/
│   ├── svfeye.py              # Core inference logic
│   ├── svfeye_model.py        # Model wrapper
│   ├── svfeye_model_qwenvl.py # Qwen2.5-VL model interface
│   ├── utils.py                 # Utility functions
│   ├── qwen2_5_methods.py       # Qwen-specific methods
│   ├── eval/
│   │   ├── perform_svfeye.py # Evaluation inference script
│   │   ├── eval_results_hr-bench.py   # HR-Bench evaluation script
│   │   └── eval_results_vstar.py       # V* Bench evaluation script
│   └── ic_examples/             # In-context learning examples
├── perform_svfeye_4k.sh       # HR-Bench 4K evaluation script
├── perform_svfeye_8k.sh        # HR-Bench 8K evaluation script
├── requirements.txt             # Project dependencies
└── README.md                    # Project documentation
```

## 🤝 Citation

If you use SvfEye in your research, please cite our paper:

```bibtex
@article{svfeye2024,
  title={SvfEye: A Semantic–Visual Fusion Framework with Multi-Scale Visual Context for Multimodal Reasoning},
  author={},
  journal={arXiv preprint arXiv:},
  year={2024}
}
```

## 📧 Contact

For questions or suggestions, please submit an Issue or contact the authors.
