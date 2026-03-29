# Activation Pattern Extraction for LLMs

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Paper](https://img.shields.io/badge/Paper-ACL%202025-red)](https://aclanthology.org/2025.acl-long.831.pdf)

This repository provides the official minimal, method-faithful implementation for the ACL 2025 paper: **[Analyzing the Rapid Generalization of SFT via the Perspective of Attention](https://aclanthology.org/2025.acl-long.831.pdf)**.

It contains a standalone analytical probe (`compute_ap.py`) designed to extract and calculate the **Activation Patterns (AP)** of attention heads for a given Large Language Model (LLM) checkpoint evaluated on the GSM8K dataset.

## 📌 Scope & Disclaimer

* **What this is:** A lightweight, method-faithful extraction script to calculate the $L \times H$ Activation Pattern matrix (based on gradient sensitivity) as described in Equation 2 of our paper.
* **What this is NOT:** A full training pipeline. This repository does not contain the code to fine-tune models or reproduce every figure in the paper. It is designed for researchers who want to apply our AP extraction method to their own saved checkpoints.
* **Dataset:** Currently hardcoded and optimized for **GSM8K**.
* **Environment:** Tested primarily in a single-GPU setting to ensure deterministic gradient extraction.

## ⚙️ Requirements & Installation

This codebase requires **Python 3.10**. We recommend setting up a virtual environment:

```bash
conda create -n sft_ap python=3.10 -y
conda activate sft_ap
```

Install the required dependencies. PyTorch should be installed according to your specific CUDA version:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # Adjust CUDA version as needed
pip install transformers datasets tqdm numpy scipy scikit-learn
```

## 🚨 CRITICAL: The `transformers` Patch

To accurately capture the gradients of the attention scores without relying on complex and unstable PyTorch hooks in distributed environments, **you must apply a minimal 2-line patch to your local `transformers` library.**

If you are evaluating **Llama** models, locate your local installation of `transformers` (e.g., `path/to/env/lib/python3.10/site-packages/transformers/models/llama/modeling_llama.py`).

Find the `LlamaAttention` forward pass where `attn_weights` are passed through the `softmax` function, and inject the following two lines:

```python
# ... existing code in modeling_llama.py ...
attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

# ==========================================
# INJECT THESE TWO LINES FOR AP EXTRACTION:
attn_weights.retain_grad()
self.saved_attn_weights = attn_weights
# ==========================================

attn_output = torch.matmul(attn_weights, value_states)
# ... existing code ...
```

*(Note: The script `compute_ap.py` includes a strict runtime check. It will safely fail with a `RuntimeError` on the very first step if this patch is missing or incomplete, preventing you from wasting hours generating empty matrices.)*

## 🚀 Usage

Once the environment is set up and the `transformers` library is patched, you can run the extraction script on any saved checkpoint:

```bash
python compute_ap.py \
    --model_path /path/to/your/local/checkpoint \
    --output_file ap_matrix_gsm8k_step500.npy
```

### Output
The script will output an `.npy` file containing the $L \times H$ Activation Pattern matrix. You can load this matrix using `numpy` to perform further statistical analysis (e.g., Gini coefficient, MSE, or visualizing heatmaps) as demonstrated in the paper.

```python
import numpy as np

# Load the generated Activation Pattern matrix
ap_matrix = np.load("ap_matrix_gsm8k_step500.npy")
print(f"Matrix Shape (Layers, Heads): {ap_matrix.shape}")
```

## 📖 Citation

If you find this code or our paper useful in your research, please consider citing our work:

```bibtex
@inproceedings{zhao2025analyzing,
  title={Analyzing the Rapid Generalization of SFT via the Perspective of Attention},
  author={Zhao, Yang and Du, Li and Ding, Xiao and Xiong, Kai and Liu, Ting and Qin, Bing},
  booktitle={Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={16980--16992},
  year={2025},
  url={https://aclanthology.org/2025.acl-long.831.pdf}
}
