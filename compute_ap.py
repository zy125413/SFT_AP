"""
=============================================================================
Method-Faithful Activation Pattern Extractor (GSM8K)
=============================================================================

Disclaimer & Scope:
1. This script extracts attention head Activation Patterns (AP) for a given 
   checkpoint evaluated on the GSM8K dataset.
2. It requires a local, minimal patch to the `transformers` library to retain 
   attention gradients (see README for injection details).
3. This is a method-faithful release script meant to demonstrate the core 
   AP extraction mechanism described in the paper. It is NOT a full 
   reproduction pipeline for every experiment or figure.

=============================================================================
"""

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
from datasets import load_dataset

# =============================================================================
# [核心算法] 提取并计算当前 Batch 的 Activation Pattern
# =============================================================================
def compute_activation_pattern_batch(model):
    """
    计算当前 batch 的 Activation Pattern。
    数学原理对应论文公式 (2): \Gamma^T * (\partial L / \partial \Gamma)
    """
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    
    target_device = next(model.parameters()).device
    batch_activation_pattern = torch.zeros((num_layers, num_heads), device=target_device)
    
    valid_layers_count = 0
    
    for layer_idx, layer in enumerate(model.model.layers):
        if not hasattr(layer.self_attn, "saved_attn_weights") or layer.self_attn.saved_attn_weights is None:
            continue
            
        attn_weights = layer.self_attn.saved_attn_weights  
        attn_grads = attn_weights.grad                     
        
        if attn_grads is not None:
            valid_layers_count += 1
            # 在序列长度维度上逐元素相乘并求和
            al_lh_batch = torch.sum(attn_weights * attn_grads, dim=(-2, -1)) 
            # 对当前 batch 内的样本求和 (取绝对值量化影响程度)
            batch_activation_pattern[layer_idx] = torch.sum(torch.abs(al_lh_batch), dim=0) 
            
            # 释放显存
            attn_weights.grad = None
            layer.self_attn.saved_attn_weights = None
            
    return batch_activation_pattern, valid_layers_count

# =============================================================================
# [数据处理] 严格的 SFT 数据构造 (Masking Prompt) - 硬编码 GSM8K
# =============================================================================
def prepare_gsm8k_eval_dataloader(tokenizer, split="test", batch_size=4, max_length=1024):
    """
    构建 GSM8K 评估 DataLoader，严格确保 Question 部分不参与 Loss 计算 (label = -100)
    """
    print(f"Loading GSM8K ({split} split)...")
    dataset = load_dataset("gsm8k", "main", split=split)
    
    def tokenize_and_mask(examples):
        input_ids_list = []
        labels_list = []
        attention_mask_list = []
        
        for q, a in zip(examples['question'], examples['answer']):
            prompt = f"Question: {q}\nAnswer: "
            response = f"{a}{tokenizer.eos_token}"
            
            prompt_tokens = tokenizer(prompt, add_special_tokens=True).input_ids
            response_tokens = tokenizer(response, add_special_tokens=False).input_ids
            
            input_ids = prompt_tokens + response_tokens
            labels = [-100] * len(prompt_tokens) + response_tokens
            
            if len(input_ids) > max_length:
                input_ids = input_ids[:max_length]
                labels = labels[:max_length]
                
            pad_len = max_length - len(input_ids)
            input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
            labels = labels + [-100] * pad_len
            attention_mask = [1] * (max_length - pad_len) + [0] * pad_len
            
            input_ids_list.append(input_ids)
            labels_list.append(labels)
            attention_mask_list.append(attention_mask)
            
        return {
            "input_ids": torch.tensor(input_ids_list),
            "labels": torch.tensor(labels_list),
            "attention_mask": torch.tensor(attention_mask_list)
        }

    tokenized_datasets = dataset.map(tokenize_and_mask, batched=True, remove_columns=dataset.column_names)
    tokenized_datasets.set_format("torch")
    
    dataloader = DataLoader(tokenized_datasets, batch_size=batch_size, shuffle=False)
    return dataloader

# =============================================================================
# [主流程] 计算特定 Checkpoint 在 GSM8K 上的 Activation Pattern
# =============================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Extract Activation Pattern for a given checkpoint on GSM8K.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model or checkpoint")
    parser.add_argument("--output_file", type=str, default="ap_matrix_gsm8k.npy", help="Output file path")
    args = parser.parse_args()

    # 强建议：此分析脚本最好在单卡环境下运行，避免设备张量分布不均的问题
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataloader = prepare_gsm8k_eval_dataloader(tokenizer, split="test", batch_size=4)
    
    print(f"Loading model from {args.model_path} onto {device}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, 
        torch_dtype=torch.bfloat16,
    ).to(device)
    
    model.eval() 
    
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    total_activation_pattern = torch.zeros((num_layers, num_heads), device=device)
    total_valid_samples = 0
    
    print("Extracting Activation Patterns on GSM8K...")
    
    for step, batch in enumerate(tqdm(dataloader)):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        # 统计口径说明：
        # dataset-level AP is estimated by averaging per-sample head contributions aggregated within each batch
        total_valid_samples += input_ids.size(0)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        
        loss.backward()
        
        batch_ap, valid_layers = compute_activation_pattern_batch(model)
        
        # [安全拦截] 强化 Patch 检查：如果拿到的有效层数不到一半，说明 Patch 可能失效或只拦截了部分
        if step == 0 and valid_layers < (num_layers // 2):
            raise RuntimeError(
                f"CRITICAL ERROR: 'transformers' patch check failed! "
                f"Only found gradients for {valid_layers}/{num_layers} layers. "
                f"Did you manually inject `attn_weights.retain_grad()` and "
                f"`self.saved_attn_weights = attn_weights` correctly into the modeling code?"
            )
            
        total_activation_pattern += batch_ap
        
        model.zero_grad(set_to_none=True)

    final_activation_pattern = (total_activation_pattern / total_valid_samples).cpu().float().numpy()
    
    np.save(args.output_file, final_activation_pattern)
    print(f"Done. Matrix saved to {args.output_file} (Shape: {final_activation_pattern.shape})")

if __name__ == "__main__":
    main()
