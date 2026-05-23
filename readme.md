# SPARD: Defending Harmful Fine-Tuning Attack via Safety Projection with Relevance-Diversity Data Selection

This is the official code for the paper:

**SPARD: Defending Harmful Fine-Tuning Attack via Safety Projection with Relevance-Diversity Data Selection**

*Shuhao Chen, Weisen Jiang, Yeqi Gong, Shengda Luo, Chengxiang Zhuo, Zhang Li, James Kwok, Yu Zhang*

ICML 2026

## Overview

SPARD defends safety-aligned LLMs against harmful fine-tuning attacks through two complementary components:

- **SPAG (Safety-Projected Alternating Gradient):** An optimization strategy that alternates between utility-driven updates and explicit safety projections, enforcing safety constraints in closed form during training.
- **Relevance-Diversity DPP:** A data selection method based on Determinantal Point Processes that curates compact safety data subsets balancing task relevance and behavioral diversity.

## Requirements

- Python >= 3.10
- PyTorch >= 2.0
- transformers
- peft
- datasets
- accelerate
- openai (for safety evaluation)
- wandb (optional, for logging)
- numpy, pandas, tqdm

Install dependencies:

```bash
pip install torch transformers peft datasets accelerate openai wandb numpy pandas tqdm
```

## Quick Start

### 1. Setup

Edit the shell script `scripts/finetune/Qwen2.5-7B-Instruct/SPAG_single_mix_metagsm8k_safe_MergeData.sh` and set the following variables:

```bash
MODEL_PATH="path/to/Qwen2.5-7B-Instruct"   # Path to the base model
OUTPUT_DIR="path/to/output"                  # Where to save the fine-tuned model
API_KEY="your-openai-api-key"                # OpenAI API key for safety evaluation
```

### 2. Training with SPARD

Run SPARD with the default hyperparameters under the BeaverTails attack:

```bash
cd scripts/finetune

bash Qwen2.5-7B-Instruct/SPAG_single_mix_metagsm8k_safe_MergeData.sh \
    dpp 0.03 False 43 5e-5 10 1 beaverTails 0.10 4 0.2 0.1 1
```

The arguments correspond to:

| Argument | Value | Description |
|----------|-------|-------------|
| `SAFE_SAMPLE_METHOD` | `dpp` | Safety data selection method (`dpp`, `random`) |
| `SAFE_SAMPLE_RATIO` | `0.03` | Ratio of safety data relative to fine-tuning data (p) |
| `SELECT_BOTTOM` | `False` | Whether to select lowest-similarity samples |
| `SEED` | `43` | Random seed |
| `LR` | `5e-5` | Learning rate |
| `EPOCHS` | `10` | Number of training epochs |
| `SAFE_BATCH_SIZE` | `1` | Safety mini-batch size for projection |
| `ATTACK_DATASET_NAME` | `beaverTails` | Attack dataset name |
| `HARMFUL_RATIO` | `0.10` | Ratio of harmful samples injected |
| `POWER` | `4` | Relevance exponent β |
| `SAFE_TAU` | `0.2` | Safety threshold τ |
| `SAFE_TOLERANCE` | `0.1` | Safety tolerance δ |
| `ALPHA_MAX` | `1` | Trust-region radius η_safe |

To run all four attacks sequentially:

```bash
bash Qwen2.5-7B-Instruct/auto_spag_single_safe_MergeData.sh
```

### 3. Evaluation

The training script automatically runs evaluation after training. To evaluate separately:

**Utility (GSM8K):**

```bash
accelerate launch generate_responses.py \
    --base_model path/to/Qwen2.5-7B-Instruct \
    --lora_model path/to/output/final_model \
    --apply_chat_template \
    --instruction_type "alpaca_qwen_gsm8k" \
    --dataset_name "datasets/gsm8k_meta_math_merged" \
    --output_file path/to/output/vali_gsm8k.json

python evaluate/eval_gsm8k.py \
    --input_path path/to/output/vali_gsm8k.json \
    --output_path path/to/output
```

**Safety (ASR/HS):**

```bash
python generate_responses.py \
    --base_model path/to/Qwen2.5-7B-Instruct \
    --lora_model path/to/output/final_model \
    --dataset_name "datasets/safe_data/eval/beaverTails_test.jsonl" \
    --output_file path/to/output/vali_beavertails.json \
    --instruction_type alpaca \
    --dataset_split train \
    --apply_chat_template

python evaluate/eval_harmful.py \
    --input_path path/to/output/vali_beavertails.json \
    --output_path path/to/output \
    --threshold 3 \
    --api_key your-openai-api-key \
    --num_processes 8
```

## Key Hyperparameters

The following default hyperparameters are used consistently across all experiments without per-task tuning:

| Hyperparameter | Default | Description |
|----------------|---------|-------------|
| τ (safe_tau) | 0.2 | Safety threshold; can be set by referencing the pretrained model's safety loss |
| η_safe (alpha_max) | = η_ft | Trust-region radius; set equal to the fine-tuning learning rate |
| β (power) | 4 | Relevance exponent in the DPP kernel; robust across [4, 10] |
| p (safe_sample_ratio) | 0.03 | Ratio of safety data; robust across [0.03, 0.1] |

## Datasets

All datasets are included under `datasets/`:

**Safety corpora** (used for both attack simulation and defense):
- **BeaverTails**: Safety-related QA pairs
- **I-BeaverTails**: Instruction-format version of BeaverTails
- **LatHarmful**: Instructions with paired harmful/safe completions
- **Q-LatHarmful**: QA-format version of LatHarmful

**Utility tasks:**
- **GSM8K + MetaMath**: Grade-school math word problems
- **OpenBookQA**: Science QA requiring factual reasoning

## Citation

```bibtex
@inproceedings{chen2026spard,
    title={SPARD: Defending Harmful Fine-Tuning Attack via Safety Projection with Relevance-Diversity Data Selection},
    author={Shuhao Chen and Weisen Jiang and Yeqi Gong and Shengda Luo and Chengxiang Zhuo and Zhang Li and James Kwok and Yu Zhang},
    booktitle={International Conference on Machine Learning},
    year={2026}
}
```