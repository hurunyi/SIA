# Inference-time Alignment via Sparse Junction Steering
Official implementation of [Inference-time Alignment via Sparse Junction Steering](https://arxiv.org/abs/2602.21215).


## Table of Contents

- [Model Download](#model-download)
- [Generation](#generation)
- [Evaluation](#evaluation)
- [Value Model Training](#value-model-training)

---

## Model Download

### 1. LLM

Download the base or instruction-tuned LLMs you want to use for generation:

- **Qwen3**: [https://huggingface.co/collections/Qwen/qwen3](https://huggingface.co/collections/Qwen/qwen3)
- **Llama-3.1**: [https://huggingface.co/collections/meta-llama/llama-31](https://huggingface.co/collections/meta-llama/llama-31)
- **Llama-3.2**: [https://huggingface.co/collections/meta-llama/llama-32](https://huggingface.co/collections/meta-llama/llama-32)

### 2. Value Model

We provide our trained Value Model checkpoints for the following backbone models:

| Backbone | Link |
|---|---|
| Qwen3-0.6B | _coming soon_ |
| Qwen3-1.7B | _coming soon_ |
| Qwen3-4B | _coming soon_ |
| Llama-3.2-1B | _coming soon_ |
| Llama-3.2-3B | _coming soon_ |

Each checkpoint consists of LoRA weights and a token reward head, stored alongside a `model_config.json` that specifies the base model path and LoRA hyperparameters.

---

## Generation

### Quick start with the script

```bash
bash scripts/generate.sh
```

### Manual command

```bash
python3 generate.py \
    --llm /path/to/LLM \
    --rm /path/to/value_model_backbone \
    --rm_lora /path/to/value_model_checkpoint \
    --max_new_token 128 \
    --topk 10 \
    --rm_weight 1.0 \
    --entropy_threshold 1.0
```

Key arguments:

| Argument | Description |
|---|---|
| `--llm` | Path to the LLM |
| `--rm` | Path to the base reward model used by the Value Model |
| `--rm_lora` | Path to the Value Model checkpoint directory (LoRA weights) |
| `--entropy_threshold` | Entropy threshold above which RM guidance is applied; set to `None` to guide every token |
| `--topk` | Number of candidate tokens pre-screened by the LLM |
| `--rm_weight` | Weighting coefficient for the RM scores when combining with LLM logits |
| `--max_new_token` | Maximum number of tokens to generate |
| `--llm_gpu` / `--rm_gpu` | GPU devices for the LLM and RM respectively (e.g. `cuda:0`, `cuda:1`) |

The script runs an interactive loop: for each prompt it shows the **Base** output (no RM guidance) and the **SIA** output side-by-side, along with the fraction of tokens that received RM guidance.

---

## Evaluation

### 1. Download evaluation datasets

| Dataset | Link |
|---|---|
| AdvBench | [https://huggingface.co/datasets/walledai/AdvBench](https://huggingface.co/datasets/walledai/AdvBench) |
| AlpacaEval 2 | [https://huggingface.co/datasets/tatsu-lab/alpaca_eval](https://huggingface.co/datasets/tatsu-lab/alpaca_eval) |
| TruthfulQA | [https://huggingface.co/datasets/domenicrosati/TruthfulQA](https://huggingface.co/datasets/domenicrosati/TruthfulQA) |

### 2. Download the evaluation reward model

Model weights: [Skywork-Reward-V2-Llama-3.1-8B](https://huggingface.co/Skywork/Skywork-Reward-V2-Llama-3.1-8B)

This model is used to score generated outputs and compute the final reward metric.

### 3. Run evaluation

```bash
bash scripts/evaluate.sh
```

Or manually:

```bash
python3 evaluate.py \
    --llm /path/to/LLM \
    --rm /path/to/value_model_backbone \
    --rm_lora /path/to/value_model_checkpoint \
    --dataset /path/to/dataset \
    --run_num -1 \
    --run_percent 100 \
    --max_new_token 128 \
    --rm_weight 1.0 \
    --topk 10 \
    --sample_temp 1.0 \
    --all_ratios_entropy /path/to/entropy_quantiles.json
```

Results are saved under `assets/generation_results/<model>/<dataset>/` in JSON format, together with an `experiment_config.json` that records all run hyperparameters.

---

## Value Model Training

### 1. Download training datasets

| Dataset | Link |
|---|---|
| WildGuardMix | [https://huggingface.co/datasets/allenai/wildguardmix](https://huggingface.co/datasets/allenai/wildguardmix) |
| UltraFeedback | [https://huggingface.co/datasets/openbmb/UltraFeedback](https://huggingface.co/datasets/openbmb/UltraFeedback) |

### 2. Download the training reward model

Model weights: [Skywork-Reward-V2-Qwen3-8B](https://huggingface.co/Skywork/Skywork-Reward-V2-Qwen3-8B)

This trajectory-level RM is used to annotate RM scores for the training data and serves as the backbone of the Value Model.

### 3. Dataset preprocessing

#### Step 1 — Format conversion

Convert raw datasets into the token-sequence format expected by the trainer.

```bash
bash scripts/preprocess_vm_data.sh
```

Or run the conversion script directly:

```bash
# WildGuardMix
python3 src/value_model/preprocess.py wildguardmix \
    --dataset_path /path/to/wildguardmix \
    --split wildguardtrain \
    --tokenizer_path /path/to/reward_model_tokenizer \
    --output_path assets/vm_data/wildguardmix-Qwen3.json

# UltraFeedback
python3 src/value_model/preprocess.py ultrafeedback \
    --dataset_path /path/to/ultrafeedback \
    --tokenizer_path /path/to/reward_model_tokenizer \
    --output_path assets/vm_data/ultrafeedback-Qwen3.json
```

#### Step 2 — RM score annotation

Annotate each sample with a trajectory-level reward score using the Skywork RM:

```bash
bash scripts/measure_reward.sh
```

Or manually:

```bash
python3 src/measure_reward.py \
    --input_file  assets/vm_data/wildguardmix-Qwen3.json \
    --output_file assets/vm_data/wildguardmix-Qwen3-Skywork-Reward-V2-Qwen3-8B.json \
    --rm /path/to/Skywork-Reward-V2-Qwen3-8B \
    --device cuda:0
```

### 4. Train the Value Model

```bash
bash scripts/train_vm.sh
```

Or manually:

```bash
python3 src/value_model/train.py \
    --data_file assets/vm_data/wildguardmix-Qwen3-Skywork-Reward-V2-Qwen3-8B.json \
    --base_model_path /path/to/Skywork-Reward-V2-Qwen3-4B \
    --output_dir assets/vm_checkpoints/Skywork-Reward-V2-Qwen3-4B/wildguardmix \
    --batch_size 8 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-4 \
    --num_epochs 3 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --save_model
```

Training saves LoRA weights (`lora_weights/`), a token reward head (`token_reward_head.pt`), and a configuration file (`model_config.json`) to the specified output directory. Training curves are plotted automatically and saved as `training_curves.png`.
