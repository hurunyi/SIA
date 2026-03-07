#!/bin/bash

# 训练token级别的RM模型脚本

# 设置默认参数
# DATA_FILE可以是单个文件或多个文件（用数组存储）
DATA_FILE=("/mnt/ssd2/shiqian/open-source/SIA/assets/vm_data/wildguardmix-Qwen3-Skywork-Reward-V2-Llama-3.1-8B.json")
BASE_MODEL_PATH="/mnt/ssd2/shiqian/models/Skywork-Reward-V2-Qwen3-4B"
OUTPUT_DIR="/mnt/ssd2/shiqian/open-source/SIA/assets/vm_checkpoints/Skywork-Reward-V2-Qwen3-4B/wildguardmix"
BATCH_SIZE=8
GRADIENT_ACCUMULATION_STEPS=2
LEARNING_RATE=1e-4
NUM_EPOCHS=3
MAX_LENGTH=1024
DEVICE="cuda:2"
PLOT_STEPS=10
SAVE_INTERVAL_STEPS=5000

# LoRA参数
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.1

# 训练参数
TRAIN_RATIO=0.99
WEIGHT_DECAY=1e-4
RESUME_FROM=""

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --data_file)
            # 支持多个文件路径，收集所有--data_file后面的参数直到遇到下一个--开头的参数
            DATA_FILE=()
            shift  # 跳过--data_file
            # 收集所有文件路径，直到遇到下一个选项或参数结束
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
                DATA_FILE+=("$1")
                shift
            done
            ;;
        --base_model_path)
            BASE_MODEL_PATH="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --gradient_accumulation_steps)
            GRADIENT_ACCUMULATION_STEPS="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --num_epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --lora_r)
            LORA_R="$2"
            shift 2
            ;;
        --lora_alpha)
            LORA_ALPHA="$2"
            shift 2
            ;;
        --resume_from)
            RESUME_FROM="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

# 运行训练脚本
if [ -n "$RESUME_FROM" ]; then
    python3 src/value_model/train.py \
        --data_file "${DATA_FILE[@]}" \
        --base_model_path "$BASE_MODEL_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --batch_size "$BATCH_SIZE" \
        --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
        --learning_rate "$LEARNING_RATE" \
        --num_epochs "$NUM_EPOCHS" \
        --max_length "$MAX_LENGTH" \
        --train_ratio "$TRAIN_RATIO" \
        --device "$DEVICE" \
        --lora_r "$LORA_R" \
        --lora_alpha "$LORA_ALPHA" \
        --lora_dropout "$LORA_DROPOUT" \
        --weight_decay "$WEIGHT_DECAY" \
        --save_model \
        --resume_from "$RESUME_FROM" \
        --plot_steps "$PLOT_STEPS" \
        --save_interval_steps "$SAVE_INTERVAL_STEPS"
else
    python3 src/value_model/train.py \
        --data_file "${DATA_FILE[@]}" \
        --base_model_path "$BASE_MODEL_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --batch_size "$BATCH_SIZE" \
        --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
        --learning_rate "$LEARNING_RATE" \
        --num_epochs "$NUM_EPOCHS" \
        --max_length "$MAX_LENGTH" \
        --train_ratio "$TRAIN_RATIO" \
        --device "$DEVICE" \
        --lora_r "$LORA_R" \
        --lora_alpha "$LORA_ALPHA" \
        --lora_dropout "$LORA_DROPOUT" \
        --weight_decay "$WEIGHT_DECAY" \
        --save_model \
        --plot_steps "$PLOT_STEPS" \
        --save_interval_steps "$SAVE_INTERVAL_STEPS"
fi

