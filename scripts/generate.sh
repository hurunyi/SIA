export CUDA_VISIBLE_DEVICES=3

LLM_NAME="Qwen3-4B-Base"
RM_NAME="Qwen3-4B-Base"

python3 generate.py \
    --llm="/mnt/ssd2/shiqian/models/${LLM_NAME}" \
    --rm="/mnt/ssd2/shiqian/models/${RM_NAME}" \
    --max_new_token=128 \
    --rm_lora="/mnt/ssd2/shiqian/research/SIA_assets/token_rm_models/${RM_NAME}/wildguardmix_UltraChat_ShareGPT_all-tokens" \
    --entropy_threshold=1.0
