export CUDA_VISIBLE_DEVICES=3

LLM_NAME="Qwen3-0.6B-Base"
RM_NAME="Qwen3-0.6B-Base"
DATASET_NAME="alpaca_eval"

python3 evaluate.py \
    --llm="/mnt/ssd2/shiqian/models/${LLM_NAME}" \
    --rm="/mnt/ssd2/shiqian/models/${RM_NAME}" \
    --dataset="/mnt/ssd2/shiqian/datasets/${DATASET_NAME}" \
    --run_num=-1 \
    --run_percent=100 \
    --max_new_token=128 \
    --rm_lora="/mnt/ssd2/shiqian/research/SIA_assets/token_rm_models/${RM_NAME}/wildguardmix_UltraChat_ShareGPT_all-tokens" \
    --all_ratios_entropy="/mnt/ssd2/shiqian/research/SIA_assets/final_results/${LLM_NAME}/${DATASET_NAME}/entropy_topk-10_entropy_quantiles.json" \
    --rm_weight=1.0 \
    --topk=10 \
    --sample_temp=1.0 \
