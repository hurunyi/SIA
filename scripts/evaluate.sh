export CUDA_VISIBLE_DEVICES=0


python3 evaluate.py \
    --llm="/path/to/LLM" \
    --rm="/path/to/value_model_backbone" \
    --dataset="/path/to/dataset" \
    --run_num=-1 \
    --run_percent=100 \
    --max_new_token=128 \
    --rm_lora="/path/to/value_model_checkpoint" \
    --all_ratios_entropy="/path/to/entropy_quantiles.json" \
    --rm_weight=1.0 \
    --topk=10 \
    --sample_temp=1.0