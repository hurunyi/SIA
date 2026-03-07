export CUDA_VISIBLE_DEVICES=0

python3 generate.py \
    --llm="/path/to/LLM" \
    --rm="/path/to/value_model_backbone" \
    --max_new_token=128 \
    --rm_lora="/path/to/value_model_checkpoint" \
    --entropy_threshold=1.0
