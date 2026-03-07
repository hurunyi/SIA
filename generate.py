import argparse
import torch
from src.sia import SIA


SEPARATOR = "=" * 80


def parse_arguments():
    parser = argparse.ArgumentParser(description="交互式生成脚本：对比 Base 模式与 Entropy 过滤模式")
    parser.add_argument("--llm", type=str, required=True, help="LLM 模型路径")
    parser.add_argument("--rm", type=str, required=True, help="Reward Model 路径")
    parser.add_argument("--rm_lora", type=str, default=None, help="RM LoRA 权重路径（可选）")
    parser.add_argument("--llm_gpu", type=str, default="cuda", help="LLM 使用的设备（默认 cuda）")
    parser.add_argument("--rm_gpu", type=str, default="cuda", help="RM 使用的设备（默认 cuda）")
    parser.add_argument("--max_new_token", type=int, default=128, help="最大生成 token 数（默认 128）")
    parser.add_argument("--topk", type=int, default=10, help="候选 token 数（默认 10）")
    parser.add_argument("--rm_weight", type=float, default=1.0, help="Entropy 模式下的 RM 权重（默认 1.0）")
    parser.add_argument("--temperature", type=float, default=1.0, help="采样温度（默认 1.0）")
    parser.add_argument("--entropy_threshold", type=float, default=None,
                        help="Entropy 过滤阈值，超过此值时启用 RM 引导（默认 None，即对所有 token 均使用 RM）")
    return parser.parse_args()


def run_prompt(sia: SIA, prompt: str, rm_weight: float, topk: int,
               max_new_token: int, temperature: float, entropy_threshold=None):
    """调用 SIA 生成并返回生成文本，若失败返回 None。"""
    result = sia.generate(
        prompt,
        weight=rm_weight,
        topk=topk,
        max_new_token=max_new_token,
        temperature=temperature,
        intervene_num=None,
        entropy_threshold=entropy_threshold,
        entropy_ratio_threshold=None,
        attention_threshold=None,
        attention_ratio_threshold=None,
        random_intervene_ratio=None,
        rm_trajectory_len=None,
    )
    if result is None:
        return None
    tokens = result["llm_tokens"]
    llm_prompt_no_special_tokens = result["llm_prompt_no_special_tokens"]
    tokens_text = sia.tokens_to_text(tokens)[0]
    response = tokens_text.removeprefix(llm_prompt_no_special_tokens)
    intervene_ratio = result["intervene_ratio"]
    return response, intervene_ratio


def main():
    args = parse_arguments()

    print(SEPARATOR)
    print("正在加载模型，请稍候...")
    print(f"  LLM      : {args.llm}")
    print(f"  RM       : {args.rm}")
    if args.rm_lora:
        print(f"  RM LoRA  : {args.rm_lora}")
    print(SEPARATOR)

    sia = SIA(
        llm_path=args.llm,
        rm_path=args.rm,
        rm_lora_path=args.rm_lora,
        llm_dev=args.llm_gpu,
        rm_dev=args.rm_gpu,
        torch_dtype=torch.float16,
    )
    print("模型加载完成！")
    print(SEPARATOR)

    print("交互式生成已就绪。输入 'exit' 或 'quit' 退出。")
    print(f"生成参数：topk={args.topk}, temperature={args.temperature}, "
          f"max_new_token={args.max_new_token}")
    print(f"Entropy 模式：rm_weight={args.rm_weight}, entropy_threshold={args.entropy_threshold}")
    print(SEPARATOR)

    while True:
        try:
            prompt_text = input("\n请输入 Prompt（输入 exit 退出）:\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            break

        if prompt_text.lower() in ("exit", "quit"):
            print("已退出。")
            break

        if not prompt_text:
            print("Prompt 不能为空，请重新输入。")
            continue

        # 将用户输入格式化为模型期望的对话格式
        formatted_prompt = f"Human:\n{prompt_text}\nAssistant:\n"

        # ── Base 模式 ──────────────────────────────────────────────────────────
        print(f"\n{SEPARATOR}")
        print("【Base 模式】正在生成（rm_weight=0.0，不使用 RM 引导）...")
        base_result = run_prompt(
            sia,
            formatted_prompt,
            rm_weight=0.0,
            topk=args.topk,
            max_new_token=args.max_new_token,
            temperature=args.temperature,
            entropy_threshold=None,
        )

        print(f"\n{'─' * 70}")
        print("【Base 模式】生成结果：")
        print(f"{'─' * 70}")
        if base_result is None:
            print("（序列过长，跳过）")
        else:
            response_base, intervene_ratio_base = base_result
            print(response_base)
            print(f"\n（RM 引导比例：{intervene_ratio_base:.2%}）")

        # ── Entropy 过滤模式 ───────────────────────────────────────────────────
        entropy_desc = (
            f"entropy_threshold={args.entropy_threshold}" if args.entropy_threshold is not None
            else "对所有 token 均使用 RM"
        )
        print(f"\n{SEPARATOR}")
        print(f"【Entropy 过滤模式】正在生成（rm_weight={args.rm_weight}，{entropy_desc}）...")
        entropy_result = run_prompt(
            sia,
            formatted_prompt,
            rm_weight=args.rm_weight,
            topk=args.topk,
            max_new_token=args.max_new_token,
            temperature=args.temperature,
            entropy_threshold=args.entropy_threshold,
        )

        print(f"\n{'─' * 70}")
        print("【Entropy 过滤模式】生成结果：")
        print(f"{'─' * 70}")
        if entropy_result is None:
            print("（序列过长，跳过）")
        else:
            response_entropy, intervene_ratio_entropy = entropy_result
            print(response_entropy)
            print(f"\n（RM 引导比例：{intervene_ratio_entropy:.2%}）")

        print(f"\n{SEPARATOR}")


if __name__ == "__main__":
    main()
