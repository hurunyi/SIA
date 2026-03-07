"""
数据预处理：根据给定 dataset 类型将不同数据集转换为指定格式

支持的数据集:
  - ultrafeedback: UltraFeedback/UltraChat 等 JSONL 格式（instruction + completions）
  - wildguardmix: WildGuardMix 数据集（prompt + response）

用法示例:
  python preprocess.py ultrafeedback --dataset_path /path/to/ultrachat.jsonl --tokenizer_path /path/to/tokenizer --output_path out.json
  python preprocess.py wildguardmix --dataset_path /path/to/wildguardmix --split wildguardtrain --tokenizer_path /path/to/tokenizer --output_path out.json
  python preprocess.py ultrafeedback ... --max_samples 1000   # 只转换前 1000 条样本
"""

import os
import json
import argparse
import glob
from tqdm import tqdm
from transformers import AutoTokenizer


# ---------- 公共：rm_prompt 构建 ----------
def build_rm_prompt(instruction: str, tokenizer_path: str) -> str:
    if "Qwen3" in tokenizer_path:
        return f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
    elif "Llama-3" in tokenizer_path:
        return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    else:
        raise ValueError(f"不支持的分词器: {tokenizer_path}")


def build_result_dict(idx, prompt, result, rm_prompt, tokenizer):
    rm_prompt_tokens = tokenizer.encode(rm_prompt, add_special_tokens=False)
    rm_prompt_tokens_ids = ",".join(map(str, rm_prompt_tokens))
    generated_tokens = tokenizer.encode(result, add_special_tokens=False)
    generated_tokens_ids = ",".join(map(str, generated_tokens))
    rm_guided_tokens_mask = ",".join(["1"] * len(generated_tokens))
    return {
        "id": idx,
        "prompt": prompt,
        "result": result,
        "rm_prompt": rm_prompt,
        "rm_prompt_tokens_ids": rm_prompt_tokens_ids,
        "generated_tokens_ids": generated_tokens_ids,
        "rm_guided_tokens_mask": rm_guided_tokens_mask,
        "intervene_ratio": 1.0,
    }


# ---------- UltraFeedback ----------
def convert_ultrafeedback_single_file(
    dataset_path: str,
    tokenizer,
    tokenizer_path: str,
    start_idx: int = 0,
    max_samples: int | None = None,
):
    dataset = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                dataset.append(json.loads(line))

    if len(dataset) > 0:
        sample = dataset[0]
        available_fields = list(sample.keys())
        if "instruction" not in available_fields:
            raise ValueError(f"未找到 'instruction' 字段。可用字段: {available_fields}")
        if "completions" not in available_fields:
            raise ValueError(f"未找到 'completions' 字段。可用字段: {available_fields}")

    results = []
    idx = start_idx
    for sample in tqdm(dataset, desc="转换数据"):
        if max_samples is not None and len(results) >= max_samples:
            break
        instruction = sample.get("instruction")
        completions = sample.get("completions", [])
        if instruction is None:
            continue

        prompt = f"Human:\n{instruction}\nAssistant:\n"
        rm_prompt = build_rm_prompt(instruction, tokenizer_path)

        for completion in completions:
            if max_samples is not None and len(results) >= max_samples:
                break
            response = completion.get("response")
            if response is None:
                continue
            idx += 1
            result_dict = build_result_dict(idx, prompt, response, rm_prompt, tokenizer)
            results.append(result_dict)

    return results, idx


def run_ultrafeedback(
    dataset_path: str,
    tokenizer_path: str,
    output_path: str,
    max_samples: int | None = None,
):
    print(f"加载分词器: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("分词器加载完成")

    if os.path.isdir(dataset_path):
        jsonl_files = glob.glob(os.path.join(dataset_path, "*.jsonl"))
        jsonl_files.sort()
        print(f"在目录 {dataset_path} 中找到 {len(jsonl_files)} 个 JSONL 文件:")
        for f in jsonl_files:
            print(f"  - {os.path.basename(f)}")
    elif os.path.isfile(dataset_path):
        jsonl_files = [dataset_path]
        print(f"处理单个文件: {dataset_path}")
    else:
        raise ValueError(f"路径不存在: {dataset_path}")

    if len(jsonl_files) == 0:
        raise ValueError(f"未找到任何 JSONL 文件: {dataset_path}")

    all_results = []
    idx = 0
    for jsonl_file in tqdm(jsonl_files, desc="处理文件"):
        if max_samples is not None and len(all_results) >= max_samples:
            break
        file_basename = os.path.basename(jsonl_file)
        print(f"\n处理文件: {file_basename}")
        try:
            remaining = (max_samples - len(all_results)) if max_samples is not None else None
            results, idx = convert_ultrafeedback_single_file(
                jsonl_file, tokenizer, tokenizer_path, start_idx=idx, max_samples=remaining
            )
            all_results.extend(results)
            print(f"  处理完成: {len(results)} 个样本")
        except Exception as e:
            print(f"  处理文件 {file_basename} 时出错: {e}")
            continue

        if max_samples is not None and len(all_results) >= max_samples:
            print(f"已达到最大样本数限制: {max_samples}")
            break

        if len(all_results) > 0 and len(all_results) % 1000 == 0:
            print(f"中间保存: 当前共 {len(all_results)} 个样本")
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\n保存最终结果到: {output_path}")
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"转换完成！共处理 {len(jsonl_files)} 个文件，{len(all_results)} 个样本")
    print(f"结果已保存到: {output_path}")


# ---------- WildGuardMix ----------
def run_wildguardmix(
    dataset_path: str,
    split: str,
    tokenizer_path: str,
    output_path: str,
    max_samples: int | None = None,
):
    from datasets import load_dataset

    print(f"加载数据集: {dataset_path}, split: {split}")
    dataset = load_dataset(dataset_path, split)["train"]
    print(f"数据集大小: {len(dataset)}")

    prompt_field = None
    response_field = None
    if len(dataset) > 0:
        sample = dataset[0]
        available_fields = list(sample.keys())
        print(f"数据集字段: {available_fields}")
        if "prompt" in available_fields:
            prompt_field = "prompt"
        else:
            raise ValueError(f"未找到 'prompt' 字段。可用字段: {available_fields}")
        if "response" in available_fields:
            response_field = "response"
        else:
            raise ValueError(f"未找到 'response' 字段。可用字段: {available_fields}")
        print(f"使用字段: prompt='{prompt_field}', response='{response_field}'")

    print(f"加载分词器: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("分词器加载完成")

    results = []
    idx = 0
    for _, sample in enumerate(tqdm(dataset, desc="转换数据")):
        if max_samples is not None and len(results) >= max_samples:
            print(f"已达到最大样本数限制: {max_samples}")
            break
        p = sample[prompt_field]
        q = sample[response_field]
        if p is None or q is None:
            continue
        idx += 1

        prompt = f"Human:\n{p}\nAssistant:\n"
        result = q
        rm_prompt = build_rm_prompt(p, tokenizer_path)
        result_dict = build_result_dict(idx, prompt, result, rm_prompt, tokenizer)
        results.append(result_dict)

        if idx % 100 == 0:
            print(f"中间保存: 当前共 {len(results)} 个样本")
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"保存结果到: {output_path}")
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"转换完成！共处理 {len(results)} 个样本")
    print(f"结果已保存到: {output_path}")


# ---------- 入口 ----------
def main():
    parser = argparse.ArgumentParser(
        description="根据 dataset 类型将数据集转换为指定格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # UltraFeedback/UltraChat (单文件或目录下的 JSONL)
  python preprocess.py ultrafeedback --dataset_path /path/to/ultrachat.jsonl --tokenizer_path /path/to/tokenizer --output_path out.json

  # WildGuardMix
  python preprocess.py wildguardmix --dataset_path /path/to/wildguardmix --split wildguardtrain --tokenizer_path /path/to/tokenizer --output_path out.json
        """,
    )
    parser.add_argument(
        "dataset",
        type=str,
        choices=["ultrafeedback", "wildguardmix"],
        help="数据集类型: ultrafeedback 或 wildguardmix",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="数据集路径（ultrafeedback: 单文件或目录; wildguardmix: HF 数据集名或路径）",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        required=True,
        help="分词器路径（如 Skywork-Reward-V2-Qwen3-1.7B 或 Llama-3.2-1B）",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="输出 JSON 文件路径",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="wildguardtrain",
        help="[仅 wildguardmix] 数据集 split，如 wildguardtrain (默认: wildguardtrain)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        metavar="N",
        help="最多转换的样本数量，不指定则转换全部",
    )

    args = parser.parse_args()

    print("参数设置:")
    print(f"  数据集类型: {args.dataset}")
    print(f"  数据集路径: {args.dataset_path}")
    print(f"  分词器路径: {args.tokenizer_path}")
    print(f"  输出路径: {args.output_path}")
    if args.dataset == "wildguardmix":
        print(f"  Split: {args.split}")
    if args.max_samples is not None:
        print(f"  最大样本数: {args.max_samples}")
    print()

    if args.dataset == "ultrafeedback":
        run_ultrafeedback(
            dataset_path=args.dataset_path,
            tokenizer_path=args.tokenizer_path,
            output_path=args.output_path,
            max_samples=args.max_samples,
        )
    else:
        run_wildguardmix(
            dataset_path=args.dataset_path,
            split=args.split,
            tokenizer_path=args.tokenizer_path,
            output_path=args.output_path,
            max_samples=args.max_samples,
        )


if __name__ == "__main__":
    main()
