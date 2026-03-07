import os
import json
import time
import argparse
import random
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from datasets import load_dataset
import torch
from src.sia import SIA


class DatasetProcessor:
    """数据集处理器"""
    
    @staticmethod
    def load_and_process_dataset(dataset_name: str, split: str, add_sys_prompt: bool = False):
        """加载并处理数据集"""
        print(f"[INFO]: Loading dataset ({dataset_name=}, {split=})")
        
        if "HEx-PHI" in dataset_name:
            test_ds = open(os.path.join(dataset_name, "merged.csv"), "r").readlines()
            test_ds = [line.strip() for line in test_ds]
        elif "alpaca_eval" in dataset_name:
            test_ds = json.load(open(os.path.join(dataset_name, "alpaca_eval.json"), "r"))
        elif "wildguardmix" in dataset_name:
            test_ds = load_dataset(dataset_name, split)["test"]
        else:
            test_ds = load_dataset(dataset_name, split=split)
        
        if "full-hh-rlhf" in dataset_name:
            # FOR HHRLHF
            test_ds = test_ds["prompt"]
        elif "SHP" in dataset_name:
            # FOR SHP
            unique_prompts = []
            seen_posts = set()
            for post_id, histr in zip(test_ds["post_id"], test_ds['history']):
                if post_id in seen_posts: 
                    continue
                model_prompt = "Human:\n" + histr + "\nAssistant:\n"
                unique_prompts.append(model_prompt)
                seen_posts.add(post_id)
            test_ds = unique_prompts
        elif 'AdvBench' in dataset_name:
            unique_prompts = []
            for example in test_ds["prompt"]:
                model_prompt = "Human:\n" + example + "\nAssistant:\n"
                unique_prompts.append(model_prompt)
            test_ds = unique_prompts
        elif 'TruthfulQA' in dataset_name:
            unique_prompts = []
            for example in test_ds["Question"]:
                model_prompt = "Human:\n" + example + "\nAssistant:\n"
                unique_prompts.append(model_prompt)
            test_ds = unique_prompts
        elif 'HEx-PHI' in dataset_name:
            unique_prompts = []
            for example in test_ds:
                # 去除首尾的双引号
                if example.startswith('"') and example.endswith('"'):
                    example = example[1:-1]
                model_prompt = "Human:\n" + example + "\nAssistant:\n"
                unique_prompts.append(model_prompt)
            test_ds = unique_prompts
        elif 'alpaca_eval' in dataset_name:
            unique_prompts = []
            for example in test_ds:
                model_prompt = "Human:\n" + example["instruction"] + "\nAssistant:\n"
                unique_prompts.append(model_prompt)
            test_ds = unique_prompts
        elif "wildguardmix" in dataset_name:
            unique_prompts = []
            for example in test_ds["prompt"]:
                model_prompt = "Human:\n" + example + "\nAssistant:\n"
                unique_prompts.append(model_prompt)
            test_ds = unique_prompts

        if add_sys_prompt:
            sys_prompt = '''You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.\n'''
            test_ds = list(test_ds)
            for idx in range(len(test_ds)):
                test_ds[idx] = sys_prompt + test_ds[idx]

        return test_ds


class ModelExperimentRunner:
    """模型实验运行器"""
    
    def __init__(self, llm_path: str, rm_path: str, vm_path: str, rm_lora_path: str, llm_dev: str, rm_dev: str):
        """初始化实验运行器"""
        torch_dtype = torch.float16
        self.search = SIA(
            llm_path=llm_path,
            rm_path=rm_path,
            vm_path=vm_path,
            rm_lora_path=rm_lora_path,
            llm_dev=llm_dev,
            rm_dev=rm_dev,
            torch_dtype=torch_dtype
        )
        
    def run_prompt(
        self, prompt: str, rm_weight: float = 0., topk: int = 5, 
        new_token: int = 24, sample_temp=None, intervene_num: int = 5,
        entropy_threshold=None, entropy_ratio_threshold=None,
        rm_trajectory_len=None, attention_threshold=None, attention_ratio_threshold=None,
        random_intervene_ratio=None
    ):
        """运行单个prompt"""
        generated_results = self.search.generate(
            prompt,
            temperature=sample_temp,
            topk=topk,
            max_new_token=new_token,
            weight=rm_weight,
            intervene_num=intervene_num,
            entropy_threshold=entropy_threshold,
            entropy_ratio_threshold=entropy_ratio_threshold,
            rm_trajectory_len=rm_trajectory_len,
            attention_threshold=attention_threshold,
            attention_ratio_threshold=attention_ratio_threshold,
            random_intervene_ratio=random_intervene_ratio
        )

        # too long seqlen
        if generated_results is None:
            return None
        else:
            tokens = generated_results["llm_tokens"]
            llm_prompt_no_special_tokens = generated_results["llm_prompt_no_special_tokens"]
        
        raw_tokens = tokens[0].detach().cpu().numpy().tolist()
        tokens_text = self.search.tokens_to_text(tokens)[0]
        tokens_text_no_prefix = tokens_text.removeprefix(llm_prompt_no_special_tokens)

        run_prompt_results = {
            "tokens_text_no_prefix": tokens_text_no_prefix,
            "raw_tokens": raw_tokens,
            "tokens_shift": generated_results["tokens_shift"],
            "tokens_entropy": generated_results["tokens_entropy"],
            "tokens_entropy_ratio": generated_results["tokens_entropy_ratio"],
            "llm_prompt": generated_results["llm_prompt"],
            "rm_prompt": generated_results["rm_prompt"],
            "intervene_ratio": generated_results["intervene_ratio"],
            "llm_prompt_tokens_ids": generated_results["llm_prompt_tokens_ids"],
            "rm_prompt_tokens_ids": generated_results["rm_prompt_tokens_ids"],
            "generated_tokens_ids": generated_results["generated_tokens_ids"],
            "rm_guided_tokens_mask": generated_results["rm_guided_tokens_mask"],
        }
        return run_prompt_results

    def generate_save_directory(self, base_dir: str, model_name: str, rm_name: str, vm_name: str, rm_lora_name: str, dataset_name: str, 
                              run_config: dict, add_sys_prompt: bool = False) -> str:
        """生成保存目录路径"""

        if run_config['rm_weight'] == 0.0:
            save_dir = f"{base_dir}/Base/topk-{run_config['topk']}"
        else:
            config_dir = (
                f"{'/' + 'position-' + '-' + str(run_config['intervene_num']) if run_config['intervene_num'] is not None else ''}"
                f"{'/' + 'random-' + str(run_config['random_intervene_ratio']) if run_config['random_intervene_ratio'] is not None else ''}"
                f"{'/' + 'attention-' + format(run_config['attention_threshold'], '.3f') if run_config['attention_threshold'] is not None else ''}"
                f"{'/' + 'attention-ratio-' + format(run_config['attention_ratio_threshold'], '.3f') if run_config['attention_ratio_threshold'] is not None else ''}"
                f"{'/' + 'entropy-' + format(run_config['entropy_threshold'], '.3f') if run_config['entropy_threshold'] is not None else ''}"
                f"{'/' + 'entropy-ratio-' + format(run_config['entropy_ratio_threshold'], '.3f') if run_config['entropy_ratio_threshold'] is not None else ''}"
            )

            if config_dir:
                save_dir = f"{base_dir}/Ours/topk-{run_config['topk']}/rm-{rm_name}/weight-{run_config['rm_weight']}" + config_dir
            else:
                save_dir = f"{base_dir}/Ours/topk-{run_config['topk']}/rm-{rm_name}/weight-{run_config['rm_weight']}" + "/all-tokens"
        
        save_dir += f"/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        return save_dir


class ConfigValidator:
    """配置验证器"""
    
    @staticmethod
    def validate_run_configs(run_configs: list) -> bool:
        """验证运行配置"""
        required_keys = ['rm_weight', 'topk', 'sample_temp']
        
        for run_config in run_configs:
            for key in required_keys:
                if key not in run_config:
                    print(f"Missing key '{key}' in {run_config=}")
                    return False
        
        print(f"[INFO]: Loaded {len(run_configs)} run configs.")
        print(f"[DEBUG]: {run_configs=}")
        return True


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Dahoas/full-hh-rlhf")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--run_percent", type=float, default=100.)
    parser.add_argument("--run_num", type=int, default=1000)
    parser.add_argument("--rm", type=str)
    parser.add_argument("--llm", type=str)
    parser.add_argument("--vm", type=str, default=None)
    parser.add_argument("--rm_lora", type=str, default=None)
    parser.add_argument("--max_new_token", type=int, default=128)
    parser.add_argument("--llm_gpu", type=str, default="cuda")
    parser.add_argument("--rm_gpu", type=str, default="cuda")
    parser.add_argument("--recover", action='store_true', default=False)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--add_sys_prompt", action='store_true', default=False)
    # run config parameters (used when --config is not provided)
    parser.add_argument("--rm_weight", type=float, default=1.0)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--sample_temp", type=float, default=1.0)
    parser.add_argument("--intervene_num", type=int, default=None)
    parser.add_argument("--entropy_threshold", type=float, default=None)
    parser.add_argument("--entropy_ratio_threshold", type=float, default=None)
    parser.add_argument("--attention_threshold", type=float, default=None)
    parser.add_argument("--attention_ratio_threshold", type=float, default=None)
    parser.add_argument("--rm_trajectory_len", type=int, default=None)
    parser.add_argument("--random_intervene_ratio", type=float, default=None)
    parser.add_argument("--all_ratios_random", action='store_true', default=False)
    parser.add_argument("--all_ratios_position", action='store_true', default=False)
    parser.add_argument("--all_ratios_attention", type=str, default=None)
    parser.add_argument("--all_ratios_attention_ratio", type=str, default=None)
    parser.add_argument("--all_ratios_entropy", type=str, default=None)
    parser.add_argument("--all_ratios_entropy_ratio", type=str, default=None)
    
    return parser.parse_args()


def main():
    """主函数"""
    # 解析参数
    args = parse_arguments()
    
    print(f"{args=}")
    
    # 提取模型和数据集名称
    model_name = args.llm.split("/")[-1]
    rm_name = args.rm.split("/")[-1]
    vm_name = "vm" if args.vm else None
    rm_lora_name = args.rm_lora.split("/")[-2] if args.rm_lora else None
    dataset_name = args.dataset.split("/")[-1]
    
    # 调整split参数
    if "AdvBench" in args.dataset or "TruthfulQA" in args.dataset:
        args.split = "train"
    elif 'HEx-PHI' in args.dataset:
        args.split = None
    elif "wildguardmix" in args.dataset:
        args.split = "wildguardtest"
    
    # 加载配置文件，若未指定则从命令行参数构建
    if args.config is not None:
        cfg_path = Path(args.config)
        with open(cfg_path) as f:
            run_configs = json.loads(f.read())
    else:
        run_configs = [{
            "rm_weight": args.rm_weight,
            "topk": args.topk,
            "sample_temp": args.sample_temp,
            "intervene_num": args.intervene_num,
            "entropy_threshold": args.entropy_threshold,
            "entropy_ratio_threshold": args.entropy_ratio_threshold,
            "attention_threshold": args.attention_threshold,
            "attention_ratio_threshold": args.attention_ratio_threshold,
            "rm_trajectory_len": args.rm_trajectory_len,
            "random_intervene_ratio": args.random_intervene_ratio,
        }]
    
    if args.all_ratios_random:
        assert len(run_configs) == 1, f"{run_configs=} is not a list of length 1"
        run_configs_random = []
        for random_intervene_ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]:
            run_config = run_configs[0].copy()
            run_config["random_intervene_ratio"] = random_intervene_ratio
            run_configs_random.append(run_config)
        run_configs = run_configs_random
    elif args.all_ratios_position:
        assert len(run_configs) == 1, f"{run_configs=} is not a list of length 1"
        run_configs_position = []
        for position_value in [20, 40, 60, 80, 100, 120, 180, args.max_new_token]:
            run_config = run_configs[0].copy()
            run_config["intervene_num"] = position_value
            run_configs_position.append(run_config)
        run_configs = run_configs_position
    elif args.all_ratios_attention is not None:
        assert len(run_configs) == 1, f"{run_configs=} is not a list of length 1"
        all_ratios_attention_data = json.load(open(args.all_ratios_attention))
        run_configs_attention = []
        for attention_quantile in all_ratios_attention_data["quantiles"].keys():
            run_config = run_configs[0].copy()
            value = all_ratios_attention_data["quantiles"][attention_quantile]
            run_config["attention_threshold"] = round(value, 3)
            run_configs_attention.append(run_config)
        run_configs = run_configs_attention
    elif args.all_ratios_attention_ratio is not None:
        assert len(run_configs) == 1, f"{run_configs=} is not a list of length 1"
        all_ratios_attention_ratio_data = json.load(open(args.all_ratios_attention_ratio))
        run_configs_attention_ratio = []
        for attention_ratio_quantile in all_ratios_attention_ratio_data["quantiles"].keys():
            run_config = run_configs[0].copy()
            value = all_ratios_attention_ratio_data["quantiles"][attention_ratio_quantile]
            run_config["attention_ratio_threshold"] = round(value, 3)
            run_configs_attention_ratio.append(run_config)
        run_configs = run_configs_attention_ratio
    elif args.all_ratios_entropy is not None:
        assert len(run_configs) == 1, f"{run_configs=} is not a list of length 1"
        all_ratios_entropy_data = json.load(open(args.all_ratios_entropy))
        run_configs_entropy = []
        for entropy_quantile in all_ratios_entropy_data["quantiles"].keys():
            run_config = run_configs[0].copy()
            value = all_ratios_entropy_data["quantiles"][entropy_quantile]
            run_config["entropy_threshold"] = round(value, 3)
            run_configs_entropy.append(run_config)
        run_configs = run_configs_entropy
    elif args.all_ratios_entropy_ratio is not None:
        assert len(run_configs) == 1, f"{run_configs=} is not a list of length 1"
        all_ratios_entropy_ratio_data = json.load(open(args.all_ratios_entropy_ratio))
        run_configs_entropy_ratio = []
        for entropy_ratio_quantile in all_ratios_entropy_ratio_data["quantiles"].keys():
            run_config = run_configs[0].copy()
            value = all_ratios_entropy_ratio_data["quantiles"][entropy_ratio_quantile]
            run_config["entropy_ratio_threshold"] = round(value, 3)
            run_configs_entropy_ratio.append(run_config)
        run_configs = run_configs_entropy_ratio
    # 验证配置
    if not ConfigValidator.validate_run_configs(run_configs):
        exit(1)
    
    # 加载和处理数据集
    test_ds = DatasetProcessor.load_and_process_dataset(
        args.dataset, args.split, args.add_sys_prompt
    )
    
    # 计算运行范围
    end_idx = int(len(test_ds) * (args.run_percent/100.)) if args.run_num == -1 else args.run_num
    print(f"[INFO]: {end_idx=}, {len(test_ds)=}")
    
    truncated_ds = test_ds[0:end_idx]
    print(f"{len(truncated_ds)=}")
    
    # 初始化模型实验运行器
    print(f"[INFO]: Loading models ({args.llm=}, {args.rm=})")
    experiment_runner = ModelExperimentRunner(args.llm, args.rm, args.vm, args.rm_lora, args.llm_gpu, args.rm_gpu)
    print(f"[INFO]: Done")
    
    # 运行每个配置
    for _, run_config in enumerate(run_configs):
        print(f"[INFO]: Running config: {run_config=}")
        if "Skywork-Reward-V2" not in rm_name and run_config["rm_weight"] != 0.0:
            assert args.rm_lora is not None, f"{args.rm_lora=} is None"
        
        # 生成保存目录
        base_dir = f"assets/generation_results/{model_name}/{dataset_name}"
        
        save_dir = experiment_runner.generate_save_directory(
            base_dir, model_name, rm_name, vm_name, rm_lora_name, dataset_name, run_config, args.add_sys_prompt
        )
        
        if not Path(save_dir).exists():
            Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        # 初始化数据收集
        text_shift_count = {}
        tokens_entropy_collect = {"shift": [], "no_shift": [], "all": [], "combined": []}
        tokens_entropy_ratio_collect = {"shift": [], "no_shift": [], "all": []}

        if os.path.exists(Path(save_dir + f"/results.json")) and run_config["continue_save_results"]:
            print(f"[INFO]: {save_dir} already exists, loading data...")
            data = json.load(open(Path(save_dir + f"/results.json")))
        else:
            data = []

        # 保存实验设置：合并run_config和args
        experiment_config = {
            **run_config,
            **vars(args)
        }
        with open(Path(save_dir + f"/experiment_config.json"), "w") as outfile:
            json.dump(experiment_config, outfile, ensure_ascii=False, indent=4)
        
        # 运行实验
        for idx, ds_row in enumerate(tqdm(truncated_ds)):
            if idx+1 <= len(data):
                print(f"[INFO]: {idx+1} already exists, skipping")
                continue
            print(f"{ds_row=}")
            current_prompt = ds_row
            start = time.time()
            run_prompt_results = experiment_runner.run_prompt(
                current_prompt,
                run_config["rm_weight"],
                run_config["topk"],
                args.max_new_token,
                run_config["sample_temp"],
                intervene_num=run_config["intervene_num"],
                entropy_threshold=run_config["entropy_threshold"],
                entropy_ratio_threshold=run_config["entropy_ratio_threshold"],
                rm_trajectory_len=run_config["rm_trajectory_len"],
                attention_threshold=run_config["attention_threshold"],
                attention_ratio_threshold=run_config["attention_ratio_threshold"],
                random_intervene_ratio=run_config["random_intervene_ratio"]
            )
            
            if run_prompt_results is None:
                print("[INFO]: Too long, skipped")
                continue
            else:
                tokens_shift = run_prompt_results["tokens_shift"]
                tokens_entropy = run_prompt_results["tokens_entropy"]
                tokens_entropy_ratio = run_prompt_results["tokens_entropy_ratio"]

            elapsed = time.time() - start

            # 保存结果
            data.append({
                "id": idx+1,
                "prompt": current_prompt, 
                "result": run_prompt_results["tokens_text_no_prefix"], 
                # "response": current_prompt + run_prompt_results["tokens_text_no_prefix"],
                "llm_prompt": run_prompt_results["llm_prompt"],
                "rm_prompt": run_prompt_results["rm_prompt"],
                "llm_prompt_tokens_ids": ",".join(map(str, run_prompt_results["llm_prompt_tokens_ids"])),
                "rm_prompt_tokens_ids": ",".join(map(str, run_prompt_results["rm_prompt_tokens_ids"])),
                "generated_tokens_ids": ",".join(map(str, run_prompt_results["generated_tokens_ids"])),
                "rm_guided_tokens_mask": ",".join(map(str, run_prompt_results["rm_guided_tokens_mask"])),
                "intervene_ratio": run_prompt_results["intervene_ratio"],
                "elapsed": elapsed,
            })
            
            # 处理token shift统计
            highlighted_tokens = []  # 保存所有token的列表
            highlighted_token_ids = [] # 保存所有token的id
            tokens_with_entropy = [] # 将每个token和对应的熵一起保存到列表中
            tokens_with_entropy_ratio = [] # 将每个token和对应的熵比值一起保存到列表中
            
            for token_idx, token_shift in enumerate(tokens_shift):
                first_token = experiment_runner.search.llm_tokenizer.convert_ids_to_tokens(token_shift)[0]
                last_token = experiment_runner.search.llm_tokenizer.convert_ids_to_tokens(token_shift)[-1]
                
                if first_token != last_token:
                    # 高亮最后一个token（使用红色背景）
                    # highlighted_token = f" \033[41m{last_token}\033[0m"
                    highlighted_token = f" [{first_token} -> {last_token}]"
                    highlighted_tokens.append(highlighted_token)
                    highlighted_token_ids.append(f"{token_shift[0]}->{token_shift[-1]}")
                    
                    text_shift = " -> ".join(experiment_runner.search.llm_tokenizer.convert_ids_to_tokens(token_shift))
                    if text_shift not in text_shift_count:
                        text_shift_count[text_shift] = 0
                    text_shift_count[text_shift] += 1
                else:
                    # 不高亮，直接添加原token
                    highlighted_tokens.append(last_token)
                    highlighted_token_ids.append(f"{token_shift[-1]}")
                
                tokens_with_entropy.append(f"{last_token}({tokens_entropy['all'][token_idx]:.3f})")
                tokens_with_entropy_ratio.append(f"{last_token}({tokens_entropy_ratio['all'][token_idx]:.3f})")


            with open(Path(save_dir + f"/results.json"), "w") as outfile:
                json.dump(data, outfile, ensure_ascii=False, indent=4)
            
            # 收集熵值数据
            tokens_entropy_collect["shift"].extend(tokens_entropy["shift"])
            tokens_entropy_collect["no_shift"].extend(tokens_entropy["no_shift"])
            tokens_entropy_collect["all"].extend(tokens_entropy["shift"] + tokens_entropy["no_shift"])
            tokens_entropy_collect["combined"].extend(tokens_entropy["combined"])
            
            tokens_entropy_ratio_collect["shift"].extend(tokens_entropy_ratio["shift"])
            tokens_entropy_ratio_collect["no_shift"].extend(tokens_entropy_ratio["no_shift"])
            tokens_entropy_ratio_collect["all"].extend(tokens_entropy_ratio["shift"] + tokens_entropy_ratio["no_shift"])

            if (idx+1) % 10 == 0:
                with open(Path(save_dir + f"/tokens_entropy_collect.json"), "w") as outfile:
                    json.dump(tokens_entropy_collect, outfile, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
