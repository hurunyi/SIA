"""
重构的奖励分数测量脚本

该脚本使用 skywork 奖励模型(RM)计算对话数据的奖励分数。
主要功能包括：
1. 批量处理对话数据
2. 输出奖励统计和带奖励的原始数据
"""

import os
import json
import re
import argparse
from typing import List, Dict, Any, Optional, Union
from glob import glob
from dataclasses import dataclass

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from .utils import ConversationProcessor


@dataclass
class Config:
    """配置类，统一管理所有配置参数"""
    input_file: str
    output_file: str
    rm: str
    device: str = "cuda:0"
    
    @classmethod
    def from_args(cls) -> 'Config':
        """从命令行参数创建配置对象"""
        parser = argparse.ArgumentParser()
        parser.add_argument("--input_file", type=str, required=True)
        parser.add_argument("--output_file", type=str, required=True)
        parser.add_argument("--rm", type=str, default=None)
        parser.add_argument("--device", type=str, default="cuda:0")
        args = parser.parse_args()
        return cls(**vars(args))


class ModelManager:
    """模型管理器，负责不同RM模型的加载和初始化"""
    
    def __init__(self, config: Config):
        self.config = config
        self.rm_model = None
        self.rm_path = None
        
    def load_model(self):
        """加载Skywork RM模型和tokenizer"""
        self.rm_path = self.config.rm
        self.tokenizer = AutoTokenizer.from_pretrained(self.rm_path, use_fast=True)
        self.rm_model = AutoModelForSequenceClassification.from_pretrained(
            self.rm_path,
            dtype=torch.bfloat16,
            device_map=self.config.device,
            attn_implementation="flash_attention_2",
            num_labels=1,
        )
        print(f"使用Skywork RM模型: {self.rm_path}")
        return self.rm_model, self.tokenizer


class DataExtractor:
    """数据提取器，负责从输出数据中提取对话内容"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def extract_output(self, output_data: Dict[str, Any]):
        """
        根据RM类型提取对话内容
        """
        prompt_key = "prompt" if "prompt" in output_data.keys() else "instruction"
        # 提取原始输出
        if "result" in output_data:
            output = output_data["result"]
        elif "output" in output_data:
            output = output_data["output"]
        else:
            raise ValueError("输出数据中缺少 'result' 或 'output' 字段")

        # 处理输出
        output_np = output.removeprefix(output_data[prompt_key])
        
        if 'Human:\n' not in output_data[prompt_key]:
            output_data[prompt_key] = "Human:\n" + output_data[prompt_key] + "\nAssistant:\n"

        if output_np.startswith(": "): 
            output_np = output_np[2:]
        output_np = re.split("human:", output_np, flags=re.IGNORECASE)[0]
        output_np = re.split("assistant:", output_np, flags=re.IGNORECASE)[0]
        
        output = output_data[prompt_key] + output_np
        output = ConversationProcessor.parse_conversation_to_format(output, add_system_prompt=False)
        
        return output


class RewardCalculator:
    """奖励计算器，统一不同RM模型的奖励计算逻辑"""
    
    def __init__(self, rm_model, tokenizer, config: Config):
        self.rm_model = rm_model
        self.tokenizer = tokenizer
        self.config = config
    
    def calculate_reward(self, conversations: List[Dict[str, str]]) -> Optional[float]:
        """使用Skywork RM模型获取奖励分数"""
        conv_formatted = self.tokenizer.apply_chat_template(conversations, tokenize=False)

        # 移除潜在的重复bos token
        if self.tokenizer.bos_token is not None and conv_formatted.startswith(self.tokenizer.bos_token):
            conv_formatted = conv_formatted[len(self.tokenizer.bos_token):]
        
        # 分词化
        conv_tokenized = self.tokenizer(conv_formatted, return_tensors="pt").to(self.config.device)
        
        # 检查长度限制
        if conv_tokenized.input_ids.shape[1] >= 2048:
            return None
        
        # 获取奖励分数
        with torch.no_grad():
            rm_out = self.rm_model(**conv_tokenized)
            rm_val = rm_out.logits[0][0].item()
        
        del rm_out
        del conv_tokenized
        return rm_val


class RewardMeasurer:
    """主控制器，统一管理整个奖励测量流程"""
    
    def __init__(self, config: Config):
        self.config = config
        self.model_manager = ModelManager(config)
        self.data_extractor = DataExtractor(config)
    
    def run(self):
        """执行主要的奖励测量流程"""
        # 1. 加载模型
        rm_model, tokenizer = self.model_manager.load_model()
        
        # 2. 初始化组件
        reward_calculator = RewardCalculator(rm_model, tokenizer, self.config)
        
        # 3. 加载数据
        input_file = self.config.input_file
        print(f"处理文件: {input_file}")
        with open(input_file, "r") as f:
            lines = json.load(f)[:]
        
        print(f"文件 {input_file} 中共有 {len(lines)} 个样本")
        
        # 4. 处理数据并计算奖励
        rm_scores, lines_with_reward, num_skip, intervene_ratios = self._process_data(lines, reward_calculator)
        
        # 5. 保存结果
        output_file = self.config.output_file
        with open(output_file, "w") as f:
            json.dump(lines_with_reward, f, indent=4, ensure_ascii=False)
        
        print(f"原始数据+reward已保存到: {output_file}")
        print(f"共保存了 {len(lines_with_reward)} 个样本")
        
        # 6. 输出总结
        self._print_summary(rm_scores, num_skip, intervene_ratios)
    
    def _process_data(self, lines: List[Dict[str, Any]], reward_calculator: RewardCalculator) -> tuple:
        """处理数据并计算奖励分数"""
        rm_scores = []
        intervene_ratios = []
        lines_with_reward = []
        num_skip = 0
        
        for i, line in enumerate(tqdm(lines, desc="处理数据")):
            # 提取输出
            try:
                output = self.data_extractor.extract_output(line)
            except Exception as e:
                print(f"第{i+1}个样本数据提取失败: {e}")
                num_skip += 1
                continue
            
            # 创建当前行的副本
            line_copy = line.copy()
            
            # 检查输出是否为空
            if self._is_empty_output(output):
                rm_scores.append(0.0)
                line_copy["reward"] = 0.0
                # line_copy["test_rm_prompt"] = output
                intervene_ratios.append(line_copy["intervene_ratio"])
                lines_with_reward.append(line_copy)
                continue
            
            # 计算奖励分数
            reward = reward_calculator.calculate_reward(output)
            
            if reward is None:
                print("跳过一个样本")
                num_skip += 1
                continue
            
            rm_scores.append(reward)
            line_copy["reward"] = reward
            # line_copy["test_rm_prompt"] = output
            intervene_ratios.append(line_copy["intervene_ratio"])
            lines_with_reward.append(line_copy)
        
        return rm_scores, lines_with_reward, num_skip, intervene_ratios
    
    def _is_empty_output(self, output: Union[str, List[Dict[str, str]]]) -> bool:
        """检查输出是否为空"""
        return not output or len(output) == 0
    
    def _print_summary(self, rm_scores: List[float], num_skip: int, intervene_ratios: List[float]):
        """打印总结信息"""
        print(f"平均奖励分数: {np.mean(rm_scores)}")
        print(f"平均干预比例: {np.mean(intervene_ratios)}")
        print(f"跳过的样本数: {num_skip}")
        print(f"处理完成!")


def main():
    """主函数"""
    # 1. 加载配置
    config = Config.from_args()
    
    # 2. 创建主控制器并运行
    measurer = RewardMeasurer(config)
    measurer.run()


if __name__ == "__main__":
    main()
