import torch
import re
from typing import List, Dict


class ConversationProcessor:
    """对话处理器，负责对话格式转换和数据提取"""
    
    @staticmethod
    def parse_conversation_to_format(text: str, add_system_prompt: bool = False) -> List[Dict[str, str]]:
        """
        将对话文本解析为用户/回答角色的格式，适用于Skywork RM
        """
        conversations = []

        if add_system_prompt:
            conversations.append({"role": "system", "content": ""})
        
        text = text.strip()
        
        parts = re.split(r'(Human|Assistant):\s*', text, flags=re.IGNORECASE)
        
        if len(parts) < 3:
            if text.strip().lower().startswith("human:"):
                return [{"role": "user", "content": text.strip()[6:].strip()}]
            else:
                return [{"role": "user", "content": text}]
        
        current_role = None
        for i, part in enumerate(parts):
            part = part.strip()
            if not part:
                continue
                
            if part.lower() == "human":
                current_role = "user"
            elif part.lower() == "assistant":
                current_role = "assistant"
            else:
                if current_role:
                    conversations.append({"role": current_role, "content": part})
                elif i == 0:
                    if part and add_system_prompt:
                        # 第一部分如果不为空，可能是初始内容
                        conversations[0]["content"] = part
        
        # 如果解析失败，返回原始文本作为用户输入
        if not conversations:
            conversations = [{"role": "user", "content": text}]
        
        return conversations


def compute_weighted_attention_sum(avg_attn, target_pos, window_size):
    """
    计算目标位置对之前所有token的加权attention之和
    
    Args:
        avg_attn: 平均的attention矩阵，形状 (batch_size, seq_len, seq_len)
        target_pos: 目标位置（从1开始，不包括第一个token）
        window_size: 窗口大小
    
    Returns:
        加权和，形状 (batch_size,)
    """
    # 如果target_pos <= 1，没有之前的token，直接返回0
    if target_pos <= 1:
        return torch.zeros(avg_attn.shape[0], device=avg_attn.device, dtype=avg_attn.dtype)
    
    # 一次性获取所有之前token的attention值
    # 形状: (batch_size, target_pos-1)
    attn_values = avg_attn[:, target_pos, 1:target_pos]
    
    # 计算权重向量：对于每个位置s，权重为 min(target_pos - s, window_size)
    # s 从 1 到 target_pos-1，所以 target_pos - s 从 target_pos-1 到 1
    distances = torch.arange(target_pos - 1, 0, -1, device=avg_attn.device, dtype=avg_attn.dtype)
    weights = torch.clamp(distances, max=window_size)  # 限制在window_size以内
    
    # 计算加权和：对每个batch，将attention值与权重相乘后求和
    # weights需要扩展维度以匹配batch维度: (1, target_pos-1)
    weighted_sum = torch.sum(attn_values * weights.unsqueeze(0), dim=1)
    
    return weighted_sum
