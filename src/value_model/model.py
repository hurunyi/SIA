import torch
import torch.nn as nn
import os
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoModelForCausalLM, AutoConfig


def load_base_model(model_path, model_type="auto", num_labels=1, torch_dtype=torch.bfloat16, device_map=None, trust_remote_code=True):
    """
    加载基础模型（支持AutoModelForSequenceClassification和AutoModelForCausalLM）
    
    Args:
        model_path: 模型路径
        model_type: 模型类型，"auto"（自动检测）、"sequence_classification"或"causal_lm"
        num_labels: 对于sequence_classification模型，标签数量
        torch_dtype: 模型数据类型
        device_map: 设备映射
        trust_remote_code: 是否信任远程代码
    
    Returns:
        model: 加载的模型
        actual_model_type: 实际使用的模型类型
    """
    if model_type == "auto":
        # 尝试自动检测模型类型
        try:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
            # 检查config中是否有相关属性来判断模型类型
            # 通常sequence classification模型会有num_labels配置
            # 而causal LM模型通常有vocab_size等属性
            if hasattr(config, 'architectures') and config.architectures:
                arch = config.architectures[0].lower()
                if 'sequenceclassification' in arch or 'forsequenceclassification' in arch:
                    model_type = "sequence_classification"
                elif 'causallm' in arch or 'forcausallanguage' in arch:
                    model_type = "causal_lm"
                else:
                    # 默认尝试sequence_classification
                    print(f"警告: 无法从架构名称 '{config.architectures[0]}' 确定模型类型，默认使用 sequence_classification")
                    model_type = "sequence_classification"
            else:
                # 如果无法确定，默认使用sequence_classification
                print("警告: 无法自动检测模型类型，默认使用 sequence_classification")
                model_type = "sequence_classification"
        except Exception as e:
            print(f"警告: 自动检测模型类型时出错: {e}，默认使用 sequence_classification")
            model_type = "sequence_classification"
    
    if model_type == "sequence_classification":
        print(f"加载 AutoModelForSequenceClassification 模型: {model_path}")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=num_labels,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            device_map=device_map,
            trust_remote_code=trust_remote_code
        )
        return model, "sequence_classification"
    elif model_type == "causal_lm":
        print(f"加载 AutoModelForCausalLM 模型: {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            device_map=device_map,
            trust_remote_code=trust_remote_code
        )
        return model, "causal_lm"
    else:
        raise ValueError(f"不支持的模型类型: {model_type}，请使用 'sequence_classification' 或 'causal_lm'")


# 创建一个简单的输出类，用于兼容search.py中的使用方式
class ValueModelOutput:
    """ValueModel的输出类，用于兼容AutoModelForSequenceClassification和AutoModelForCausalLM的输出格式"""
    def __init__(self, logits, token_rewards=None, hidden_states=None, past_key_values=None):
        self.logits = logits  # (batch_size, seq_len) 或 (batch_size, seq_len, vocab_size)
        self.token_rewards = token_rewards  # (batch_size, seq_len)
        self.hidden_states = hidden_states
        self.past_key_values = past_key_values


class ValueModel(nn.Module):
    """Token级别的Value Model"""
    
    def __init__(self, base_model, hidden_size, model_type="sequence_classification"):
        """
        Args:
            base_model: 基础RM模型（AutoModelForSequenceClassification 或 AutoModelForCausalLM）
            hidden_size: 隐藏层大小
            model_type: 模型类型，"sequence_classification" 或 "causal_lm"
        """
        super(ValueModel, self).__init__()
        self.base_model = base_model
        self.hidden_size = hidden_size
        self.model_type = model_type
        
        # 添加token级别的预测头
        # 从base_model的隐藏状态预测token级别的reward
        if hasattr(base_model, 'config'):
            self.hidden_size = base_model.config.hidden_size
            # 为了兼容search.py中的使用，需要暴露config属性
            self.config = base_model.config
        else:
            self.hidden_size = hidden_size
            self.config = None
        
        # Token级别的预测头（每个token位置输出一个reward分数）
        self.token_reward_head = nn.Linear(self.hidden_size, 1)
    
    @classmethod
    def from_pretrained(cls, base_model_path, model_path, torch_dtype=torch.bfloat16, device_map=None, model_type="auto"):
        """
        从保存的模型路径加载ValueModel模型
        
        Args:
            base_model_path: 基础模型路径
            model_path: 模型保存路径（包含model_config.json的目录）
            torch_dtype: 模型数据类型
            device_map: 设备映射
            model_type: 模型类型，"auto"（自动检测）、"sequence_classification"或"causal_lm"
        
        Returns:
            ValueModel: 加载的模型实例
        """
        import json
        
        # 尝试从model_config.json读取模型类型
        config_path = os.path.join(model_path, "model_config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                saved_config = json.load(f)
            # 如果保存的配置中有model_type，使用它
            if 'model_type' in saved_config:
                model_type = saved_config['model_type']
                print(f"从保存的配置中读取模型类型: {model_type}")
        
        # 加载基础RM模型
        base_model, actual_model_type = load_base_model(
            base_model_path,
            model_type=model_type,
            num_labels=1,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True
        )
        
        # 加载LoRA权重
        lora_weights_path = os.path.join(model_path, "lora_weights")
        if os.path.exists(lora_weights_path):
            base_model = PeftModel.from_pretrained(base_model, lora_weights_path, device_map=device_map)
            # 合并LoRA权重到基础模型
            base_model = base_model.merge_and_unload()
        else:
            print(f"警告: LoRA权重目录不存在: {lora_weights_path}，将使用基础模型")
        
        # 获取hidden_size
        if hasattr(base_model, 'config') and hasattr(base_model.config, 'hidden_size'):
            hidden_size = base_model.config.hidden_size
        else:
            # 尝试从保存的配置中读取
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                hidden_size = config.get('hidden_size', 4096)  # 默认值
            else:
                hidden_size = 4096  # 默认值
                print(f"警告: 无法确定hidden_size，使用默认值: {hidden_size}")
        
        # 创建ValueModel实例
        model = cls(base_model, hidden_size, model_type=actual_model_type)
        
        # 加载token_reward_head权重
        head_path = os.path.join(model_path, "token_reward_head.pt")
        if os.path.exists(head_path):
            head_state = torch.load(head_path, map_location='cpu')
            model.token_reward_head.load_state_dict(head_state['token_reward_head'])
            print(f"已加载token_reward_head权重: {head_path}")
        else:
            print(f"警告: token_reward_head权重文件不存在: {head_path}，将使用随机初始化的权重")
        
        return model
    
    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, **kwargs):
        """
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            past_key_values: 用于KV缓存的过去键值对（如果支持）
            use_cache: 是否使用缓存
            **kwargs: 其他参数
        
        Returns:
            ValueModelOutput: 包含logits、token_rewards、hidden_states等属性的输出对象
        """
        # 获取base_model的输出
        # AutoModelForSequenceClassification的底层模型通常是model或transformer
        # AutoModelForCausalLM通常直接就是transformer模型，或者也有model属性
        if self.model_type == "causal_lm":
            # 对于CausalLM，通常直接使用模型本身，或者通过model属性访问
            if hasattr(self.base_model, 'model'):
                base_transformer = self.base_model.model
            else:
                # 如果没有model属性，直接使用base_model
                base_transformer = self.base_model
        else:
            # 对于SequenceClassification，通常通过model或transformer属性访问
            if hasattr(self.base_model, 'model'):
                base_transformer = self.base_model.model
            elif hasattr(self.base_model, 'transformer'):
                base_transformer = self.base_model.transformer
            else:
                # 如果没有找到，尝试直接使用base_model
                base_transformer = self.base_model
        
        # 准备transformer的输入参数
        transformer_kwargs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'output_hidden_states': True
        }
        
        # 如果支持past_key_values，则添加
        if past_key_values is not None and hasattr(base_transformer, 'forward'):
            # 检查forward签名是否支持past_key_values
            import inspect
            sig = inspect.signature(base_transformer.forward)
            if 'past_key_values' in sig.parameters:
                transformer_kwargs['past_key_values'] = past_key_values
                transformer_kwargs['use_cache'] = use_cache
        
        outputs = base_transformer(**transformer_kwargs)
        
        # 使用最后一层的隐藏状态
        if isinstance(outputs, tuple):
            hidden_states = outputs[0]  # 第一个元素通常是last_hidden_state
            past_key_values_out = outputs[1] if len(outputs) > 1 and use_cache else None
        else:
            hidden_states = outputs.hidden_states[-1] if hasattr(outputs, 'hidden_states') else outputs.last_hidden_state
            past_key_values_out = outputs.past_key_values if hasattr(outputs, 'past_key_values') and use_cache else None
        
        # 预测每个token位置的reward
        token_rewards = self.token_reward_head(hidden_states.float())  # (batch_size, seq_len, 1)
        token_rewards = token_rewards.squeeze(-1)  # (batch_size, seq_len)
        
        # 为了兼容search.py中的使用方式，logits应该是最后一个token的reward
        # 在序列分类模式下，代码期望 logits 的形状是 (batch_size, 1)，然后可以 flatten()
        # 我们返回最后一个token的reward作为logits
        # token_rewards 的形状是 (batch_size, seq_len)
        # 取最后一个token的reward，并保持维度为 (batch_size, 1)
        logits = token_rewards[:, -1].unsqueeze(-1)  # (batch_size,) -> (batch_size, 1)
        
        return ValueModelOutput(
            logits=logits,
            token_rewards=token_rewards,
            hidden_states=hidden_states,
            past_key_values=past_key_values_out
        )
