"""
训练token级别的value model脚本

该脚本从一个trajectory level的RM模型初始化，使用LoRA方式进行高效微调，
使其能够预测mask token处的RM分数。

数据格式：
- rm_prompt_tokens_ids: 逗号分隔的token id列表
- generated_tokens_ids: 逗号分隔的token id列表
- rm_guided_tokens_mask: 与generated_tokens_ids对应的mask（1表示需要计算reward的位置）
- reward: 每个case对应的reward值（标量）
"""

import os
import json
import argparse
import random
from glob import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from collections import namedtuple
from .model import ValueModel, load_base_model


class ValueModelDataset(Dataset):
    """数据集类，用于加载token级别的RM训练数据"""
    
    def __init__(self, json_files, tokenizer, max_length=2048, stop_ids=None):
        """
        Args:
            json_files: JSON文件路径列表，包含训练数据
            tokenizer: tokenizer对象
            max_length: 最大序列长度
            stop_ids: 停止token id列表
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stop_ids = stop_ids
        
        # 加载数据
        self.data = []
        for file in json_files:
            with open(file, 'r', encoding='utf-8') as f:
                file_data = json.load(f)
                self.data.extend(file_data)
        
        # 处理数据
        self.processed_data = []
        for item in tqdm(self.data, desc="处理数据"):
            if not item['generated_tokens_ids']:
                print("没有generated_tokens_ids，跳过该样本")
                continue
            try:
                # 解析token ids
                rm_prompt_ids = [int(x.strip()) for x in item['rm_prompt_tokens_ids'].split(',') if x.strip()]
                generated_ids = [int(x.strip()) for x in item['generated_tokens_ids'].split(',') if x.strip()]
                reward = float(item['reward'])
                
                # 原始mask
                guided_mask = [int(x.strip()) for x in item['rm_guided_tokens_mask'].split(',') if x.strip()]
                
                if generated_ids[-1] not in self.stop_ids:
                    generated_ids.append(random.choice(self.stop_ids))
                    guided_mask.append(1)
                
                if len(guided_mask) != len(generated_ids):
                    print(f"警告: rm_guided_tokens_mask长度({len(guided_mask)})与generated_ids长度({len(generated_ids)})不匹配，跳过该样本")
                    continue
                
                mask = guided_mask
                token_loss_weights = [float(m) for m in mask]
                
                # 拼接prompt和generated tokens
                input_ids = rm_prompt_ids + generated_ids
                
                # mask的长度应该等于generated_ids的长度
                if len(mask) != len(generated_ids):
                    print(f"警告: mask长度({len(mask)})与generated_ids长度({len(generated_ids)})不匹配，跳过该样本")
                    continue
                
                # 创建完整的mask（prompt部分全为0，generated部分使用mask）
                full_mask = [0] * len(rm_prompt_ids) + mask
                full_loss_weights = [0.0] * len(rm_prompt_ids) + token_loss_weights
                
                # 截断到最大长度
                if len(input_ids) > max_length:
                    input_ids = input_ids[:max_length]
                    full_mask = full_mask[:max_length]
                    full_loss_weights = full_loss_weights[:max_length]
                
                # 确保mask长度与input_ids一致
                if len(full_mask) != len(input_ids):
                    full_mask = full_mask[:len(input_ids)]
                if len(full_loss_weights) != len(input_ids):
                    full_loss_weights = full_loss_weights[:len(input_ids)]
                
                self.processed_data.append({
                    'input_ids': input_ids,
                    'reward_mask': full_mask,  # 标记哪些位置需要预测reward
                    'token_loss_weights': full_loss_weights,
                    'reward': reward,
                    'prompt_len': len(rm_prompt_ids),
                    'generated_len': len(generated_ids)
                })
            except Exception as e:
                continue
        
        print(f"\n数据集加载完成:")
        print(f"  总样本数: {len(self.processed_data)}")
        if len(self.processed_data) > 0:
            rewards = [x['reward'] for x in self.processed_data]
            print(f"  Reward范围: [{min(rewards):.4f}, {max(rewards):.4f}]")
            print(f"  Reward均值: {np.mean(rewards):.4f}, 标准差: {np.std(rewards):.4f}")
            avg_mask_ratio = np.mean([np.mean(x['reward_mask']) for x in self.processed_data])
            print(f"  平均mask比例: {avg_mask_ratio:.4f}")
    
    def __len__(self):
        return len(self.processed_data)
    
    def __getitem__(self, idx):
        item = self.processed_data[idx]
        return {
            'input_ids': torch.LongTensor(item['input_ids']),
            'reward_mask': torch.LongTensor(item['reward_mask']),
            'token_loss_weights': torch.FloatTensor(item['token_loss_weights']),
            'reward': torch.FloatTensor([item['reward']]),
            'prompt_len': item['prompt_len'],
            'generated_len': item['generated_len']
        }


def collate_fn(batch):
    """自定义collate函数，处理变长序列"""
    # 找到最大长度
    max_len = max([len(item['input_ids']) for item in batch])
    
    # 填充
    input_ids = []
    reward_masks = []
    loss_weights = []
    rewards = []
    prompt_lens = []
    generated_lens = []
    
    for item in batch:
        seq_len = len(item['input_ids'])
        pad_len = max_len - seq_len
        
        # 填充input_ids（使用pad_token_id，如果没有则使用0）
        padded_input_ids = torch.cat([
            item['input_ids'],
            torch.zeros(pad_len, dtype=torch.long)
        ])
        input_ids.append(padded_input_ids)
        
        # 填充reward_mask
        padded_reward_mask = torch.cat([
            item['reward_mask'],
            torch.zeros(pad_len, dtype=torch.long)
        ])
        reward_masks.append(padded_reward_mask)
        
        rewards.append(item['reward'])
        prompt_lens.append(item['prompt_len'])
        generated_lens.append(item['generated_len'])
        
        # 填充token_loss_weights
        padded_loss_weights = torch.cat([
            item['token_loss_weights'],
            torch.zeros(pad_len, dtype=torch.float)
        ])
        loss_weights.append(padded_loss_weights)
    
    return {
        'input_ids': torch.stack(input_ids),
        'reward_mask': torch.stack(reward_masks),
        'loss_weights': torch.stack(loss_weights),
        'rewards': torch.stack(rewards),
        'prompt_lens': prompt_lens,
        'generated_lens': generated_lens
    }


def train_epoch(model, dataloader, criterion, optimizer, device, global_step=0, gradient_accumulation_steps=1):
    """训练一个epoch
    
    Args:
        model: 模型
        dataloader: 数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 设备
        global_step: 全局step计数器
        gradient_accumulation_steps: 梯度累积步数
    
    Yields:
        (step, loss, mae, mse, rmse, r2): 每个累积步骤的指标
    """
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    num_samples = 0
    num_batches = 0
    num_optimization_steps = 0  # 实际优化器更新的次数
    
    # 梯度累积相关变量
    accumulated_loss = 0.0
    accumulated_preds = []
    accumulated_labels = []
    accumulation_counter = 0
    
    # 创建tqdm进度条，用于显示实时训练信息
    pbar = tqdm(dataloader, desc="训练中")
    
    for batch_idx, batch in enumerate(pbar):
        input_ids = batch['input_ids'].to(device)
        reward_mask = batch['reward_mask'].to(device)  # (batch_size, seq_len)
        loss_weights = batch['loss_weights'].to(device)  # (batch_size, seq_len)
        rewards = batch['rewards'].to(device)  # (batch_size, 1)

        # 前向传播
        outputs = model(input_ids)
        token_rewards = outputs.token_rewards  # (batch_size, seq_len)
        
        # 只在mask位置计算损失
        # 对于每个样本，在mask=1的位置，预测值应该接近全局reward
        batch_size = token_rewards.size(0)
        batch_losses = []
        batch_preds = []  # 用于step级别指标计算
        batch_labels = []  # 用于step级别指标计算
        
        for i in range(batch_size):
            # 获取该样本的mask位置
            mask_positions = reward_mask[i] > 0  # (seq_len,)
            
            if mask_positions.sum() == 0:
                # 如果没有mask位置，跳过该样本
                continue
            
            # 获取mask位置的预测值
            masked_predictions = token_rewards[i][mask_positions]  # (num_masked,)
            masked_weights = loss_weights[i][mask_positions]  # (num_masked,)
            if masked_weights.sum() == 0:
                continue
            
            # 计算加权预测均值
            weight_sum = masked_weights.sum()
            avg_prediction = (masked_predictions * masked_weights).sum() / weight_sum
            target_reward = rewards[i].squeeze()
            
            # 计算损失
            loss = criterion(avg_prediction, target_reward)
            batch_losses.append(loss)
            
            # 保存用于step级别和epoch级别的评估
            batch_preds.append(avg_prediction.item())
            batch_labels.append(target_reward.item())
            all_preds.append(avg_prediction.item())
            all_labels.append(target_reward.item())
            num_samples += 1
        
        if len(batch_losses) == 0:
            continue
        
        # 平均损失
        loss = torch.stack(batch_losses).mean()
        
        # 将损失除以累积步数，以便在累积后得到正确的平均损失
        loss = loss / gradient_accumulation_steps
        
        # 反向传播（累积梯度）
        loss.backward()
        
        # 累积损失和指标
        accumulated_loss += loss.item() * gradient_accumulation_steps  # 恢复原始损失值用于显示
        accumulated_preds.extend(batch_preds)
        accumulated_labels.extend(batch_labels)
        accumulation_counter += 1
        
        # 更新统计信息（用于epoch级别计算）
        batch_loss = loss.item() * gradient_accumulation_steps  # 恢复原始损失值
        total_loss += batch_loss
        num_batches += 1
        
        # 当累积到指定步数时，执行优化器更新
        if accumulation_counter >= gradient_accumulation_steps:
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 优化器更新
            optimizer.step()
            optimizer.zero_grad()
            
            # 计算累积步骤的指标
            if len(accumulated_preds) > 0:
                batch_mae = mean_absolute_error(accumulated_labels, accumulated_preds)
                batch_mse = mean_squared_error(accumulated_labels, accumulated_preds)
                batch_rmse = np.sqrt(batch_mse)
                batch_r2 = r2_score(accumulated_labels, accumulated_preds) if len(accumulated_preds) > 1 else 0.0
            else:
                batch_mae = batch_mse = batch_rmse = batch_r2 = 0.0
            
            # 计算累积步骤的平均损失
            avg_accumulated_loss = accumulated_loss / accumulation_counter
            
            # 更新优化器步数
            num_optimization_steps += 1
            
            # 计算当前平均loss（基于所有已处理的batch）
            avg_loss_so_far = total_loss / num_batches if num_batches > 0 else 0
            
            # 获取当前学习率
            current_lr = optimizer.param_groups[0]['lr']
            
            # 更新tqdm显示信息
            pbar.set_postfix({
                'loss': f'{avg_accumulated_loss:.4f}',
                'avg_loss': f'{avg_loss_so_far:.4f}',
                'samples': num_samples,
                'lr': f'{current_lr:.2e}',
                'acc': f'{accumulation_counter}/{gradient_accumulation_steps}'
            })
            
            # 生成step级别的指标（在优化器更新后）
            current_step = global_step + num_optimization_steps
            yield ('step', current_step, avg_accumulated_loss, batch_mae, batch_mse, batch_rmse, batch_r2)
            
            # 重置累积变量
            accumulated_loss = 0.0
            accumulated_preds = []
            accumulated_labels = []
            accumulation_counter = 0
    
    # 处理最后一个不完整的累积批次（如果存在）
    if accumulation_counter > 0:
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 优化器更新
        optimizer.step()
        optimizer.zero_grad()
        
        # 计算最后一个累积步骤的指标
        if len(accumulated_preds) > 0:
            batch_mae = mean_absolute_error(accumulated_labels, accumulated_preds)
            batch_mse = mean_squared_error(accumulated_labels, accumulated_preds)
            batch_rmse = np.sqrt(batch_mse)
            batch_r2 = r2_score(accumulated_labels, accumulated_preds) if len(accumulated_preds) > 1 else 0.0
        else:
            batch_mae = batch_mse = batch_rmse = batch_r2 = 0.0
        
        # 计算最后一个累积步骤的平均损失
        avg_accumulated_loss = accumulated_loss / accumulation_counter
        
        # 更新优化器步数
        num_optimization_steps += 1
        
        # 生成step级别的指标
        current_step = global_step + num_optimization_steps
        yield ('step', current_step, avg_accumulated_loss, batch_mae, batch_mse, batch_rmse, batch_r2)
    
    # 计算epoch级别的指标
    if num_samples == 0:
        yield ('epoch', global_step + num_optimization_steps, 0, 0, 0, 0, 0)
        return
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    
    # 计算回归指标（epoch级别）
    mae = mean_absolute_error(all_labels, all_preds)
    mse = mean_squared_error(all_labels, all_preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(all_labels, all_preds)
    
    # 最后yield epoch级别的指标
    yield ('epoch', global_step + num_optimization_steps, avg_loss, mae, mse, rmse, r2)


def _plot_training_curves(train_history, val_history, plot_path, use_steps=False):
    """绘制训练曲线
    
    Args:
        train_history: 训练历史字典
        val_history: 验证历史字典
        plot_path: 保存路径
        use_steps: 是否使用step作为x轴（True）还是epoch（False）
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    metrics = ['loss', 'mae', 'rmse', 'r2']
    metric_names = ['Loss', 'MAE', 'RMSE', 'R²']
    
    for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax = axes[i // 2, i % 2]
        
        if use_steps:
            # 使用step作为x轴
            if 'step' in train_history and len(train_history['step']) > 0:
                ax.plot(train_history['step'], train_history[metric], 
                       label=f'Train {metric_name}', marker='o', markersize=3, linewidth=1)
            if 'step' in val_history and len(val_history['step']) > 0:
                ax.plot(val_history['step'], val_history[metric], 
                       label=f'Val {metric_name}', marker='s', markersize=3, linewidth=1)
            ax.set_xlabel('Step')
        else:
            # 使用epoch作为x轴
            if len(train_history[metric]) > 0:
                ax.plot(train_history[metric], label=f'Train {metric_name}', marker='o')
            if len(val_history[metric]) > 0:
                ax.plot(val_history[metric], label=f'Val {metric_name}', marker='s')
            ax.set_xlabel('Epoch')
        
        ax.set_ylabel(metric_name)
        ax.set_title(f'{metric_name} over {"steps" if use_steps else "epochs"}')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图形以释放内存


def save_checkpoint(model, output_dir, step, hidden_size, base_model_path, lora_r, lora_alpha, lora_dropout, model_type, train_loss=None, train_r2=None, val_loss=None, val_r2=None):
    """保存模型checkpoint
    
    Args:
        model: ValueModel模型
        output_dir: 输出目录
        step: 当前步数
        hidden_size: 隐藏层大小
        base_model_path: 基础模型路径
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        model_type: 模型类型
        train_loss: 训练损失（可选）
        train_r2: 训练R²（可选）
        val_loss: 验证损失（可选）
        val_r2: 验证R²（可选）
    """
    checkpoint_dir = os.path.join(output_dir, f"step{step}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 保存LoRA权重
    lora_output_dir = os.path.join(checkpoint_dir, "lora_weights")
    model.base_model.save_pretrained(lora_output_dir)
    
    # 保存token_reward_head
    head_path = os.path.join(checkpoint_dir, "token_reward_head.pt")
    torch.save({
        'token_reward_head': model.token_reward_head.state_dict(),
        'hidden_size': hidden_size,
    }, head_path)
    
    # 保存完整配置
    config_path = os.path.join(checkpoint_dir, "model_config.json")
    config = {
        'base_model_path': base_model_path,
        'lora_r': lora_r,
        'lora_alpha': lora_alpha,
        'lora_dropout': lora_dropout,
        'hidden_size': hidden_size,
        'model_type': model_type,
        'step': step,
    }
    if train_loss is not None:
        config['train_loss'] = train_loss
    if train_r2 is not None:
        config['train_r2'] = train_r2
    if val_loss is not None:
        config['val_loss'] = val_loss
    if val_r2 is not None:
        config['val_r2'] = val_r2
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Checkpoint已保存到: {checkpoint_dir}")


def evaluate(model, dataloader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    num_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="评估中"):
            input_ids = batch['input_ids'].to(device)
            reward_mask = batch['reward_mask'].to(device)
            loss_weights = batch['loss_weights'].to(device)
            rewards = batch['rewards'].to(device)
            
            # 前向传播
            outputs = model(input_ids)
            token_rewards = outputs.token_rewards  # (batch_size, seq_len)
            
            # 计算损失
            batch_size = token_rewards.size(0)
            batch_losses = []
            
            for i in range(batch_size):
                mask_positions = reward_mask[i] > 0
                
                if mask_positions.sum() == 0:
                    continue
                
                masked_predictions = token_rewards[i][mask_positions]
                masked_weights = loss_weights[i][mask_positions]
                if masked_weights.sum() == 0:
                    continue
                target_reward = rewards[i].squeeze()
                
                weight_sum = masked_weights.sum()
                avg_prediction = (masked_predictions * masked_weights).sum() / weight_sum
                loss = criterion(avg_prediction, target_reward)
                batch_losses.append(loss)
                
                all_preds.append(avg_prediction.item())
                all_labels.append(target_reward.item())
                num_samples += 1
            
            if len(batch_losses) == 0:
                continue
            
            loss = torch.stack(batch_losses).mean()
            total_loss += loss.item()
    
    if num_samples == 0:
        return 0, 0, 0, 0, 0
    
    avg_loss = total_loss / len(dataloader)
    
    # 计算回归指标
    mae = mean_absolute_error(all_labels, all_preds)
    mse = mean_squared_error(all_labels, all_preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(all_labels, all_preds)
    
    return avg_loss, mae, mse, rmse, r2


def main():
    parser = argparse.ArgumentParser(description="训练token级别的RM模型")
    parser.add_argument("--data_file", type=str, required=True, nargs='+', help="训练数据JSON文件路径（可指定多个文件）")
    parser.add_argument("--base_model_path", type=str, required=True, 
                       help="基础RM模型路径（如/mnt/ssd2/shiqian/models/Skywork-Reward-V2-Qwen3-1.7B）")
    parser.add_argument("--output_dir", type=str, default="./token_rm_output", help="输出目录")
    parser.add_argument("--batch_size", type=int, default=4, help="批次大小")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="学习率")
    parser.add_argument("--num_epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--max_length", type=int, default=2048, help="最大序列长度")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="训练集比例")
    parser.add_argument("--device", type=str, default="cuda:0", help="设备")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="权重衰减")
    parser.add_argument("--save_model", action="store_true", help="是否保存模型")
    parser.add_argument("--resume_from", type=str, default=None,
                       help="从已训练的模型路径继续训练（包含lora_weights和token_reward_head.pt的目录）")
    parser.add_argument("--plot_steps", type=int, default=None,
                       help="每隔多少step绘制一次训练曲线（如果为None，则每个epoch绘制一次）")
    parser.add_argument("--save_interval_steps", type=int, default=None,
                       help="每隔多少step保存一次模型checkpoint（如果为None，则不定期保存）")
    parser.add_argument("--model_type", type=str, default="auto",
                       choices=["auto", "sequence_classification", "causal_lm"],
                       help="基础模型类型：'auto'（自动检测）、'sequence_classification'（序列分类模型）或'causal_lm'（因果语言模型）")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                       help="梯度累积步数，用于在较小的批次大小下模拟较大的批次大小")
    
    args = parser.parse_args()
    
    print(f"参数设置: {args}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载tokenizer
    print(f"\n加载tokenizer: {args.base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载基础RM模型
    print(f"\n加载基础RM模型: {args.base_model_path}")
    print(f"模型类型: {args.model_type}")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    base_model, actual_model_type = load_base_model(
        args.base_model_path,
        model_type=args.model_type,
        num_labels=1,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True
    )
    print(f"实际使用的模型类型: {actual_model_type}")
    
    # 检查是否从已训练模型继续训练
    if args.resume_from is not None:
        print(f"\n从已训练模型继续训练: {args.resume_from}")
        resume_path = args.resume_from
        
        # 检查配置文件是否存在
        config_path = os.path.join(resume_path, "model_config.json")
        resume_config = None
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                resume_config = json.load(f)
            
            # 显示已训练模型的配置信息
            print(f"检测到已训练模型配置:")
            print(f"  LoRA r: {resume_config.get('lora_r', 'N/A')}")
            print(f"  LoRA alpha: {resume_config.get('lora_alpha', 'N/A')}")
            print(f"  LoRA dropout: {resume_config.get('lora_dropout', 'N/A')}")
            print(f"  最佳验证R²: {resume_config.get('best_val_r2', 'N/A')}")
            print(f"  最佳epoch: {resume_config.get('best_epoch', 'N/A')}")
            
            # 检查LoRA参数是否一致
            if resume_config.get('lora_r') != args.lora_r or \
               resume_config.get('lora_alpha') != args.lora_alpha or \
               resume_config.get('lora_dropout') != args.lora_dropout:
                print(f"\n警告: 命令行LoRA参数与已训练模型配置不一致！")
                print(f"  已训练模型: r={resume_config.get('lora_r')}, alpha={resume_config.get('lora_alpha')}, dropout={resume_config.get('lora_dropout')}")
                print(f"  命令行参数: r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
                print(f"  将使用命令行参数创建新的LoRA配置，这可能导致权重不兼容")
        
        # 配置LoRA（使用命令行参数）
        print(f"\n配置LoRA (r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout})")
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,  # 用于特征提取
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # 针对attention层
            bias="none",
        )
        
        # 应用LoRA
        model = get_peft_model(base_model, lora_config)
        
        # 加载LoRA权重
        lora_weights_path = os.path.join(resume_path, "lora_weights")
        if os.path.exists(lora_weights_path):
            print(f"加载LoRA权重: {lora_weights_path}")
            # 如果模型已经是PeftModel，需要先卸载旧的LoRA，然后加载新的
            # 或者直接加载权重
            try:
                model = PeftModel.from_pretrained(model, lora_weights_path)
                print("LoRA权重加载成功")
            except Exception as e:
                print(f"警告: 加载LoRA权重时出错: {e}")
                print("将使用新初始化的LoRA权重")
        else:
            print(f"警告: LoRA权重目录不存在: {lora_weights_path}，将使用新初始化的LoRA权重")
        
        model.print_trainable_parameters()
        
        # 创建token级别的RM模型
        hidden_size = base_model.config.hidden_size
        token_rm_model = ValueModel(model, hidden_size, model_type=actual_model_type).to(device)
        
        # 加载token_reward_head权重
        head_path = os.path.join(resume_path, "token_reward_head.pt")
        if os.path.exists(head_path):
            print(f"加载token_reward_head权重: {head_path}")
            try:
                head_state = torch.load(head_path, map_location='cpu')
                token_rm_model.token_reward_head.load_state_dict(head_state['token_reward_head'])
                print("token_reward_head权重加载成功")
            except Exception as e:
                print(f"警告: 加载token_reward_head权重时出错: {e}")
                print("将使用随机初始化的权重")
        else:
            print(f"警告: token_reward_head权重文件不存在: {head_path}，将使用随机初始化的权重")
        
        # LoRA参数已经通过get_peft_model自动设置为可训练
        # 我们只需要确保token_reward_head是可训练的
        for param in token_rm_model.token_reward_head.parameters():
            param.requires_grad = True
    else:
        # 从头开始训练
        # 配置LoRA
        print(f"\n配置LoRA (r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout})")
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,  # 用于特征提取
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # 针对attention层
            bias="none",
        )
        
        # 应用LoRA
        model = get_peft_model(base_model, lora_config)
        model.print_trainable_parameters()
        
        # 创建token级别的RM模型
        hidden_size = base_model.config.hidden_size
        token_rm_model = ValueModel(model, hidden_size, model_type=actual_model_type).to(device)
        
        # LoRA参数已经通过get_peft_model自动设置为可训练
        # 我们只需要确保token_reward_head是可训练的
        for param in token_rm_model.token_reward_head.parameters():
            param.requires_grad = True
    
    # 统计参数数量
    total_params = sum(p.numel() for p in token_rm_model.parameters())
    trainable_params = sum(p.numel() for p in token_rm_model.parameters() if p.requires_grad)
    print(f"\n总参数数: {total_params:,}")
    print(f"可训练参数数: {trainable_params:,}")
    print(f"可训练参数比例: {100 * trainable_params / total_params:.2f}%")
    
    # 加载数据集
    print(f"\n加载数据集: {args.data_file}")
    # 处理多个文件路径，支持通配符
    json_files = []
    
    # 展开通配符并建立对应关系
    for data_file in args.data_file:
        # 支持通配符（将XXX替换为*）
        expanded_files = glob(data_file.replace("XXX", "*"))
        if expanded_files:
            json_files.extend(expanded_files)
        else:
            # 如果没有匹配到通配符文件，尝试直接使用原路径
            if os.path.exists(data_file):
                json_files.append(data_file)
            else:
                print(f"警告: 文件路径不存在: {data_file}")
    
    if not json_files:
        raise ValueError(f"未找到任何数据文件，请检查文件路径: {args.data_file}")
    
    print(f"找到 {len(json_files)} 个数据文件:")
    for f in json_files:
        print(f"  - {f}")
    
    if "Qwen3" in args.base_model_path:
        stop_ids = [tokenizer.convert_tokens_to_ids("<|endoftext|>"), tokenizer.convert_tokens_to_ids("<|im_end|>")]
    elif "Llama-3" in args.base_model_path:
        stop_ids = [tokenizer.convert_tokens_to_ids("<|end_of_text|>"), tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    else:
        raise ValueError(f"不支持的模型类型: {args.base_model_path}")
    full_dataset = ValueModelDataset(
        json_files, 
        tokenizer, 
        max_length=args.max_length,
        stop_ids=stop_ids
    )
    
    # 划分训练集和验证集
    train_size = int(args.train_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        token_rm_model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # 训练历史
    # 如果使用step级别绘制，则记录step和对应的指标
    # 如果使用epoch级别绘制，则记录epoch和对应的指标
    if args.plot_steps is not None:
        train_history = {'step': [], 'loss': [], 'mae': [], 'mse': [], 'rmse': [], 'r2': []}
        val_history = {'step': [], 'loss': [], 'mae': [], 'mse': [], 'rmse': [], 'r2': []}
        print(f"训练曲线将每隔 {args.plot_steps} 个step绘制一次")
    else:
        train_history = {'loss': [], 'mae': [], 'mse': [], 'rmse': [], 'r2': []}
        val_history = {'loss': [], 'mae': [], 'mse': [], 'rmse': [], 'r2': []}
        print(f"训练曲线将在每个epoch结束后绘制")
    
    # 训练循环
    best_val_r2 = -float('inf')
    best_model_state = None
    best_epoch = 0
    global_step = 0  # 全局step计数器
    
    print(f"\n开始训练...")
    print(f"批次大小: {args.batch_size}")
    print(f"梯度累积步数: {args.gradient_accumulation_steps}")
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps
    print(f"有效批次大小: {effective_batch_size} (batch_size × gradient_accumulation_steps)")
    if args.save_interval_steps is not None:
        print(f"模型checkpoint将每隔 {args.save_interval_steps} 个step保存一次")
    
    for epoch in range(args.num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{args.num_epochs}")
        print(f"{'='*60}")
        
        # 训练（使用生成器获取step级别的指标）
        epoch_train_losses = []
        epoch_train_maes = []
        epoch_train_mses = []
        epoch_train_rmses = []
        epoch_train_r2s = []
        
        # 使用生成器获取每个step的指标
        train_gen = train_epoch(token_rm_model, train_loader, criterion, optimizer, device, global_step, args.gradient_accumulation_steps)
        
        train_loss = train_mae = train_mse = train_rmse = train_r2 = 0.0
        
        for item in train_gen:
            item_type, step, loss_val, mae_val, mse_val, rmse_val, r2_val = item
            
            if item_type == 'step':
                # step级别的指标
                global_step = step
                epoch_train_losses.append(loss_val)
                epoch_train_maes.append(mae_val)
                epoch_train_mses.append(mse_val)
                epoch_train_rmses.append(rmse_val)
                epoch_train_r2s.append(r2_val)
                
                # 如果设置了plot_steps，则每隔一定step记录并绘制
                if args.plot_steps is not None and step % args.plot_steps == 0:
                    train_history['step'].append(step)
                    train_history['loss'].append(loss_val)
                    train_history['mae'].append(mae_val)
                    train_history['mse'].append(mse_val)
                    train_history['rmse'].append(rmse_val)
                    train_history['r2'].append(r2_val)
                    
                    # 绘制训练曲线
                    if len(train_history['step']) > 0:
                        plot_path = os.path.join(args.output_dir, "training_curves.png")
                        _plot_training_curves(train_history, val_history, plot_path, use_steps=True)
                
                # 如果设置了save_interval_steps，则每隔一定step保存checkpoint
                if args.save_interval_steps is not None and step % args.save_interval_steps == 0 and step > 0:
                    save_checkpoint(
                        token_rm_model,
                        args.output_dir,
                        step,
                        hidden_size,
                        args.base_model_path,
                        args.lora_r,
                        args.lora_alpha,
                        args.lora_dropout,
                        actual_model_type,
                        train_loss=loss_val,
                        train_r2=r2_val
                    )
            
            elif item_type == 'epoch':
                # epoch级别的指标（最后一个yield的值）
                train_loss = loss_val
                train_mae = mae_val
                train_mse = mse_val
                train_rmse = rmse_val
                train_r2 = r2_val
        
        # 如果没有获取到epoch级别的指标，使用step级别的平均值
        if train_loss == 0.0 and len(epoch_train_losses) > 0:
            train_loss = np.mean(epoch_train_losses)
            train_mae = np.mean(epoch_train_maes)
            train_mse = np.mean(epoch_train_mses)
            train_rmse = np.mean(epoch_train_rmses)
            train_r2 = np.mean(epoch_train_r2s)
        
        # 如果使用epoch级别绘制，则记录epoch级别的指标
        if args.plot_steps is None:
            train_history['loss'].append(train_loss)
            train_history['mae'].append(train_mae)
            train_history['mse'].append(train_mse)
            train_history['rmse'].append(train_rmse)
            train_history['r2'].append(train_r2)
        
        print(f"训练 - Loss: {train_loss:.4f}, MAE: {train_mae:.4f}, "
              f"RMSE: {train_rmse:.4f}, R²: {train_r2:.4f}")
        
        # 验证
        val_loss, val_mae, val_mse, val_rmse, val_r2 = evaluate(
            token_rm_model, val_loader, criterion, device
        )
        
        # 如果使用step级别绘制，验证指标也记录到对应的step
        if args.plot_steps is not None:
            val_history['step'].append(global_step)
            val_history['loss'].append(val_loss)
            val_history['mae'].append(val_mae)
            val_history['mse'].append(val_mse)
            val_history['rmse'].append(val_rmse)
            val_history['r2'].append(val_r2)
        else:
            val_history['loss'].append(val_loss)
            val_history['mae'].append(val_mae)
            val_history['mse'].append(val_mse)
            val_history['rmse'].append(val_rmse)
            val_history['r2'].append(val_r2)
        
        print(f"验证 - Loss: {val_loss:.4f}, MAE: {val_mae:.4f}, "
              f"RMSE: {val_rmse:.4f}, R²: {val_r2:.4f}")
        
        # 保存最佳模型
        if val_r2 > best_val_r2:
            improvement = val_r2 - best_val_r2
            best_val_r2 = val_r2
            best_epoch = epoch + 1
            best_model_state = {
                'model_state_dict': token_rm_model.state_dict(),
                'base_model_path': args.base_model_path,
                'lora_config': lora_config,
            }
            print(f"✓ 新的最佳模型 (R²: {best_val_r2:.4f}, 改善: {improvement:.4f})")

            # 保存模型
            if args.save_model:
                # 保存LoRA权重
                lora_output_dir = os.path.join(args.output_dir, "lora_weights")
                token_rm_model.base_model.save_pretrained(lora_output_dir)
                print(f"LoRA权重已保存到: {lora_output_dir}")
                
                # 保存token_reward_head
                head_path = os.path.join(args.output_dir, "token_reward_head.pt")
                torch.save({
                    'token_reward_head': token_rm_model.token_reward_head.state_dict(),
                    'hidden_size': hidden_size,
                }, head_path)
                print(f"Token reward head已保存到: {head_path}")
                
                # 保存完整配置
                config_path = os.path.join(args.output_dir, "model_config.json")
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        'base_model_path': args.base_model_path,
                        'lora_r': args.lora_r,
                        'lora_alpha': args.lora_alpha,
                        'lora_dropout': args.lora_dropout,
                        'hidden_size': hidden_size,
                        'model_type': actual_model_type,
                        'best_val_r2': best_val_r2,
                        'best_epoch': best_epoch,
                    }, f, indent=2, ensure_ascii=False)
                print(f"模型配置已保存到: {config_path}")
        else:
            print(f"  (无改善，当前最佳R²: {best_val_r2:.4f}, 最佳epoch: {best_epoch})")
        
        # 保存训练历史
        history_path = os.path.join(args.output_dir, "training_history.json")
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump({
                'train': train_history,
                'val': val_history,
                'use_steps': args.plot_steps is not None
            }, f, indent=2, ensure_ascii=False)
        print(f"训练历史已保存到: {history_path}")
        
        # 绘制训练曲线（如果使用epoch级别绘制，则每个epoch绘制一次）
        if args.plot_steps is None:
            plot_path = os.path.join(args.output_dir, "training_curves.png")
            _plot_training_curves(train_history, val_history, plot_path, use_steps=False)
            print(f"训练曲线已保存到: {plot_path}")
    
    # 加载最佳模型
    if best_model_state is not None:
        token_rm_model.load_state_dict(best_model_state['model_state_dict'])
        print(f"\n已加载最佳模型 (验证R²: {best_val_r2:.4f}, Epoch {best_epoch})")
    
    # 保存模型
    if args.save_model:
        # 保存LoRA权重
        lora_output_dir = os.path.join(args.output_dir, "lora_weights")
        token_rm_model.base_model.save_pretrained(lora_output_dir)
        print(f"LoRA权重已保存到: {lora_output_dir}")
        
        # 保存token_reward_head
        head_path = os.path.join(args.output_dir, "token_reward_head.pt")
        torch.save({
            'token_reward_head': token_rm_model.token_reward_head.state_dict(),
            'hidden_size': hidden_size,
        }, head_path)
        print(f"Token reward head已保存到: {head_path}")
        
        # 保存完整配置
        config_path = os.path.join(args.output_dir, "model_config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump({
                'base_model_path': args.base_model_path,
                'lora_r': args.lora_r,
                'lora_alpha': args.lora_alpha,
                'lora_dropout': args.lora_dropout,
                'hidden_size': hidden_size,
                'model_type': actual_model_type,
                'best_val_r2': best_val_r2,
                'best_epoch': best_epoch,
            }, f, indent=2, ensure_ascii=False)
        print(f"模型配置已保存到: {config_path}")
    
    # 保存训练历史
    history_path = os.path.join(args.output_dir, "training_history.json")
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump({
            'train': train_history,
            'val': val_history,
            'use_steps': args.plot_steps is not None
        }, f, indent=2, ensure_ascii=False)
    print(f"训练历史已保存到: {history_path}")
    
    # 绘制最终训练曲线
    plot_path = os.path.join(args.output_dir, "training_curves.png")
    _plot_training_curves(train_history, val_history, plot_path, use_steps=(args.plot_steps is not None))
    print(f"训练曲线已保存到: {plot_path}")
    
    print(f"\n训练完成！")
    print(f"训练完成所有 {args.num_epochs} 个epoch")
    print(f"最佳验证R²: {best_val_r2:.4f} (Epoch {best_epoch})")


if __name__ == "__main__":
    main()

