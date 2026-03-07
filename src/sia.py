import os
import random
from typing import List
import torch
from torch.nn import functional as F
from tqdm import tqdm
import torch.distributions as dist
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
import re
from typing import Dict
from .value_model.model import ValueModel
from .utils import compute_weighted_attention_sum, ConversationProcessor


class SIA:
    def __init__(self, llm_path, rm_path, vm_path=None, rm_lora_path=None, llm_dev="cuda:0", rm_dev="cuda:1", torch_dtype=torch.float16):
        self.llm_dev = llm_dev
        self.rm_dev = rm_dev
        self.vm_dev = rm_dev
        self.torch_dtype = torch_dtype
        
        print(f"Loading LLM from {llm_path}...")
        self.LLM = AutoModelForCausalLM.from_pretrained(llm_path, torch_dtype=torch_dtype, low_cpu_mem_usage=True, attn_implementation="eager").to(self.llm_dev)
        self.LLM.eval()
        
        print(f"Loading tokenizer from {llm_path}...")
        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_path)
        self.rm_tokenizer = AutoTokenizer.from_pretrained(rm_path)
        self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
        self.llm_tokenizer.pad_token_id = self.llm_tokenizer.eos_token_id
        self.rm_tokenizer.pad_token = self.rm_tokenizer.eos_token
        self.rm_tokenizer.pad_token_id = self.rm_tokenizer.eos_token_id
        
        print(f"Loading RM from {rm_path}...")
        if rm_lora_path is not None:
            print(f"Loading LoRA from {rm_lora_path}...")
            self.RM = ValueModel.from_pretrained(
                base_model_path=rm_path,
                model_path=rm_lora_path, 
                torch_dtype=torch_dtype, 
                device_map=self.rm_dev
            ).to(self.rm_dev)
        else:
            self.RM = AutoModelForSequenceClassification.from_pretrained(rm_path, num_labels=1, torch_dtype=torch_dtype, low_cpu_mem_usage=True).to(self.rm_dev)
        
        self.RM.eval()

        self.LLM.config.pad_token_id = self.llm_tokenizer.eos_token_id
        self.RM.config.pad_token_id = self.rm_tokenizer.eos_token_id

        self.llm_type = os.path.basename(llm_path)
        self.rm_type = os.path.basename(rm_path)

        self.VM = None
        
    def get_llm_input_ids(self, prompt: str) -> torch.Tensor:
        conversations = ConversationProcessor.parse_conversation_to_format(prompt, add_system_prompt=False)
        # 格式化对话
        if "Qwen3" in self.llm_type:
            if "Base" not in self.llm_type:
                conv_formatted = self.llm_tokenizer.apply_chat_template(conversations, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            else:
                conv_formatted = prompt
        else:
            if "Instruct" in self.llm_type:
                conv_formatted = self.llm_tokenizer.apply_chat_template(conversations, tokenize=False, add_generation_prompt=True)
            else:
                conv_formatted = prompt
        # 移除潜在的重复bos token
        if self.llm_tokenizer.bos_token is not None and conv_formatted.startswith(self.llm_tokenizer.bos_token):
            conv_formatted = conv_formatted[len(self.llm_tokenizer.bos_token):]
        tokens = self.llm_tokenizer(conv_formatted, return_tensors="pt").input_ids.to(self.llm_dev)
        return tokens

    def get_rm_input_ids(self, prompt: str) -> torch.Tensor:
        conversations = ConversationProcessor.parse_conversation_to_format(prompt, add_system_prompt=False)
        # 格式化对话
        if "Qwen3" in self.rm_type:
            conv_formatted = f"<|im_start|>user\n{conversations[0]['content']}<|im_end|>\n<|im_start|>assistant\n"
        elif "Llama-3" in self.rm_type:
            conv_formatted = f"<|start_header_id|>user<|end_header_id|>\n\n{conversations[0]['content']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        else:
            raise ValueError(f"不支持的分词器: {self.rm_type}")
        
        # 移除潜在的重复bos token
        if self.rm_tokenizer.bos_token is not None and conv_formatted.startswith(self.rm_tokenizer.bos_token):
            conv_formatted = conv_formatted[len(self.rm_tokenizer.bos_token):]
        
        # 分词化
        tokens = self.rm_tokenizer(conv_formatted, return_tensors="pt").input_ids.to(self.rm_dev)
        
        return tokens
    
    def tokens_to_text(self, tokens: torch.Tensor) -> List[str]:
        return self.llm_tokenizer.batch_decode(tokens, skip_special_tokens=True)
        
    def generate_step(
        self,
        mout,
        generate_results,
        topk=40,
        weight=0.,
        temperature=0.7,
        token_idx=None,
        intervene_num=None,
        entropy_threshold=None,
        entropy_ratio_threshold=None,
        rm_trajectory_len=None,
        layer_idxs=None,
        window_size=10,
        attention_threshold=None,
        attention_ratio_threshold=None,
        intervene_list=None,
        random_intervene_ratio=None,
    ):  
        llm_input_ids = generate_results["llm_tokens"]
        rm_input_ids = generate_results["rm_tokens"]
        llm_prompt_input_ids = generate_results["llm_prompt_tokens"]
        rm_prompt_input_ids = generate_results["rm_prompt_tokens"]
        tokens_shift = generate_results["tokens_shift"]
        tokens_entropy = generate_results["tokens_entropy"]
        tokens_entropy_ratio = generate_results["tokens_entropy_ratio"]
        previous_entropy = generate_results["previous_entropy"]
        generated_tokens_ids = generate_results["generated_tokens_ids"]
        rm_guided_tokens_mask = generate_results["rm_guided_tokens_mask"]
        
        out_attentions = mout.attentions
        original_weight = weight
        weight = 0
            
        out_logits = mout.logits[:, -1]

        prescreen_logits, prescreen_tokens = torch.topk(out_logits, dim=-1, k=topk)

        actual_rm_topk = topk
        rm_prescreen_tokens = prescreen_tokens[:, :actual_rm_topk]

        expanded_llm_tis = torch.unsqueeze(llm_input_ids, 1).repeat(1, topk, 1)
        expanded_rm_tis = torch.unsqueeze(rm_input_ids, 1).repeat(1, actual_rm_topk, 1)

        to_llm_eval = torch.dstack((expanded_llm_tis, prescreen_tokens.to(self.llm_dev)))
        to_rm_eval = torch.dstack((expanded_rm_tis, rm_prescreen_tokens.to(self.rm_dev)))

        flat_llm_trme = to_llm_eval.view(out_logits.shape[0] * topk, -1)
        flat_rm_trme = to_rm_eval.view(out_logits.shape[0] * actual_rm_topk, -1)

        orig_scores = prescreen_logits.flatten()
        probs = F.softmax(orig_scores, dim=-1).float()
        categorical = dist.Categorical(probs=probs)
        entropy_ = categorical.entropy().item()
        token_entropy_ratio = np.log(entropy_ / previous_entropy) if previous_entropy is not None else -100

        if random_intervene_ratio is not None and random.random() <= random_intervene_ratio:
            weight = original_weight

        if intervene_num is not None and token_idx < intervene_num:
            weight = original_weight
        
        if entropy_threshold is not None and entropy_ >= entropy_threshold:
            weight = original_weight
        
        if entropy_ratio_threshold is not None and token_entropy_ratio >= entropy_ratio_threshold:
            weight = original_weight

        # 计算位置加权和
        if attention_threshold is not None or attention_ratio_threshold is not None:
            # 计算位置加权和
            # attentions 是一个元组，每个元素对应一个层
            # 形状: (batch_size, num_heads, seq_len, seq_len)
            if out_attentions is not None and len(out_attentions) > 0:
                if self.llm_type == "Llama-3.2-1B":
                    head_idxs = [7, 6, 30, 29, 1, 3, 31, 0, 2]
                elif self.llm_type == "Llama-3.2-3B":
                    head_idxs = [3, 0, 4, 2, 1, 23, 22]
                elif self.llm_type == "Llama-3.1-8B":
                    head_idxs = [19, 16, 18, 1, 29, 0, 30, 2, 31]
                elif self.llm_type == "Qwen3-0.6B-Base":
                    head_idxs = [9, 2, 7, 3]
                elif self.llm_type == "Qwen3-1.7B-Base":
                    head_idxs = [2, 3, 5, 7]
                elif self.llm_type == "Qwen3-4B-Base":
                    head_idxs = [28, 18, 24, 15, 26, 22, 1, 9, 2]
                elif self.llm_type == "Qwen3-8B-Base":
                    head_idxs = [18, 2, 10, 26, 15, 24, 22, 9, 1]
                else:
                    head_idxs = None
                
                # 选择要使用的层
                if layer_idxs is not None:
                    selected_layers = [out_attentions[idx] for idx in layer_idxs if idx < len(out_attentions)]
                else:
                    selected_layers = list(out_attentions)
                
                # 堆叠所有选中的层
                if len(selected_layers) > 0:
                    # 形状: (num_selected_layers, batch_size, num_heads, seq_len, seq_len)
                    stacked_attn = torch.stack(selected_layers, dim=0)
                    
                    # 选择要使用的头
                    if head_idxs is not None:
                        # 形状: (num_selected_layers, batch_size, num_selected_heads, seq_len, seq_len)
                        selected_heads = stacked_attn[:, :, head_idxs, :, :]
                    else:
                        selected_heads = stacked_attn
                    
                    # 平均所有选中的层和头
                    avg_attn = selected_heads.mean(dim=(0, 2))  # 形状: (batch_size, seq_len, seq_len)
                else:
                    avg_attn = None
                
                # 获取当前序列长度
                current_seq_len = avg_attn.shape[-1]
                i = current_seq_len - 1  # 当前位置

                if attention_threshold is not None:
                    weighted_sum_i = compute_weighted_attention_sum(avg_attn, i, window_size)
                    if weighted_sum_i.mean().item() >= attention_threshold:
                        weight = original_weight
                
                if attention_ratio_threshold is not None:
                    if i >= 1:  # 需要至少一个前导 token
                        # 计算位置 i 和 i-1 的加权attention和
                        # weighted_sum_current = compute_weighted_attention_sum(avg_attn, i, window_size)
                        weighted_sum_i = compute_weighted_attention_sum(avg_attn, i, window_size)
                        weighted_sum_i_minus_1 = compute_weighted_attention_sum(avg_attn, i - 1, window_size)
                        
                        ratio = weighted_sum_i / (weighted_sum_i_minus_1 + 1e-8)
                        
                        if ratio.mean().item() >= attention_ratio_threshold:
                            weight = original_weight
        
        all_thresholds_none = all(
            x is None for x in [
                entropy_threshold, entropy_ratio_threshold,
                attention_threshold, attention_ratio_threshold,
                random_intervene_ratio, intervene_num
            ]
        )
        if all_thresholds_none:
            weight = original_weight

        if weight > 0.1:
            intervene_list.append(1)
            rm_guided_tokens_mask.append(1)

            eos_token_id = self.rm_tokenizer.eos_token_id
            batch_size = flat_rm_trme.shape[0]
            eos_tokens = torch.full((batch_size, 1), eos_token_id, dtype=flat_rm_trme.dtype, device=flat_rm_trme.device)
            if rm_trajectory_len:
                # 先用LLM生成一部分内容（只对 rm_topk 个 token 进行）
                flat_llm_trme_rm = to_llm_eval[:, :actual_rm_topk, :].view(out_logits.shape[0] * actual_rm_topk, -1)
                flat_llm_ext = self.LLM.generate(
                    input_ids=flat_llm_trme_rm,
                    max_new_tokens=rm_trajectory_len,
                    pad_token_id=self.llm_tokenizer.eos_token_id,
                    do_sample=True,
                )
                
                # 提取非 llm_prompt_input_ids 对应的部分（LLM新生成的部分）
                llm_prompt_len = llm_prompt_input_ids.shape[1]
                llm_generated_tokens = flat_llm_ext[:, llm_prompt_len:]
                
                # 将非prompt部分的tokens解码成文本（使用batch_decode更高效）
                output = self.llm_tokenizer.batch_decode(llm_generated_tokens, skip_special_tokens=True)
                
                # 将文本转换成RM的token ids
                rm_generated_tokens = self.rm_tokenizer(output, return_tensors='pt', padding=True)['input_ids'].to(self.rm_dev)
                
                # 扩展 rm_prompt_input_ids 以匹配batch size
                batch_size = flat_llm_trme_rm.shape[0]
                expanded_rm_prompt = torch.unsqueeze(rm_prompt_input_ids, 1).repeat(1, actual_rm_topk, 1).view(out_logits.shape[0] * actual_rm_topk, -1)
                
                # 将RM生成的tokens拼接到 rm_prompt_input_ids 上
                flat_rm_trme_ext = torch.cat([expanded_rm_prompt, rm_generated_tokens], dim=1)
                
                if not isinstance(self.RM, ValueModel):
                    # 添加EOS token
                    flat_rm_trme_ext = torch.cat([flat_rm_trme_ext, eos_tokens], dim=1)
                
                rm_out = self.RM(input_ids=flat_rm_trme_ext)
            else:
                if isinstance(self.RM, ValueModel):
                    rm_out = self.RM(input_ids=flat_rm_trme.to(self.rm_dev))
                else:
                    flat_rm_trme_with_eos = torch.cat([flat_rm_trme, eos_tokens], dim=1)
                    rm_out = self.RM(input_ids=flat_rm_trme_with_eos.to(self.rm_dev))

            rewards_rm = rm_out.logits.flatten().to(self.llm_dev)
            del rm_out
            # 将 rewards 扩展到 pre_screen_beam_width 维度，前 actual_rm_topk 个有值，其余为负无穷
            # 这样在采样时只会从前 actual_rm_topk 个 token 中选择
            rewards = torch.full_like(orig_scores, float('-inf')).to(self.llm_dev)
            rewards[:out_logits.shape[0] * actual_rm_topk] = rewards_rm
        else:
            if weight == 0:
                intervene_list.append(0)
                rm_guided_tokens_mask.append(0)
                rewards = torch.zeros_like(prescreen_logits.flatten()).to(self.llm_dev)
            else:
                intervene_list.append(1)
                rm_guided_tokens_mask.append(1)
                rewards = torch.zeros_like(prescreen_logits.flatten()).to(self.llm_dev)
        
        # 原有的逻辑：weight是浮点数
        if weight >= 10:
            combined_scores = rewards
        elif weight <= 0.1:
            # 即使 weight <= 0.1，如果 rm_topk < pre_screen_beam_width，也只从前 rm_topk 个 token 中选择
            if actual_rm_topk < topk:
                combined_scores = torch.full_like(orig_scores, float('-inf')).to(self.llm_dev)
                combined_scores[:out_logits.shape[0] * actual_rm_topk] = orig_scores[:out_logits.shape[0] * actual_rm_topk]
            else:
                combined_scores = orig_scores
        else:
            combined_scores = rewards * weight + orig_scores

        combined_scores_probs = F.softmax(combined_scores, dim=-1).float()
        combined_scores_categorical = dist.Categorical(probs=combined_scores_probs)
        combined_scores_entropy = combined_scores_categorical.entropy().item()
        tokens_entropy["combined"].append(combined_scores_entropy)
        
        assert llm_input_ids.shape[0] == 1
        orig_scores = orig_scores / temperature
        orig_scores_probs = F.softmax(orig_scores, dim=-1)
        orig_top_k_ids = torch.multinomial(orig_scores_probs, num_samples=1)
        
        combined_scores_probs_temp = F.softmax(combined_scores / temperature, dim=-1)
        top_k_ids = torch.multinomial(combined_scores_probs_temp, num_samples=1)
            
        tokens_shift.append((flat_llm_trme[orig_top_k_ids][:, -1].item(), flat_llm_trme[top_k_ids][:, -1].item()))
        
        token_entropy_ratio = np.log(entropy_ / previous_entropy) if previous_entropy is not None else 0
        if flat_llm_trme[orig_top_k_ids][:, -1].item() == flat_llm_trme[top_k_ids][:, -1].item():
            tokens_entropy["no_shift"].append(entropy_)
            tokens_entropy_ratio["no_shift"].append(token_entropy_ratio)
        else:
            tokens_entropy["shift"].append(entropy_)
            tokens_entropy_ratio["shift"].append(token_entropy_ratio)
        tokens_entropy["all"].append(entropy_)
        tokens_entropy_ratio["all"].append(token_entropy_ratio)
        previous_entropy = entropy_
        generated_tokens_ids.append(flat_llm_trme[top_k_ids][:, -1].item())
        
        generate_step_results = {
            "llm_tokens": flat_llm_trme[top_k_ids.to(self.llm_dev)],
            "rm_tokens": flat_rm_trme[top_k_ids.to(self.rm_dev)],
            "tokens_shift": tokens_shift,
            "tokens_entropy": tokens_entropy,
            "tokens_entropy_ratio": tokens_entropy_ratio,
            "previous_entropy": previous_entropy,
            "generated_tokens_ids": generated_tokens_ids,
            "rm_guided_tokens_mask": rm_guided_tokens_mask,
        }
        return generate_step_results
    
    def generate(
        self,
        prompt,
        weight=0.,
        topk=1,
        max_new_token=128,
        temperature=0.7,
        intervene_num=5,
        entropy_threshold=None,
        entropy_ratio_threshold=None,
        attention_threshold=None,
        attention_ratio_threshold=None,
        random_intervene_ratio=None,
        rm_trajectory_len=None,
    ):
        llm_tokens = self.get_llm_input_ids(prompt)
        llm_prompt_tokens = llm_tokens
        llm_prompt = self.llm_tokenizer.batch_decode(llm_prompt_tokens, skip_special_tokens=False)[0]
        llm_prompt_no_special_tokens = self.llm_tokenizer.batch_decode(llm_prompt_tokens, skip_special_tokens=True)[0]
        rm_tokens = self.get_rm_input_ids(prompt)
        rm_prompt_tokens = rm_tokens
        rm_prompt = self.rm_tokenizer.batch_decode(rm_prompt_tokens, skip_special_tokens=False)[0]
        if llm_tokens.shape[-1] > 1024:
            print("The sequence of tokens is too long!!! Returning none!")
            return None
        
        if rm_tokens.shape[-1] > 1024:
            print("The sequence of tokens is too long!!! Returning none!")
            return None
          
        intervene_list = []
            
        iterator_obj = range(max_new_token)
        iterator_obj = tqdm(iterator_obj)

        generate_results = {
            "llm_tokens": llm_tokens,
            "llm_prompt_tokens": llm_prompt_tokens,
            "rm_tokens": rm_tokens,
            "rm_prompt_tokens": rm_prompt_tokens,
            "tokens_shift": [],
            "tokens_entropy": {"shift": [], "no_shift": [], "all": [], "combined": []},
            "tokens_entropy_ratio": {"shift": [], "no_shift": [], "all": []},
            "previous_entropy": None,
            "intervene_ratio": None,
            "llm_prompt": llm_prompt,
            "rm_prompt": rm_prompt,
            "llm_prompt_no_special_tokens": llm_prompt_no_special_tokens,
            "llm_prompt_tokens_ids": llm_prompt_tokens[0].tolist(),
            "rm_prompt_tokens_ids": rm_prompt_tokens[0].tolist(),
            "generated_tokens_ids": [],
            "rm_guided_tokens_mask": [],
        }
        
        for token_idx in iterator_obj:
            with torch.no_grad():
                mout = self.LLM(input_ids=generate_results["llm_tokens"], output_attentions=True, output_hidden_states=False)
                generate_step_results = self.generate_step(
                    mout,
                    generate_results,
                    topk,
                    weight,
                    temperature,
                    token_idx,
                    intervene_num,
                    entropy_threshold,
                    entropy_ratio_threshold,
                    rm_trajectory_len,
                    layer_idxs=None,
                    window_size=10,
                    attention_threshold=attention_threshold,
                    attention_ratio_threshold=attention_ratio_threshold,
                    intervene_list=intervene_list,
                    random_intervene_ratio=random_intervene_ratio,
                )
                for key, value in generate_step_results.items():
                    generate_results[key] = value
                del mout

            if generate_results["llm_tokens"][0][-1] == self.llm_tokenizer.eos_token_id:
                break

        intervene_ratio = sum(intervene_list) / len(intervene_list)
        generate_results["intervene_ratio"] = intervene_ratio

        return generate_results
