from easyrl.data_processor.training_data_processor import TrainingDataProcessor
from easyrl.data_processor.validation_data_processor import ValidationDataProcessor
from easyrl.inference_engine.inference_engine import InferenceEngine
from easyrl.reward_verifier.math_verifier import MathVerifier
from easyrl.advantage_calculator.group_advantage_calculator import GroupAdvantageCalculator
from easyrl.fsdp_backend.fsdp_backend import FSDPBackend
from easyrl.loss_calculator.grpo_loss_calculator import GRPOLossCalculator
from transformers import AutoTokenizer
import torch
import logging
import os
import shutil

logger = logging.getLogger(__name__)

class GRPOTrainer:
    def __init__(self, batch_size: int = 64, epochs: int = 1, rollout_n: int = 8, off_policy_num: int = 1, kl_loss_coeff: float = 0.0, num_gpus: int = 8, forward_size: int = 64, backward_size: int = 16, model_path: str = '/run/determined/workdir/jiayanglyu/model_zoo/Qwen3-8B', checkpoint_save_dir: str = '/run/determined/workdir/jiayanglyu/EasyRL/checkpoint', save_every_n_steps: int = 10):
        self.batch_size = batch_size
        self.epochs = epochs
        self.rollout_n = rollout_n
        self.off_policy_num = off_policy_num
        self.kl_loss_coeff = kl_loss_coeff
        self.num_gpus = num_gpus
        self.forward_size = forward_size
        self.backward_size = backward_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model_path = model_path
        self.checkpoint_save_dir = checkpoint_save_dir
        self.save_every_n_steps = save_every_n_steps
        
        if checkpoint_save_dir:
            os.makedirs(checkpoint_save_dir, exist_ok=True)
        
        self.training_data_processor = TrainingDataProcessor(
            data_path='/run/determined/workdir/jiayanglyu/EasyRL/data/rl_train_with_sys_prompt.parquet',
            batch_size=self.batch_size,
            epochs=self.epochs
        )
        self.validation_data_processor = ValidationDataProcessor(
            data_path='/run/determined/workdir/jiayanglyu/EasyRL/data/rl_valid_with_sys_prompt.parquet'
        )

        self.inference_engine = None
        self.fsdp_backend = None
        self.math_verifier = MathVerifier()
        self.group_advantage_calculator = GroupAdvantageCalculator()
        
        

    def fit(self):
        curr_index = 0
        fsdp_checkpoint_path = None  
        vllm_checkpoint_path = None 
        
        while True:
            
            if self.inference_engine is None:
                self.inference_engine = InferenceEngine(
                    model_path=self.model_path,
                    rollout_n=self.rollout_n
                )
            
            self.inference_engine.wake_up_engine(vllm_checkpoint_path)

            if curr_index % 10 == 0:
                validation_group = self.validation_data_processor.get_validation_group()
                validation_group = self.submit_validation_task_to_inference_engine(validation_group)
                validation_group = self.verify_validation_answer(validation_group)
            

            all_batch = []
            for _ in range(self.off_policy_num):
                all_batch.append(self.training_data_processor.get_next_batch())
            
            for curr_batch in all_batch:
                curr_batch = self.submit_training_task_to_inference_engine(curr_batch)
                curr_batch = self.verify_training_answer(curr_batch)
                curr_batch = self.compute_advantage(curr_batch)

    
            self.inference_engine.sleep_engine()
            
            all_batch = self.tokenize_and_pad_all_responses(all_batch)
            
            self.fsdp_backend = FSDPBackend(
                model_path=self.model_path,
                num_processes=self.num_gpus,
                learning_rate=1e-6,
                cpu_offload=True,
                checkpoint_path=fsdp_checkpoint_path  
            )

    
            old_log_probs_list = []
            for curr_batch in all_batch:
                all_input_ids = []
                all_labels = []
                for row in curr_batch:
                    for i in range(len(row['input_ids'])):
                        all_input_ids.append(row['input_ids'][i])
                        all_labels.append(row['labels'][i])
                
                total_responses = len(all_input_ids)
                all_old_log_probs = []
                
                for start_idx in range(0, total_responses, self.forward_size):
                    end_idx = start_idx + self.forward_size
                    batch_input_ids = all_input_ids[start_idx:end_idx]
                    batch_labels = all_labels[start_idx:end_idx]
                    
                    old_log_probs_batch, _ = self.fsdp_backend.forward_compute_logprobs(
                        batch_input_ids,
                        batch_labels,
                        requires_grad=False  
                    )
                    all_old_log_probs.append(old_log_probs_batch.cpu())
                
                old_log_probs_list.append(all_old_log_probs)
            
            for batch_idx, curr_batch in enumerate(all_batch):
                self.fsdp_backend.set_train_mode()
                self.fsdp_backend.zero_grad()
                
                all_input_ids = []
                all_labels = []
                all_advantages = []
                for row in curr_batch:
                    for i in range(len(row['input_ids'])):
                        all_input_ids.append(row['input_ids'][i])
                        all_labels.append(row['labels'][i])
                        all_advantages.append(row['advantage'][i])
                
                total_responses = len(all_input_ids)
                num_steps = total_responses // self.backward_size
                
                for step_idx in range(num_steps):
                    start_idx = step_idx * self.backward_size
                    end_idx = start_idx + self.backward_size
                    
                    batch_input_ids = all_input_ids[start_idx:end_idx]
                    batch_labels = all_labels[start_idx:end_idx]
                    batch_advantages = all_advantages[start_idx:end_idx]
                    
                    _ = self.fsdp_backend.forward_compute_logprobs(
                        batch_input_ids,
                        batch_labels,
                        requires_grad=True  
                    )

                    advantages = torch.tensor(batch_advantages, dtype=torch.float32)
                    
                    forward_batch_idx = start_idx // self.forward_size
                    offset_in_forward_batch = start_idx % self.forward_size
                    
                    old_log_probs_batch = old_log_probs_list[batch_idx][forward_batch_idx][offset_in_forward_batch:offset_in_forward_batch + (end_idx - start_idx)]
                    
                    is_last = (step_idx == num_steps - 1)
                    
                    loss_info = self.fsdp_backend.backward_step(
                        loss_fn={'type': 'grpo', 'kl_coeff': self.kl_loss_coeff, 'low_clip_coeff': 0.2, 'high_clip_coeff': 0.2},
                        old_log_probs=old_log_probs_batch,
                        advantages=advantages,
                        is_last_micro_batch=is_last,
                        num_accumulation_steps=num_steps,
                        traj_batch_size= self.batch_size * self.rollout_n
                    )
                    
                    logger.info(f"Epoch {curr_index} - Batch {batch_idx} - Step {step_idx}/{num_steps} - "
                              f"Responses [{start_idx}:{end_idx}] - "
                              f"Policy Loss: {loss_info['policy_loss']:.4f}, "
                              f"KL Loss: {loss_info['kl_loss']:.4f}, "
                              f"Total Loss: {loss_info['total_loss']:.4f}")
                
                self.fsdp_backend.update_model()

            checkpoint_result = self.fsdp_backend.sleep_backend()
            self.fsdp_backend = None  

            vllm_checkpoint_path, fsdp_checkpoint_path = self.ckpt_temporary_storage(checkpoint_result)
            
            # 定期保存checkpoint
            if self.checkpoint_save_dir and (curr_index + 1) % self.save_every_n_steps == 0:
                self._save_checkpoint(curr_index + 1, vllm_checkpoint_path)
            
            curr_index += 1

    def ckpt_temporary_storage(self, checkpoint_result):
        if checkpoint_result and checkpoint_result[0] and checkpoint_result[1]:
            new_vllm_checkpoint_path = checkpoint_result[0]  
            new_fsdp_checkpoint_path = checkpoint_result[1]  
            
            return new_vllm_checkpoint_path, new_fsdp_checkpoint_path
        else: 
            raise ValueError("Checkpoint result is None")

    def submit_training_task_to_inference_engine(self, training_batch):
        prompt_list = []
        for row in training_batch:
            prompt_list.append(row['prompt'])
        infer_results, processed_prompts = self.inference_engine.infer_task(prompt_list, 'train')

        for i, row in enumerate(training_batch):
            row['infer_content'] = infer_results[i]
            row['templated_prompt'] = processed_prompts[i]
        return training_batch

    def submit_validation_task_to_inference_engine(self, validation_group):
        prompt_list = []
        for group in validation_group:
            for row in validation_group[group]['content']:
                prompt_list.append(row['prompt'])
        
        infer_results, _ = self.inference_engine.infer_task(prompt_list, 'val')

        idx = 0
        for group in validation_group:
            for row in validation_group[group]['content']:
                row['infer_content'] = infer_results[idx]
                idx += 1
        return validation_group
    
    def verify_validation_answer(self, validation_group):
        judge_list = []
        for group in validation_group:
            for row in validation_group[group]['content']:
                judge_list.append({'content': row['infer_content'][0], 'ground_truth': row['ground_truth']})
        
        judge_result = self.math_verifier.verify_answer(judge_list)
        
        idx = 0
        for group in validation_group:
            for row in validation_group[group]['content']:
                row['reward'] = judge_result[idx]
                idx += 1

        return validation_group

    def verify_training_answer(self, training_batch):
        judge_list = []
        for row in training_batch:
            ground_truth = row['ground_truth']
            for content in row['infer_content']:
                judge_list.append({'content': content, 'ground_truth': ground_truth})

        rewards = self.math_verifier.verify_answer(judge_list)

        idx = 0
        for row in training_batch:
            row['reward'] = rewards[idx:idx+self.rollout_n]
            idx += self.rollout_n
        return training_batch
    
    def compute_advantage(self, training_batch):
        original_reward_batch = []
        for row in training_batch:
            original_reward_batch.append(row['reward'])
        
        advantages = self.group_advantage_calculator.compute_group_advantage(original_reward_batch)

        for i, row in enumerate(training_batch):
            row['advantage'] = advantages[i]
        
        return training_batch
    
    def tokenize_and_pad_all_responses(self, all_batch):
        
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        eos_token = self.tokenizer.eos_token
        
        max_length = 0
        for curr_batch in all_batch:
            for row in curr_batch:
                prompt_text = row['templated_prompt']
                for response in row['infer_content']:
                    full_text = prompt_text + response
                    if eos_token and not full_text.endswith(eos_token):
                        full_text = full_text + eos_token
                    full_ids = self.tokenizer(full_text, add_special_tokens=False).input_ids
                    max_length = max(max_length, len(full_ids))
        
        for curr_batch in all_batch:
            for row in curr_batch:
                prompt_text = row['templated_prompt']
                prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False).input_ids
                prompt_len = len(prompt_ids)
                
                row['input_ids'] = []
                row['labels'] = []
                
                for response in row['infer_content']:
                    full_text = prompt_text + response
                    if eos_token and not full_text.endswith(eos_token):
                        full_text = full_text + eos_token
                    
                    full_ids = self.tokenizer(full_text, add_special_tokens=False).input_ids
                    labels = [-100] * prompt_len + full_ids[prompt_len:]
                    
                    pad_len = max_length - len(full_ids)
                    if pad_len > 0:
                        full_ids = full_ids + [pad_id] * pad_len
                        labels = labels + [-100] * pad_len
                    
                    row['input_ids'].append(full_ids)
                    row['labels'].append(labels)
        
        return all_batch
    
    def _save_checkpoint(self, step, vllm_checkpoint_path):
        checkpoint_name = f"global_step_{step}"
        save_path = os.path.join(self.checkpoint_save_dir, checkpoint_name)
        
        
        if os.path.exists(vllm_checkpoint_path):
            if os.path.exists(save_path):
                shutil.rmtree(save_path)
            shutil.copytree(vllm_checkpoint_path, save_path)
        else:
            logger.error(f"[Step {step}] vLLM checkpoint path does not exist: {vllm_checkpoint_path}")


if __name__ == "__main__":
    grpo_trainer = GRPOTrainer()
    grpo_trainer.fit()
