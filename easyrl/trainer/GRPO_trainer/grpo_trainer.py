from easyrl.data_processor.training_data_processor import TrainingDataProcessor
from easyrl.data_processor.validation_data_processor import ValidationDataProcessor
from easyrl.inference_engine.inference_engine import InferenceEngine
from easyrl.reward_verifier.math_verifier import MathVerifier
from easyrl.advantage_calculator.group_advantage_calculator import GroupAdvantageCalculator
from easyrl.fsdp_backend.fsdp_backend import FSDPBackend
from easyrl.loss_calculator.grpo_loss_calculator import GRPOLossCalculator
import torch.distributed as dist
import torch
import logging
import pandas as pd

logger = logging.getLogger(__name__)

class GRPOTrainer:
    def __init__(self, batch_size: int = 64, micro_batch_size: int = 8, epochs: int = 10, rollout_n: int = 8, off_policy_num: int = 1, kl_loss_coeff: float = 0.0):
        self.batch_size = batch_size
        self.micro_batch_size = micro_batch_size
        self.epochs = epochs
        self.rollout_n = rollout_n
        self.off_policy_num = off_policy_num
        self.kl_loss_coeff = kl_loss_coeff
        self.training_data_processor = TrainingDataProcessor(data_path='/run/determined/workdir/jiayanglyu/EasyRL/data/rl_train_with_sys_prompt.parquet', batch_size=self.batch_size, micro_batch_size=self.micro_batch_size, epochs=self.epochs)
        self.validation_data_processor = ValidationDataProcessor(data_path='/run/determined/workdir/jiayanglyu/EasyRL/data/rl_valid_with_sys_prompt.parquet')
        self.inference_engine = None
        self.math_verifier = MathVerifier()
        self.group_advantage_calculator = GroupAdvantageCalculator()
        self.fsdp_backend = None
        self.loss_calculator = GRPOLossCalculator(kl_loss_coeff=self.kl_loss_coeff)

    def fit(self):

        if dist.is_available() and not dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend=backend, init_method="env://")

        is_dist = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if is_dist else 0

        curr_index = 0
        while True:
            if rank == 0:
                if self.inference_engine is None:
                    self.inference_engine = InferenceEngine(model_path='/run/determined/workdir/jiayanglyu/model_zoo/Qwen2.5-7B', rollout_n=self.rollout_n)
                self.inference_engine.wake_up_engine()

            
            # if curr_index % 10 == 0:
            #     validation_group = self.validation_data_processor.get_validation_group()
            #     validation_group = self.submit_validation_task_to_inference_engine(validation_group)
            #     validation_group = self.verify_validation_answer(validation_group)
                

            all_batch = []
            for _ in range(self.off_policy_num):
                all_batch.append(self.training_data_processor.get_next_batch())
            
            for curr_batch in all_batch:
                curr_batch = self.submit_training_task_to_inference_engine(curr_batch)
                curr_batch = self.verify_training_answer(curr_batch)
                curr_batch = self.compute_advantage(curr_batch)

            if rank == 0:
                self.inference_engine.sleep_engine()
            if is_dist:
                dist.barrier()  

            if self.fsdp_backend is None:
                self.fsdp_backend = FSDPBackend(model_path='/run/determined/workdir/jiayanglyu/model_zoo/Qwen2.5-7B')
            else:
                self.fsdp_backend.wake_up_backend()

            old_log_probs_list = []
            for curr_batch in all_batch:
                for micro_batch in curr_batch:
                    with torch.no_grad():
                        old_log_probs, _ = self.submit_micro_batch_task_to_fsdp_backend(micro_batch)
                    old_log_probs = old_log_probs.detach() 
                    old_log_probs_list.append(old_log_probs)
            
            idx = 0
            for curr_batch in all_batch:
                self.fsdp_backend.model.train()
                self.fsdp_backend.optimizer.zero_grad(set_to_none=True)
                for j, micro_batch in enumerate(curr_batch):
                    new_log_probs, gen_mask = self.submit_micro_batch_task_to_fsdp_backend(micro_batch)
                    
                    advantages_flat = []
                    for row in micro_batch:
                        advantages_flat.extend(row['advantage'])

                    device = self.fsdp_backend.device
                    seq_level_advantages = torch.tensor(advantages_flat, dtype=torch.float32, device=device)
                    token_level_advantages = seq_level_advantages.unsqueeze(-1).expand_as(new_log_probs) * gen_mask.float()
                        
                    policy_loss = self.loss_calculator.calculate_policy_loss(old_log_probs_list[idx], new_log_probs, token_level_advantages, gen_mask)
                    kl_loss = self.loss_calculator.calculate_kl_loss(old_log_probs_list[idx], new_log_probs, gen_mask)
                    loss = (policy_loss + kl_loss) / len(curr_batch)

                    if j < len(curr_batch) - 1:
                        with self.fsdp_backend.model.no_sync():
                            loss.backward()
                    else:
                        loss.backward()

                    idx += 1
                
                self.fsdp_backend.update_model()
                logger.info(f"Epoch {curr_index} - Step {j} - Policy Loss: {policy_loss.item()}, KL Loss: {kl_loss.item()}, Loss: {loss.item()}")

            
            curr_index += 1
    
    def submit_micro_batch_task_to_fsdp_backend(self, micro_batch):
        data_list = []
        for row in micro_batch:
            data_list.append({'templated_prompt': row['templated_prompt'], 'infer_content': row['infer_content']})
        
        return self.fsdp_backend.forward_compute_logprobs(data_list)


    def submit_training_task_to_inference_engine(self, training_batch):
        prompt_list = []
        for micro_batch in training_batch:
            for row in micro_batch:
                prompt_list.append(row['prompt'])
                
        is_dist = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if is_dist else 0

        if rank == 0:
            infer_results, processed_prompts = self.inference_engine.infer_task(prompt_list, 'train')
        else: 
            infer_results, processed_prompts = None, None
        
        if is_dist:
            obj = [infer_results, processed_prompts]
            dist.broadcast_object_list(obj, src=0)
            infer_results, processed_prompts = obj

        idx = 0
        for micro_batch in training_batch:
            for row in micro_batch:
                row['infer_content'] = infer_results[idx]
                row['templated_prompt'] = processed_prompts[idx]
                idx += 1
        return training_batch
            

    def submit_validation_task_to_inference_engine(self, validation_group):
        prompt_list = []
        for group in validation_group:
            for row in validation_group[group]['content']:
                prompt_list.append(row['prompt'])
        
        is_dist = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if is_dist else 0

        if rank == 0:
            infer_results, _ = self.inference_engine.infer_task(prompt_list, 'val')
        else: 
            infer_results = None
        
        if is_dist:
            obj = [infer_results]
            dist.broadcast_object_list(obj, src=0)
            infer_results = obj[0]

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
        for micro_batch in training_batch:
            for row in micro_batch:
                ground_truth = row['ground_truth']
                for content in row['infer_content']:
                    judge_list.append({'content': content, 'ground_truth': ground_truth})

        rewards = self.math_verifier.verify_answer(judge_list)

        idx = 0
        for micro_batch in training_batch:
            for row in micro_batch:
                row['reward'] = rewards[idx:idx+self.rollout_n]
                idx += self.rollout_n
        return training_batch
    
    def compute_advantage(self, training_batch):
        original_reward_batch = []
        for micro_batch in training_batch:
            for row in micro_batch:
                original_reward_batch.append(row['reward'])
        
        advantages = self.group_advantage_calculator.compute_group_advantage(original_reward_batch)
        
        idx = 0
        for micro_batch in training_batch:
            for row in micro_batch:
                row['advantage'] = advantages[idx]
                idx += 1
        
        return training_batch


if __name__ == "__main__":
    grpo_trainer = GRPOTrainer()
    grpo_trainer.fit()