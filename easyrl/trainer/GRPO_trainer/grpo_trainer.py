from easyrl.data_processor.training_data_processor import TrainingDataProcessor
from easyrl.data_processor.validation_data_processor import ValidationDataProcessor
from easyrl.inference_engine.inference_engine import InferenceEngine
from easyrl.reward_verifier.math_verifier import MathVerifier
from easyrl.advantage_calculator.group_advantage_calculator import GroupAdvantageCalculator
import pandas as pd

class GRPOTrainer:
    def __init__(self, batch_size: int = 64, micro_batch_size: int = 8, epochs: int = 10, rollout_n: int = 8):
        self.batch_size = batch_size
        self.micro_batch_size = micro_batch_size
        self.epochs = epochs
        self.rollout_n = rollout_n
        self.training_data_processor = TrainingDataProcessor(data_path='/run/determined/workdir/jiayanglyu/EasyRL/data/rl_train.parquet', batch_size=self.batch_size, micro_batch_size=self.micro_batch_size, epochs=self.epochs)
        self.validation_data_processor = ValidationDataProcessor(data_path='/run/determined/workdir/jiayanglyu/EasyRL/data/rl_valid.parquet')
        self.inference_engine = InferenceEngine(model_path='/run/determined/workdir/jiayanglyu/model_zoo/Qwen2.5-7B', rollout_n=self.rollout_n)
        self.math_verifier = MathVerifier()
        self.group_advantage_calculator = GroupAdvantageCalculator()
    
    def fit(self):
        curr_index = 0
        while True:
            self.inference_engine.wake_up_engine()
            
            if curr_index % 10 == 0:
                validation_group = self.validation_data_processor.get_validation_group()
                validation_group = self.submit_validation_task_to_inference_engine(validation_group)
                validation_group = self.verify_validation_answer(validation_group)
                

            current_batch = self.training_data_processor.get_next_batch()
            current_batch = self.submit_training_task_to_inference_engine(current_batch)
            current_batch = self.verify_training_answer(current_batch)
            current_batch = self.compute_advantage(current_batch)


            self.inference_engine.sleep_engine()

            curr_index += 1
 

    def submit_training_task_to_inference_engine(self, training_batch):
        prompt_list = []
        for micro_batch in training_batch:
            for row in micro_batch:
                prompt_list.append(row['prompt'])
        
        infer_results = self.inference_engine.infer_task(prompt_list, 'train')

        idx = 0
        for micro_batch in training_batch:
            for row in micro_batch:
                row['infer_content'] = infer_results[idx]
                idx += 1
        return training_batch
            

    def submit_validation_task_to_inference_engine(self, validation_group):
        prompt_list = []
        for group in validation_group:
            for row in validation_group[group]['content']:
                prompt_list.append(row['prompt'])

        infer_results = self.inference_engine.infer_task(prompt_list, 'val')

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