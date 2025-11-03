from easyrl.data_processor.training_data_processor import TrainingDataProcessor
from easyrl.data_processor.validation_data_processor import ValidationDataProcessor
from easyrl.inference_engine.inference_engine import InferenceEngine
from time import sleep

class GRPOTrainer:
    def __init__(self, batch_size: int = 512, micro_batch_size: int = 64, epochs: int = 10):
        self.batch_size = batch_size
        self.micro_batch_size = micro_batch_size
        self.epochs = epochs
        self.training_data_processor = TrainingDataProcessor(data_path='/run/determined/workdir/jiayanglyu/EasyRL/data/rl_train.parquet', batch_size=self.batch_size, micro_batch_size=self.micro_batch_size, epochs=self.epochs)
        self.validation_data_processor = ValidationDataProcessor(data_path='/run/determined/workdir/jiayanglyu/EasyRL/data/rl_valid.parquet')
        self.inference_engine = InferenceEngine(model_path='/run/determined/workdir/jiayanglyu/model_zoo/Qwen2.5-7B')
    
    def fit(self):
        curr_index = 0
        while True:
            self.inference_engine.wake_up_engine()
            
            if curr_index % 10 == 0:
                validation_group = self.validation_data_processor.get_validation_group()
                validation_group = self.submit_validation_task_to_inference_engine(validation_group)

            current_batch = self.training_data_processor.get_next_batch()
            current_batch = self.submit_training_task_to_inference_engine(current_batch)

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
    


if __name__ == "__main__":
    grpo_trainer = GRPOTrainer()
    grpo_trainer.fit()