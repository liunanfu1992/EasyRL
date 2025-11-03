from vllm import LLM, SamplingParams
from typing import List
import logging
import gc
import torch
from typing import Optional, Callable

logger = logging.getLogger(__name__)


class InferenceEngine:
    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 4,
        pipeline_parallel_size: int = 2,
        temperature: float = 0.6,
        top_p: float = 0.95,
        max_tokens: int = 8192, 
        gpu_memory_utilization: float = 0.8,
        trust_remote_code: bool = True,
        dtype: str = 'auto',
        max_num_seqs: int = 512,
        rollout_n: int = 8, 
        load_first_from_local: bool = True,
    ):

        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.gpu_memory_utilization = gpu_memory_utilization
        self.trust_remote_code = trust_remote_code
        self.dtype = dtype
        self.max_num_seqs = max_num_seqs
        self.rollout_n = rollout_n
        self.load_first_from_local = load_first_from_local

        self.llm = None
        self.training_sampling_params = self._initialize_training_sampling_params()
        self.validation_sampling_params = self._initialize_validation_sampling_params()

    def wake_up_engine(self):
        if self.load_first_from_local:
            logger.info(f"Initializing VLLM engine with model: {self.model_path}, tensor_parallel_size: {self.tensor_parallel_size}, pipeline_parallel_size: {self.pipeline_parallel_size}, gpu_memory_utilization: {self.gpu_memory_utilization}, trust_remote_code: {self.trust_remote_code}, dtype: {self.dtype}")

            self.llm = LLM(
                model=self.model_path,
                tensor_parallel_size=self.tensor_parallel_size,
                pipeline_parallel_size=self.pipeline_parallel_size,
                gpu_memory_utilization=self.gpu_memory_utilization,
                trust_remote_code=self.trust_remote_code,
                dtype=self.dtype,
                max_num_seqs=self.max_num_seqs,
                enable_sleep_mode=True
            )

            self.load_first_from_local = False
        else:
            if self.llm is not None:
                self.llm.wake_up(tags=['weights'])

                training_params = self._get_params_from_training_backend()
                self._update_weights(training_params)

                self.llm.wake_up(tags=["kv_cache"]) 
    
    def sleep_engine(self):
        if self.llm is not None:
            self.llm.sleep(level=1)

        torch.cuda.empty_cache()
        gc.collect()

    
    def _update_weights(self, params):  
        model = self.llm.llm_engine.model_executor.driver_worker.worker.model_runner.model  
        model.load_weights(  
            (name, param) for name, param in params.items()  
        )  
        logger.info(f"Loaded {len(params)} parameters from training backend")  

    def _initialize_training_sampling_params(self):
        return SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            n=self.rollout_n
        )

    def _initialize_validation_sampling_params(self):
        return SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            n=1
        )
    
    def _prepare_prompts(self, prompt_list):
        processed_prompts = []
        tokenizer = self.llm.get_tokenizer()

        for prompt in prompt_list:
            text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            processed_prompts.append(text)

        return processed_prompts
    
    def infer_task(self, prompt_list, task_mode: str):
        if self.llm is None:
            raise RuntimeError("InferenceEngine is not awake. Call wake_up_engine() before infer_task().")
        processed_prompts = self._prepare_prompts(prompt_list)

        outputs = None

        if task_mode == 'train':
            outputs = self.llm.generate(processed_prompts, self.training_sampling_params)
        elif task_mode == 'val':
            outputs = self.llm.generate(processed_prompts, self.validation_sampling_params)
        else:
            raise ValueError(f"Invalid task mode: {task_mode}")
        
        results = []

        for output in outputs:
            generated_texts = [o.text for o in output.outputs]
            results.append(generated_texts)

        if len(results) > 0:
            logger.info(f"Generated {len(results)} samples with {len(results[0])} responses each")
        else:
            logger.warning("No results generated (empty prompt list)")
        return results

        
        







    
