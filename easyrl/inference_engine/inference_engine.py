from vllm import LLM, SamplingParams
import logging
import gc
import torch
import os

logger = logging.getLogger(__name__)


def aggressive_empty_cache(force_sync: bool = True, max_retries: int = 3):
    if not torch.cuda.is_available():
        return
    
    num_gpus = torch.cuda.device_count()
    total_freed_reserved = 0
    total_freed_allocated = 0
    
    for attempt in range(max_retries):
        attempt_freed_reserved = 0
        attempt_freed_allocated = 0
        
        for gpu_id in range(num_gpus):
            before_reserved = torch.cuda.memory_reserved(gpu_id)
            before_allocated = torch.cuda.memory_allocated(gpu_id)
            
            gc.collect()
            
            if force_sync:
                torch.cuda.synchronize(gpu_id)
            
            with torch.cuda.device(gpu_id):
                torch.cuda.empty_cache()
            
            after_reserved = torch.cuda.memory_reserved(gpu_id)
            after_allocated = torch.cuda.memory_allocated(gpu_id)
            
            freed_reserved = before_reserved - after_reserved
            freed_allocated = before_allocated - after_allocated
            
            attempt_freed_reserved += freed_reserved
            attempt_freed_allocated += freed_allocated
            
            if freed_reserved > 0 or freed_allocated > 0:
                logger.debug(
                    f"GPU {gpu_id} - Freed {freed_reserved / 1024**3:.2f} GB reserved, "
                    f"{freed_allocated / 1024**3:.2f} GB allocated"
                )
        
        total_freed_reserved += attempt_freed_reserved
        total_freed_allocated += attempt_freed_allocated
        

        
        if attempt_freed_reserved < 1024**3:
            logger.info(f"Early stop: minimal memory freed (<1GB), cleanup complete")
            break



def set_expandable_segments(value: bool):
    try:
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_device(i)
            if value:
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
            else:
                os.environ.pop('PYTORCH_CUDA_ALLOC_CONF', None)
            
            if hasattr(torch.cuda.memory, 'set_per_process_memory_fraction'):
                torch.cuda.memory.set_per_process_memory_fraction(1.0, i)
        
    except Exception as e:
        logger.warning(f"Failed to set expandable_segments: {e}")


class InferenceEngine:
    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 8,
        pipeline_parallel_size: int = 1,
        temperature: float = 0.6,
        top_p: float = 0.95,
        max_tokens: int = 4096, 
        gpu_memory_utilization: float = 0.8,
        trust_remote_code: bool = True,
        dtype: str = 'auto',
        max_num_seqs: int = 512,
        rollout_n: int = 8, 
        load_first_from_local: bool = True,
    ):

        os.environ['VLLM_ALLOW_INSECURE_SERIALIZATION'] = '1'
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

    def wake_up_engine(self, checkpoint_path: str = None):
        set_expandable_segments(False)
        
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
                if checkpoint_path is not None and os.path.exists(checkpoint_path):
                    logger.info(f"Updating model path to new checkpoint: {checkpoint_path}")
                    self.llm.llm_engine.model_config.model = checkpoint_path
                
                self.llm.wake_up(tags=['weights'])
                
                if checkpoint_path is not None and os.path.exists(checkpoint_path):
                    logger.info(f"Reloading weights from {checkpoint_path}")
                    self.llm.collective_rpc("reload_weights")
                    logger.info("Weights reloaded successfully")

                self.llm.wake_up(tags=["kv_cache"]) 
    
    def sleep_engine(self):
        if self.llm is not None:
            self.llm.sleep(level=2)  
        
        aggressive_empty_cache(force_sync=True)
        gc.collect()
        
        set_expandable_segments(True)
        

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
        
        del outputs
        torch.cuda.empty_cache()

        if len(results) > 0:
            logger.info(f"Generated {len(results)} samples with {len(results[0])} responses each")
        else:
            logger.warning("No results generated (empty prompt list)")
        return (results, processed_prompts)

        
        







    
