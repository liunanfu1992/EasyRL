import logging
import torch
import torch.distributed as dist
import torch.nn.functional as F
import contextlib
import os
import gc
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision, StateDictType
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp.api import FullStateDictConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from functools import partial


logger = logging.getLogger(__name__)

class FSDPBackend:
    def __init__(self, model_path: str, learning_rate: float = 1e-5, mixed_precision: bool = True, sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD):
        self.model_path = model_path
        self.learning_rate = learning_rate
        self.mixed_precision = mixed_precision
        self.sharding_strategy = sharding_strategy

        self._init_distributed()
        self.device = torch.device(f"cuda:{self.local_rank}")

        self.tokenizer = None
        self.model = None
        self.optimizer = None

        self.mixed_precision_policy = None
        if mixed_precision:
            self.mixed_precision_policy = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                buffer_dtype=torch.float32,
            )
        
        self._init_model()

    def _init_distributed(self):
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.world_size = dist.get_world_size()
        else:
            self.rank = int(os.environ.get("RANK", 0))
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.world_size = int(os.environ.get("WORLD_SIZE", 1))
            backend = "nccl" if torch.cuda.is_available() else "gloo"

            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("MASTER_PORT", "29500")
    
            dist.init_process_group(backend=backend, init_method="env://", rank=self.rank, world_size=self.world_size)

        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
        
    def _init_model(self):
        if self.rank == 0:
            logger.info(f"Loading model from {self.model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map=None,  
        )

        transformer_layer_cls_names = getattr(self.model, "_no_split_modules", None) 
        transformer_cls_to_wrap = set()

        if transformer_layer_cls_names:  
            name_to_class = {}
            for m in self.model.modules():
                name_to_class[m.__class__.__name__] = m.__class__

            for name in transformer_layer_cls_names:
                if name in name_to_class:
                    transformer_cls_to_wrap.add(name_to_class[name])
                else:
                    logger.warning(f"_no_split_modules item not found as class on scan: {name}")


        if len(transformer_cls_to_wrap) == 0:
            raise ValueError("No transformer layer classes to wrap")

        auto_wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=tuple(transformer_cls_to_wrap)
        )
        
        self.model = FSDP(
            self.model,
            sharding_strategy=self.sharding_strategy,
            mixed_precision=self.mixed_precision_policy,
            auto_wrap_policy=auto_wrap_policy,
            device_id=self.local_rank,
            sync_module_states=True,
            use_orig_params=True
        )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
        )
        if self.rank == 0:
            logger.info(f"Model and optimizer initialized successfully on rank {self.rank}")

    def sleep_backend(self):
        if dist.is_initialized():
            dist.barrier()

        self._aggressive_empty_cache(force_sync=True)

        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, save_policy):
            state_dict = self.model.state_dict()

        if self.model is not None:
            for p in self.model.parameters():
                p.grad = None

            for module in self.model.modules():
                for name, buf in module.named_buffers(recurse=False):
                    if buf.is_cuda:
                        module._buffers[name] = buf.to("cpu", non_blocking=True)

        self._offload_model_to_cpu()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        if self.optimizer is not None:
            try:
                self._saved_optim_state = FSDP.optim_state_dict(self.model, self.optimizer)
            except Exception as e:
                logger.warning(f"Failed to collect optimizer state dict for CPU offload: {e}")
                self._saved_optim_state = None
            self.optimizer = None

        self._aggressive_empty_cache(force_sync=True)

        if dist.is_initialized():
            dist.barrier()

        return state_dict

    def wake_up_backend(self):
        if dist.is_initialized():
            dist.barrier()

        device = torch.device(f"cuda:{self.local_rank}")
        self._load_model_to_gpu()

        if self.model is not None:
            for module in self.model.modules():
                for name, buf in module.named_buffers(recurse=False):
                    if not buf.is_cuda:
                        module._buffers[name] = buf.to(device, non_blocking=True)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
        )

        if hasattr(self, "_saved_optim_state") and self._saved_optim_state is not None:
            try:
                sharded_osd = FSDP.shard_full_optim_state_dict(self._saved_optim_state, self.model)
                self.optimizer.load_state_dict(sharded_osd)
                device = torch.device(f"cuda:{self.local_rank}")
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v) and not v.is_cuda:
                            state[k] = v.to(device, non_blocking=True)
               
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except Exception as e:
                logger.warning(f"Failed to restore optimizer state from saved CPU snapshot: {e}")
            finally:
                self._saved_optim_state = None

        if dist.is_initialized():
            dist.barrier()
    
    def _offload_model_to_cpu(self):
        if self.model is None:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        for handle in self.model._all_handles:  
            if handle._offload_params:  
                continue  
            flat_param = handle.flat_param  
            handle.flat_param_to(torch.device("cpu"), non_blocking=True)  

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()  
        gc.collect()
    
    def _aggressive_empty_cache(self, force_sync: bool = True, max_retries: int = 3):  
        if not torch.cuda.is_available():  
            return  
        
        for attempt in range(max_retries):  
            before_reserved = torch.cuda.memory_reserved()  
            gc.collect()  
            torch.cuda.empty_cache()  
            
            if force_sync and torch.cuda.is_available():
                torch.cuda.synchronize()  
            
            after_reserved = torch.cuda.memory_reserved()  
            reserved_freed = before_reserved - after_reserved  

            if reserved_freed < 1024**3:  
                break
            
    def _load_model_to_gpu(self):  
        if self.model is None:  
            return  
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        device = torch.device(f"cuda:{self.local_rank}")  
        
        for handle in self.model._all_handles:  
            if handle._offload_params:  
                continue  
            flat_param = handle.flat_param  
            handle.flat_param_to(device, non_blocking=True)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
    def _prepare_data_for_forward(self, data_list):
        prompt_with_response = []
        for row in data_list:
            prompt_text = row['templated_prompt']
            for response in row['infer_content']:
                prompt_with_response.append((prompt_text, response))

        pad_id = self.tokenizer.pad_token_id or 0
        eos_token = self.tokenizer.eos_token

        input_ids_list, labels_list, prompt_lens = [], [], []
        max_len = 0

        for prompt_text, response in prompt_with_response:
            full_text = prompt_text + response
            if eos_token and not full_text.endswith(eos_token):
                full_text = full_text + eos_token
            
            prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False).input_ids
            full_ids = self.tokenizer(full_text, add_special_tokens=False).input_ids
            prompt_ids_len = len(prompt_ids)
            label = [-100] * prompt_ids_len + full_ids[prompt_ids_len:]

            input_ids_list.append(full_ids)
            labels_list.append(label)
            prompt_lens.append(prompt_ids_len)

            max_len = max(max_len, len(full_ids))
        
        for i in range(len(input_ids_list)):
            pad_len = max_len - len(input_ids_list[i])
            input_ids_list[i] = input_ids_list[i] + [pad_id] * pad_len
            labels_list[i] = labels_list[i] + [-100] * pad_len
        
        device = self.device
        input_ids = torch.tensor(input_ids_list, device=device)
        attention_mask = (input_ids != pad_id).to(input_ids.dtype)
        labels = torch.tensor(labels_list, device=device)
        return input_ids, attention_mask, labels


    def forward_compute_logprobs(self, data_list):
        input_ids, attention_mask, labels = self._prepare_data_for_forward(data_list)

        use_amp = self.mixed_precision and torch.cuda.is_available()
        ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16) if use_amp else contextlib.nullcontext()

        with ctx:
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=None)
            logits = outputs.logits

        logits = logits.float()
        logits_shift = logits[:, :-1, :]
        ids_shift = input_ids[:, 1:]
        labels_shift = labels[:, 1:]
        attn_shift = attention_mask[:, 1:]

        logprobs = F.log_softmax(logits_shift, dim=-1)
        token_logprobs = logprobs.gather(-1, ids_shift.unsqueeze(-1)).squeeze(-1)
        gen_mask = (labels_shift != -100) & (attn_shift > 0)
        token_logprobs = token_logprobs * gen_mask

        return token_logprobs, gen_mask
    
    def update_model(self):
        max_norm = 1.0
        FSDP.clip_grad_norm_(self.model, max_norm)
        self.optimizer.step()
        self.optimizer.zero_grad()



