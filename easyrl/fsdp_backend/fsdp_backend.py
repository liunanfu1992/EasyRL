import logging
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import contextlib
import os
import gc
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision, CPUOffload, StateDictType, FullStateDictConfig
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import AutoModelForCausalLM, AutoTokenizer
from easyrl.loss_calculator.grpo_loss_calculator import GRPOLossCalculator
from functools import partial
from transformers import AutoConfig
from safetensors.torch import save_file


logger = logging.getLogger(__name__)


_WORKER_MODEL = None
_WORKER_TOKENIZER = None
_WORKER_OPTIMIZER = None
_WORKER_RANK = None
_WORKER_DEVICE = None


class FSDPBackend:
    def __init__(self, model_path: str, learning_rate: float, mixed_precision: bool, num_processes: int,
                 cpu_offload: bool, exchange_path: str, checkpoint_path: str,
                 sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD):

        self.model_path = model_path
        self.learning_rate = learning_rate
        self.mixed_precision = mixed_precision
        self.sharding_strategy = sharding_strategy
        self.num_processes = num_processes
        self.cpu_offload = cpu_offload
        self.exchange_path = exchange_path
        self.checkpoint_path = checkpoint_path  
        
        os.makedirs(exchange_path, exist_ok=True)
        
        self.mixed_precision_policy = None
        if mixed_precision:
            self.mixed_precision_policy = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                buffer_dtype=torch.float32,
            )
        
        self.request_queue = None
        self.response_queue = None
        self.worker_processes = None
        
        self._start_workers()

    def _start_workers(self):
        ctx = mp.get_context('spawn')
        self.request_queue = ctx.Queue()
        self.response_queue = ctx.Queue()
        
        self.worker_processes = []
        for rank in range(self.num_processes):
            p = ctx.Process(
                target=_worker_main,
                args=(rank, self.num_processes, self.model_path, self.learning_rate,
                      self.mixed_precision_policy, self.sharding_strategy, self.cpu_offload,
                      self.request_queue, self.response_queue, self.checkpoint_path)
            )
            p.start()
            self.worker_processes.append(p)
        
        for _ in range(self.num_processes):
            status = self.response_queue.get()
            if status != 'ready':
                raise RuntimeError(f"Worker initialization failed: {status}")
        

    def forward_compute_logprobs(self, input_ids_list, labels_list, requires_grad=True):
        self.request_queue.put(('forward', {'input_ids': input_ids_list, 'labels': labels_list, 'requires_grad': requires_grad}))
        result = self.response_queue.get()
        
        if isinstance(result, Exception):
            raise result
        
        return result

    def backward_step(self, loss_fn, old_log_probs, advantages, is_last_micro_batch, traj_batch_size=512):
        self.request_queue.put(('backward', {
            'loss_fn': loss_fn,
            'old_log_probs': old_log_probs,
            'advantages': advantages,
            'is_last': is_last_micro_batch,
            'traj_batch_size': traj_batch_size
        }))
        
        result = self.response_queue.get()
        if isinstance(result, Exception):
            raise result
        
        return result

    def update_model(self):
        self.request_queue.put(('update', None))
        
        for _ in range(self.num_processes):
            status = self.response_queue.get()
            if status != 'updated':
                raise RuntimeError(f"Model update failed: {status}")

    def set_train_mode(self):
        self.request_queue.put(('set_train', None))
        for _ in range(self.num_processes):
            self.response_queue.get()

    def zero_grad(self):
        self.request_queue.put(('zero_grad', None))
        for _ in range(self.num_processes):
            self.response_queue.get()

    def sleep_backend(self):
        checkpoint_path = os.path.join(
            self.exchange_path,
            "fsdp_checkpoint_latest.pt"  
        )
        
        self.request_queue.put(('save_and_shutdown', checkpoint_path))
        
        hf_checkpoint_dir = None
        fsdp_checkpoint_path = None
        
        for _ in range(self.num_processes):
            status = self.response_queue.get()
            if isinstance(status, tuple) and status[0] == 'saved':
                hf_checkpoint_dir = status[1]
                fsdp_checkpoint_path = status[2]
            elif status != 'shutdown_ready':
                raise RuntimeError(f"Save and shutdown failed: {status}")
        
        logger.info("All workers saved state, terminating processes...")
        
        for p in self.worker_processes:
            p.join(timeout=30)
            if p.is_alive():
                logger.warning(f"Worker process {p.pid} did not terminate, forcing...")
                p.terminate()
                p.join(timeout=5)
        
        if hf_checkpoint_dir and fsdp_checkpoint_path:
            logger.info(f"FSDP backend destroyed")
            logger.info(f"  - HuggingFace format (for vLLM): {hf_checkpoint_dir}")
            logger.info(f"  - Full checkpoint (for FSDP): {fsdp_checkpoint_path}")
            return (hf_checkpoint_dir, fsdp_checkpoint_path)
        else:
            logger.error("Failed to create checkpoint")
            return (None, None)

    def shutdown(self):
        logger.info("Shutting down FSDP backend...")
        self.request_queue.put(('shutdown', None))
        
        for p in self.worker_processes:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
        
        logger.info("FSDP backend shut down")

    @property  
    def device(self):
        return torch.device('cuda:0')


def _worker_main(rank, world_size, model_path, learning_rate, mixed_precision_policy, 
                 sharding_strategy, cpu_offload, request_queue, response_queue, checkpoint_path=None):
    global _WORKER_MODEL, _WORKER_TOKENIZER, _WORKER_OPTIMIZER, _WORKER_RANK, _WORKER_DEVICE
    
    try:
        os.environ['RANK'] = str(rank)
        os.environ['LOCAL_RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29600'
        
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://", rank=rank, world_size=world_size)
        
        if torch.cuda.is_available():
            torch.cuda.set_device(rank)
        
        _WORKER_DEVICE = torch.device(f"cuda:{rank}")
        _WORKER_RANK = rank
        
        _init_model(model_path, mixed_precision_policy, sharding_strategy, cpu_offload, rank)
        _WORKER_OPTIMIZER = torch.optim.AdamW(_WORKER_MODEL.parameters(), lr=learning_rate)
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            _load_checkpoint(checkpoint_path)
        
        if rank == 0:
            response_queue.put('ready')
        
        dist.barrier()
        if rank == 0:
            for _ in range(1, world_size):
                response_queue.put('ready')
        
        last_forward_result = None  
        
        while True:
            if rank == 0:
                task, data = request_queue.get()
                task_data = [task, data]
            else:
                task_data = [None, None]
            
            dist.broadcast_object_list(task_data, src=0)
            task, data = task_data
            
            if task == 'shutdown':
                dist.barrier()
                dist.destroy_process_group()
                break
            
            elif task == 'forward':
                all_input_ids = data['input_ids']
                all_labels = data['labels']
                requires_grad = data.get('requires_grad', True)
                
                per_rank_size = len(all_input_ids) // world_size
                start_idx = rank * per_rank_size
                end_idx = start_idx + per_rank_size
                
                my_input_ids = all_input_ids[start_idx:end_idx]
                my_labels = all_labels[start_idx:end_idx]
                
                if requires_grad:
                    my_result = _forward_compute_logprobs(my_input_ids, my_labels)
                    last_forward_result = my_result  
                    
                    if rank == 0:
                        response_queue.put((None, None))
                else:
                    with torch.no_grad():
                        my_result = _forward_compute_logprobs(my_input_ids, my_labels)
                    
                    all_results = [None] * world_size
                    dist.all_gather_object(all_results, my_result)
                    
                    if rank == 0:
                        token_logprobs_list = [r['token_logprobs'].cpu() for r in all_results]
                        gen_mask_list = [r['gen_mask'].cpu() for r in all_results]
                        
                        combined_logprobs = torch.cat(token_logprobs_list, dim=0)
                        combined_mask = torch.cat(gen_mask_list, dim=0)
                        
                        response_queue.put((combined_logprobs, combined_mask))
            
            elif task == 'backward':
                if last_forward_result is None:
                    raise RuntimeError("Must call forward before backward")
                
                all_old_log_probs = data['old_log_probs']
                all_advantages = data['advantages']
                
                per_rank_size = len(all_old_log_probs) // world_size
                start_idx = rank * per_rank_size
                end_idx = start_idx + per_rank_size
                
                my_old_log_probs = all_old_log_probs[start_idx:end_idx]
                my_advantages = all_advantages[start_idx:end_idx]
                
                loss_info = _backward_step(
                    last_forward_result,  
                    data['loss_fn'],
                    my_old_log_probs,
                    my_advantages,
                    data['is_last'],
                    data.get('traj_batch_size')
                )
                
                if rank == 0:
                    response_queue.put(loss_info)
                
                last_forward_result = None
            
            elif task == 'update':
                max_norm = 1.0
                FSDP.clip_grad_norm_(_WORKER_MODEL, max_norm)
                _WORKER_OPTIMIZER.step()
                _WORKER_OPTIMIZER.zero_grad()
                if rank == 0:
                    for _ in range(world_size):
                        response_queue.put('updated')
            
            elif task == 'set_train':
                _WORKER_MODEL.train()
                if rank == 0:
                    for _ in range(world_size):
                        response_queue.put('ok')
            
            elif task == 'zero_grad':
                _WORKER_OPTIMIZER.zero_grad(set_to_none=True)
                if rank == 0:
                    for _ in range(world_size):
                        response_queue.put('ok')
            
            elif task == 'save_and_shutdown':
                checkpoint_path = data
                
                with FSDP.state_dict_type(_WORKER_MODEL, 
                                         StateDictType.FULL_STATE_DICT,
                                         FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
                    full_model_state_dict = _WORKER_MODEL.state_dict()
                    full_optimizer_state_dict = FSDP.optim_state_dict(_WORKER_MODEL, _WORKER_OPTIMIZER)
                
                if rank == 0:
                    cleaned_model_state_dict = {}
                    for k, v in full_model_state_dict.items():
                        new_k = k.replace('_fsdp_wrapped_module.', '')
                        cleaned_model_state_dict[new_k] = v.cpu()
                    
                    import shutil
                    hf_checkpoint_dir = checkpoint_path.replace('.pt', '_hf') 
                    
                    if os.path.exists(hf_checkpoint_dir):
                        shutil.rmtree(hf_checkpoint_dir)
                    os.makedirs(hf_checkpoint_dir, exist_ok=True)
                    
                    safetensors_path = os.path.join(hf_checkpoint_dir, 'model.safetensors')
                    save_file(cleaned_model_state_dict, safetensors_path)
                    
                    try:
                        config = AutoConfig.from_pretrained(_WORKER_TOKENIZER.name_or_path, trust_remote_code=True)
                        config.save_pretrained(hf_checkpoint_dir)
                        _WORKER_TOKENIZER.save_pretrained(hf_checkpoint_dir)
                    except Exception as e:
                        logger.warning(f"Failed to save config/tokenizer: {e}")
                    
                    optimizer_path = checkpoint_path 
                    checkpoint = {
                        'model_state_dict': cleaned_model_state_dict,
                        'optimizer_state_dict': full_optimizer_state_dict,
                    }
                    torch.save(checkpoint, optimizer_path)
                    
                    response_queue.put(('saved', hf_checkpoint_dir, optimizer_path))
                    del full_model_state_dict, full_optimizer_state_dict, cleaned_model_state_dict, checkpoint
                else:
                    response_queue.put('shutdown_ready')
                
                dist.barrier()
                
                dist.destroy_process_group()
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                
                break
        
    except Exception as e:
        logger.error(f"Worker {rank} error: {e}")
        import traceback
        traceback.print_exc()
        if rank == 0:
            response_queue.put(e)
        raise


def _load_checkpoint(checkpoint_path):
    global _WORKER_MODEL, _WORKER_OPTIMIZER   
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'model_state_dict' in checkpoint:
        model_state_dict = checkpoint['model_state_dict']
        
        wrapped_state_dict = {}
        for k, v in model_state_dict.items():
            if not k.startswith('_fsdp_wrapped_module.'):
                wrapped_state_dict[f'_fsdp_wrapped_module.{k}'] = v
            else:
                wrapped_state_dict[k] = v
        
        _WORKER_MODEL.load_state_dict(wrapped_state_dict, strict=False)
    
    if 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict']:
        full_optimizer_state_dict = checkpoint['optimizer_state_dict']
        sharded_optimizer_state_dict = FSDP.shard_full_optim_state_dict(
            full_optimizer_state_dict,
            _WORKER_MODEL
        )
        _WORKER_OPTIMIZER.load_state_dict(sharded_optimizer_state_dict)
    
    del checkpoint
    

def _init_model(model_path, mixed_precision_policy, sharding_strategy, cpu_offload, rank):
    global _WORKER_MODEL, _WORKER_TOKENIZER
    
    _WORKER_TOKENIZER = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if _WORKER_TOKENIZER.pad_token is None and _WORKER_TOKENIZER.eos_token is not None:
        _WORKER_TOKENIZER.pad_token = _WORKER_TOKENIZER.eos_token
    
    if rank == 0:
        logger.info(f"Loading model from {model_path}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=None,
    )
    
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    
    transformer_layer_cls_names = getattr(model, "_no_split_modules", None)
    transformer_cls_to_wrap = set()
    
    if transformer_layer_cls_names:
        name_to_class = {}
        for m in model.modules():
            name_to_class[m.__class__.__name__] = m.__class__
        
        for name in transformer_layer_cls_names:
            if name in name_to_class:
                transformer_cls_to_wrap.add(name_to_class[name])
    
    if len(transformer_cls_to_wrap) == 0:
        raise ValueError("No transformer layer classes to wrap")
    
    auto_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=tuple(transformer_cls_to_wrap)
    )
    
    cpu_offload_policy = None
    if cpu_offload:
        cpu_offload_policy = CPUOffload(offload_params=True)
    
    _WORKER_MODEL = FSDP(
        model,
        sharding_strategy=sharding_strategy,
        mixed_precision=mixed_precision_policy,
        auto_wrap_policy=auto_wrap_policy,
        device_id=rank,
        sync_module_states=True,
        use_orig_params=True,
        limit_all_gathers=True,  
        cpu_offload=cpu_offload_policy 
    )


def _forward_compute_logprobs(input_ids_list, labels_list):
    global _WORKER_MODEL, _WORKER_TOKENIZER, _WORKER_DEVICE
    
    device = _WORKER_DEVICE
    pad_id = _WORKER_TOKENIZER.pad_token_id or 0
    
    input_ids = torch.tensor(input_ids_list, device=device)
    labels = torch.tensor(labels_list, device=device)
    attention_mask = (input_ids != pad_id).to(input_ids.dtype)
    
    use_amp = _WORKER_MODEL.mixed_precision is not None and torch.cuda.is_available()
    ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16) if use_amp else contextlib.nullcontext()
    
    with ctx:
        outputs = _WORKER_MODEL(input_ids=input_ids, attention_mask=attention_mask, labels=None)
        logits = outputs.logits

    logits_shift = logits[:, :-1, :]  
    ids_shift = input_ids[:, 1:]      
    labels_shift = labels[:, 1:]
    attn_shift = attention_mask[:, 1:]
    
    target_logits = logits_shift.gather(-1, ids_shift.unsqueeze(-1)).squeeze(-1)  
    logsumexp = torch.stack([
        torch.logsumexp(logit, dim=-1) 
        for logit in logits_shift
    ]) 
    
    del logits, logits_shift, outputs
    
    token_logprobs = target_logits - logsumexp 
    token_logprobs = token_logprobs.float()
    
    del target_logits, logsumexp
    
    gen_mask = (labels_shift != -100) & (attn_shift > 0)
    token_logprobs = token_logprobs * gen_mask
    
    return {
        'token_logprobs': token_logprobs,
        'gen_mask': gen_mask
    }


def _backward_step(forward_result, loss_fn_config, old_log_probs, advantages, is_last, traj_batch_size=512):
    global _WORKER_MODEL
    
    new_log_probs = forward_result['token_logprobs']
    gen_mask = forward_result['gen_mask']
    
    device = new_log_probs.device
    old_log_probs = old_log_probs.to(device)
    advantages = advantages.to(device)
    
    token_level_advantages = advantages.unsqueeze(-1).expand_as(new_log_probs) * gen_mask.float()


    kl_coeff = loss_fn_config.get('kl_coeff', 0.0)
    low_clip = loss_fn_config.get('low_clip_coeff', 0.2)
    high_clip = loss_fn_config.get('high_clip_coeff', 0.2)
    
    loss_calculator = GRPOLossCalculator(
        low_clip_coeff=low_clip,
        high_clip_coeff=high_clip,
        kl_loss_coeff=kl_coeff,
        traj_batch_size=traj_batch_size
    )

    policy_loss = loss_calculator.calculate_policy_loss(
        old_log_probs, 
        new_log_probs, 
        token_level_advantages, 
        gen_mask
    )
    
    if kl_coeff > 0:
        kl_loss = loss_calculator.calculate_kl_loss(
            old_log_probs,
            new_log_probs,
            gen_mask
        )
    else:
        kl_loss = torch.tensor(0.0, device=device)
    
    total_loss = policy_loss + kl_loss
    
    if is_last:
        total_loss.backward()
    else:
        with _WORKER_MODEL.no_sync():
            total_loss.backward()
    
    policy_loss_tensor = torch.tensor(policy_loss.item(), device=device)
    kl_loss_tensor = torch.tensor(kl_loss.item(), device=device)
    total_loss_tensor = torch.tensor(total_loss.item(), device=device)

    dist.all_reduce(policy_loss_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(kl_loss_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
    
    return {
        'policy_loss': policy_loss_tensor.item(),
        'kl_loss': kl_loss_tensor.item(),
        'total_loss': total_loss_tensor.item()
    }
