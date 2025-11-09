

export TOKENIZERS_PARALLELISM=false


export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False

# 直接用Python启动，不使用torchrun
python -m easyrl.trainer.GRPO_trainer.grpo_trainer

