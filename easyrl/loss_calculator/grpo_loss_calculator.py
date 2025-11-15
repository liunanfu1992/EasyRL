import torch

class GRPOLossCalculator:
    def __init__(self, low_clip_coeff: float = 0.2, high_clip_coeff: float = 0.2, kl_loss_coeff: float = 0.0, traj_batch_size: int = 512):
        self.kl_loss_coeff = kl_loss_coeff
        self.low_clip_coeff = low_clip_coeff
        self.high_clip_coeff = high_clip_coeff
        self.traj_batch_size = traj_batch_size
        
    def calculate_policy_loss(self, old_log_probs, new_log_probs, token_level_advantages, gen_mask):

        importance_ratio = torch.exp(new_log_probs - old_log_probs)
        unclipped_policy = token_level_advantages * importance_ratio
        clipped_policy = token_level_advantages * torch.clamp(importance_ratio, 1 - self.low_clip_coeff, 1 + self.high_clip_coeff)
        
        token_policy_loss = -torch.minimum(unclipped_policy, clipped_policy) 
        mask = gen_mask.to(token_policy_loss.dtype)
        token_policy_loss = token_policy_loss * mask

        policy_loss = (token_policy_loss.sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)).sum() / self.traj_batch_size
        
        return policy_loss
    
    def calculate_kl_loss(self, old_log_probs, new_log_probs, gen_mask):

        kl = old_log_probs - new_log_probs
        kl = torch.clamp(kl, min=-20, max=20)

        ratio = torch.exp(kl)
        token_level_kld = (ratio - kl - 1).contiguous()  
        token_level_kld = torch.clamp(token_level_kld, min=-10, max=10)

        mask = gen_mask.to(token_level_kld.dtype)
        token_level_kld = token_level_kld * mask

        kld = (token_level_kld.mean(dim=1) / self.traj_batch_size).sum()

        kld = kld * self.kl_loss_coeff  
        return kld