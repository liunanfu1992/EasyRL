import torch



class GRPOLossCalculator:
    def __init__(self, low_clip_coffe: float = 0.2, high_clip_coffe: float = 0.2, kl_loss_coeff: float = 0.0):
        self.kl_loss_coeff = kl_loss_coeff
        self.low_clip_coffe = low_clip_coffe
        self.high_clip_coffe = high_clip_coffe

    def calculate_policy_loss(self, old_log_probs, new_log_probs, token_level_advantages, gen_mask):

        importance_ratio = torch.exp(new_log_probs - old_log_probs)
        unclipped_policy = -token_level_advantages * importance_ratio
        clipped_policy = -token_level_advantages * torch.clamp(importance_ratio, 1 - self.low_clip_coffe, 1 + self.high_clip_coffe)
        
        token_policy_loss = torch.maximum(unclipped_policy, clipped_policy) 
        mask = gen_mask.to(token_policy_loss.dtype)
        token_policy_loss = token_policy_loss * mask
        

        denom = mask.sum().clamp_min(1.0)
        policy_loss = token_policy_loss.sum() / denom

        return policy_loss
    
    def calculate_kl_loss(self, old_log_probs, new_log_probs, gen_mask):

        kl = old_log_probs - new_log_probs
        kl = torch.clamp(kl, min=-20, max=20)

        ratio = torch.exp(kl)
        token_level_kld = (ratio - kl - 1).contiguous()  
        token_level_kld = torch.clamp(token_level_kld, min=-10, max=10)

        mask = gen_mask.to(token_level_kld.dtype)
        token_level_kld = token_level_kld * mask

        denom = mask.sum().clamp_min(1.0)
        kld = token_level_kld.sum() / denom
        kld = kld * self.kl_loss_coeff  
        return kld