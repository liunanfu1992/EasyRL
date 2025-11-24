from torch.utils.tensorboard import SummaryWriter
import os

class IndicatorMonitor:
    def __init__(self, log_dir: str, project_name: str, experiment_name: str, flush_secs: int = 10):
        
        self.save_path = os.path.join(log_dir, project_name, experiment_name)
        os.makedirs(self.save_path, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.save_path, flush_secs=flush_secs)
    
    def log_losses(self, loss_info, step):
        self.writer.add_scalar("train/loss/policy_loss", loss_info['policy_loss'], step)
        self.writer.add_scalar("train/loss/kl_loss", loss_info['kl_loss'], step)
        self.writer.add_scalar("train/loss/total_loss", loss_info['total_loss'], step)

    def log_validation_results(self, result_info, step):
        for group_name in result_info.keys():
            for key in result_info[group_name].keys():
                self.writer.add_scalar(f"val/{group_name}/{key}", result_info[group_name][key], step)
    
    def log_training_reward(self, result_info, step):
        for i in result_info.keys():
            for j in result_info[i].keys():
                self.writer.add_scalar(f"train/{i}/{j}", result_info[i][j], step)

    def close(self):
        self.writer.close()
        
    
    
