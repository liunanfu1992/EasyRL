import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List
import numpy as np

logger = logging.getLogger(__name__)

class GroupAdvantageCalculator:
    def __init__(self, max_workers: int = 128):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def compute_group_advantage(self, original_reward_group: List[List[float]]):
        def _compute_single_group_advantage(group_rewards: List[float]) -> List[float]:
            eps = 1e-8

            group_rewards = np.array(group_rewards)
            rewards_mean = group_rewards.mean()
            rewards_std = group_rewards.std()

            advantages = (group_rewards - rewards_mean) / (rewards_std + eps)
            return advantages.tolist()
        
        advantages = list(self.executor.map(_compute_single_group_advantage, original_reward_group))
        return advantages   
    
    def __del__(self):
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
