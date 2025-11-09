import pandas as pd
from typing import List, Dict, Any, Optional, Iterator
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

class TrainingDataProcessor:
    def __init__(self, data_path: str, batch_size: int = 64, new_columns: List[str] = [], epochs: int = 1):
        self.columns: List[str] = ['prompt', 'ground_truth'] + new_columns
        self.data_pool: List[Dict[str, Any]] = self._load_dataset(data_path, epochs)
        self.batch_size: int = batch_size
        self._batch_gen: Optional[Iterator[List[Dict[str, Any]]]] = None

    def _load_dataset(self, data_path: str, epochs: int) -> List[Dict[str, Any]]:
        origin_dataset = pd.read_parquet(data_path)

        missing_columns = set(self.columns) - set(origin_dataset.columns)
        if missing_columns:
            logger.error(f"Missing columns: {missing_columns}, please check the training data_path: {data_path}")
            raise ValueError(f"Missing columns: {missing_columns}, please check the training data_path: {data_path}")

        data_pool = origin_dataset[self.columns].to_dict('records')
        data_pool = data_pool * epochs
        return data_pool

    def _batch_generator(self) -> Iterator[List[Dict[str, Any]]]:
        total_batches = len(self.data_pool) // self.batch_size
        pbar = tqdm(range(0, total_batches * self.batch_size, self.batch_size), desc="Training step: 0/0", total=total_batches)

        for batch_idx, i in enumerate(pbar, start=1):
            pbar.set_description(f"Training step: {batch_idx}/{total_batches}")
            batch = self.data_pool[i:i+self.batch_size]
            yield batch

    def get_next_batch(self) -> List[Dict[str, Any]]:
        if self._batch_gen is None:
            self._batch_gen = self._batch_generator()

        try:
            batch = next(self._batch_gen)
            return batch
        except StopIteration:
            raise StopIteration("Training is completed")

