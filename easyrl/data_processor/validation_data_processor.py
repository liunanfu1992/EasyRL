import pandas as pd
import logging
from typing import List, Dict, Any, Optional, Iterator

logger = logging.getLogger(__name__)

class ValidationDataProcessor:
    def __init__(self, data_path: str, new_columns: List[str] = []):
        self.columns: List[str] = ['prompt', 'ground_truth', 'extra_info'] + new_columns
        self.validation_group = self._create_validation_group(data_path)
    
    def _create_validation_group(self, data_path: str) -> Dict[str, Dict[str, Any]]:
        origin_dataset = pd.read_parquet(data_path)
        missing_columns = set(self.columns) - set(origin_dataset.columns)
        if missing_columns:
            logger.error(f"Missing columns: {missing_columns}, please check the validation data_path: {data_path}")
            raise ValueError(f"Missing columns: {missing_columns}, please check the validation data_path: {data_path}")

        validation_group: Dict[str, Dict[str, Any]] = {}
        for row in origin_dataset.itertuples():
            group = f"{row.extra_info['data_source']}/pass@{row.extra_info['pass@k']}"
            if group not in validation_group:
                validation_group[group] = {}
                validation_group[group]['pass_k_flag'] = False if row.extra_info['pass@k'] == 1 else True
                validation_group[group]['content'] = []
           
            row_information = {}
            for column in self.columns:
                if column == 'extra_info':
                    continue
                row_information[column] = getattr(row, column)
            
            for _ in range(row.extra_info['pass@k']):
                validation_group[group]['content'].append(row_information.copy())
            
        return validation_group
            
    def get_validation_group(self) -> Dict[str, Dict[str, Any]]:
        return self.validation_group
            
            


            
            




