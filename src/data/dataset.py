import numpy as np
import logging

import torch
from torch.utils.data import Dataset

from src.data.dataset_collection import  DatasetCollection

logger = logging.getLogger(__name__)

class SyntheticDataset(Dataset):
    """
    Test
    """
    
    def __init__(self,
                 subset_name: str,
                 source_data: str,
                 prev: int = None,
                 projection_horizon: int = None,
                 has_covariate: bool = False,
                 has_counterfactual: bool = False,                 
                ):
        """
        Args:
            subset_name: train / val / test
            source_data
        """
        self.subset_name = subset_name
        self.source_data = source_data

        self.data = {}
        self.processed = False
        self.prev = prev
        self.projection_horizon = projection_horizon
        self.has_covariate = has_covariate
        self.has_counterfactual = has_counterfactual        

    def __getitem__(self, index) -> dict:
        result = {k: v[index] for k, v in self.data.items() if hasattr(v, '__len__') and len(v) == len(self)}
        if hasattr(self, 'encoder_r'):
            if 'original_index' in self.data:
                result.update({'encoder_r': self.encoder_r[int(result['original_index'])]})
            else:
                result.update({'encoder_r': self.encoder_r[index]})
        return result

    def process_data(self, scaling_params):
        """
        Pre-process dataset for one-step-ahead prediction
        Args:
            scaling_params: dict of standard normalization parameters (calculated with train subset)
        """

        if not self.processed:
            logger.info(f'Processing {self.subset_name} dataset before training')

            loaded_data = np.load(self.source_data)

            prev = self.prev
            horizon = self.projection_horizon
            n = loaded_data['treatments'].shape[0]
            t_horizon = prev + horizon

            assert prev >= horizon , "horizon deve ser menor ou igual que prev"
            assert t_horizon <= loaded_data['treatments'].shape[1] , "o horizonte de previsão não pode ultrapassara o tam total"

            self.data['sequence_lengths']    = torch.from_numpy(np.array([prev] * n))
            self.data['active_entries']      = torch.from_numpy(np.ones((n, prev, 1)))
            self.data['prev_treatments']     = torch.from_numpy(loaded_data['treatments'][:, :prev]).unsqueeze(-1)
            self.data['prev_outcomes']       = torch.from_numpy(loaded_data['outcomes'][:, :prev]).unsqueeze(-1)
            self.data['static_features']     = torch.from_numpy(np.zeros((n, 1)))
            self.data['current_treatments']  = torch.from_numpy(loaded_data['treatments'][:, prev:t_horizon]).unsqueeze(-1)
            self.data['outcomes']            = torch.from_numpy(loaded_data['outcomes'][:, prev:t_horizon]).unsqueeze(-1)

            if self.has_covariate:
                self.data['prev_covariates']    = torch.from_numpy(loaded_data['covariates'][:, :prev]).unsqueeze(-1)
                self.data['current_covariates'] = torch.from_numpy(loaded_data['covariates'][:, prev:t_horizon]).unsqueeze(-1)

            if self.has_counterfactual:
                self.data['current_treatments_cf']  = torch.from_numpy(loaded_data['treatments_cf'][:, prev:t_horizon]).unsqueeze(-1)
                self.data['outcomes_cf']            = torch.from_numpy(loaded_data['outcomes_cf'][:, prev:t_horizon]).unsqueeze(-1)

                if self.has_covariate:
                    self.data['current_covariates_cf'] = torch.from_numpy(loaded_data['covariates_cf'][:, prev:t_horizon]).unsqueeze(-1)

            data_shapes = {k: v.shape for k, v in self.data.items()}
            logger.info(f'Shape of processed {self.subset_name} data: {data_shapes}')

            self.processed = True
        else:
            logger.info(f'{self.subset_name} Dataset already processed')

        return self.data

    def __len__(self):
        return self.data['sequence_lengths'].shape[0]

class SyntheticDatasetCollection(DatasetCollection):
    """
    Dataset collection (train_f, val_f, test_f)
    """

    def __init__(self,
                 source_data: dict,
                 prev: int = 1,
                 projection_horizon: int = 1,
                 has_covariate: bool = False,
                 has_counterfactual: bool = False,
                 **kwargs):
        """
        Args:
            source_data
        """
        super(SyntheticDatasetCollection, self).__init__()

        self.has_covariate = has_covariate
        self.has_counterfactual = has_counterfactual

        self.train_f = SyntheticDataset(
            subset_name='train',
            source_data=source_data['train'],
            prev=prev,
            projection_horizon=projection_horizon,
            has_covariate = has_covariate,
            has_counterfactual = has_counterfactual,
        )

        self.val_f = SyntheticDataset(
            subset_name='val',
            source_data=source_data['val'],
            prev=prev,            
            projection_horizon=projection_horizon,            
            has_covariate = has_covariate,
            has_counterfactual = has_counterfactual,            
        )

        self.test_f = SyntheticDataset(
            subset_name='test',
            source_data=source_data['test'],
            prev=prev,            
            projection_horizon=projection_horizon,            
            has_covariate = has_covariate,
            has_counterfactual = has_counterfactual,            
        )
