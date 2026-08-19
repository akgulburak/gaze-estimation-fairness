# file: your_project/my_ext/my_iter_runner.py
import torch
from mmcv.runner import IterBasedRunner, IterLoader, RUNNERS
from mmcv.parallel import DataContainer as DC

from typing import List, Optional, Tuple
from torch.utils.data import DataLoader

import warnings
import mmcv
import time

import numpy as np
from torch.utils.data import BatchSampler
from collections import Counter

class WeightedBatchSamplerWrapper(BatchSampler):
    def __init__(self, base_sampler, weights):
        self.base_sampler = base_sampler
        self.weights = torch.as_tensor(weights, dtype=torch.float)

    def __iter__(self):
        for batch_indices in self.base_sampler:
            # Return both the data indices and the corresponding weights
            yield batch_indices

    def __len__(self):
        return len(self.base_sampler)

class WeightedDataloader(mmcv.runner.iter_based_runner.IterLoader):
    def calculate_weights(self, data_loader):
        ethnicities_labels = data_loader._dataloader.dataset.ethnicities_dict["ethnicities"]
        counts = Counter(ethnicities_labels.values())
        number_of_blacks = counts["black"]
        number_of_asian = counts["asian"]
        number_of_caucasian = counts["caucasian"]
        number_of_other = counts["other"]
        number_of_total = number_of_blacks + number_of_asian + number_of_caucasian + number_of_other

        weight_black = number_of_total / (number_of_blacks)
        weight_asian = number_of_total / (number_of_asian)
        weight_caucasian = number_of_total / (number_of_caucasian)
        weight_other = weight_caucasian

        weights = []
        for ethnicity_label in ethnicities_labels.values():
            if ethnicity_label=="black":
                weights.append(weight_black)
            elif ethnicity_label=="asian":
                weights.append(weight_asian)
            elif ethnicity_label=="caucasian":
                weights.append(weight_caucasian)
            elif ethnicity_label=="other":
                weights.append(weight_other)
            else:
                print("Error in labels!")
                exit()
        return weights

    def __init__(self, data_loader):
        ethnicity_weights = self.calculate_weights(data_loader)
        #random_sampler = torch.utils.data.WeightedRandomSampler(ethnicity_weights, len(ethnicity_weights), replacement=True)
        base_batch_sampler = BatchSampler(sampler=data_loader._dataloader.sampler, batch_size=data_loader._dataloader.batch_sampler.batch_size, drop_last=False)
        batch_sampler = WeightedBatchSamplerWrapper(base_batch_sampler, ethnicity_weights)
        loader_kwargs = {
            k: v for k, v in vars(data_loader._dataloader).items()
            if not k.startswith("_") and k not in ["dataset", "sampler", "batch_sampler", "shuffle", "batch_size", "_DataLoader__multiprocessing_context",
                                                "_dataset_kind", "_DataLoader__initialized", "_IterableDataset_len_called", "_iterator", "pin_memory"]
        }
        new_data_loader = torch.utils.data.DataLoader(data_loader._dataloader.dataset, batch_sampler = batch_sampler, shuffle=False, pin_memory=False, **loader_kwargs)
        #new_data_loader = IterLoader(new_data_loader)
        self._dataloader = new_data_loader
        self.iter_loader = IterLoader(new_data_loader)

@RUNNERS.register_module()
class ClassWeightBasedRunner(IterBasedRunner):
    def train(self, data_loader, **kwargs):
        # if False:
        #     random_sampler = torch.utils.data.WeightedRandomSampler(ethnicity_weights, len(ethnicity_weights), replacement=True)
        #     new_data_loader = torch.utils.data.DataLoader(data_loader._dataloader.dataset, batch_sampler = batch_sampler, shuffle=False, pin_memory=False, **loader_kwargs)
        #     batch_sampler = BatchSampler(random_sampler, batch_size=data_loader._dataloader.batch_sampler.batch_size, drop_last=False)
        #     loader_kwargs = {
        #         k: v for k, v in vars(data_loader._dataloader).items()
        #         if not k.startswith("_") and k not in ["dataset", "sampler", "batch_sampler", "shuffle", "batch_size", "_DataLoader__multiprocessing_context",
        #                                             "_dataset_kind", "_DataLoader__initialized", "_IterableDataset_len_called", "_iterator", "pin_memory"]
        #     }
        #     new_data_loader = IterLoader(new_data_loader)
        #     self.data_loader = new_data_loader

        #     self.ethnicity_weight = ethnicity_weights
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._epoch = data_loader.epoch
        data_batch = next(data_loader)
        self.data_batch = data_batch
        self.call_hook('before_train_iter')
        outputs = self.model.train_step(data_batch, self.optimizer, **kwargs)
        if not isinstance(outputs, dict):
            raise TypeError('model.train_step() must return a dict')
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
        self.outputs = outputs
        self.call_hook('after_train_iter')
        del self.data_batch
        self._inner_iter += 1
        self._iter += 1
