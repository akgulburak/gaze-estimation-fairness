# file: your_project/my_ext/my_iter_runner.py
import torch
from mmcv.runner import IterBasedRunner, IterLoader, RUNNERS
from mmcv.parallel import DataContainer as DC

from torch.utils.data import BatchSampler

def add_dims_to_dc(dc, n_dims=2):
    """Add `n_dims` singleton dimensions to a DataContainer holding a tensor."""
    if not isinstance(dc, DC):
        raise TypeError("Expected a DataContainer.")
    if dc.cpu_only:
        return dc  # Don't modify CPU-only containers like img_metas

    data = dc.data
    if torch.is_tensor(data):
        for _ in range(n_dims):
            data = data.unsqueeze(0)
        return DC(data, stack=True)
    return dc

def add_singleton_dims(batched, add_dims_for=('img', 'gt_bboxes', 'gt_labels', 'gt_ids')):
    """After MMCV collate, insert [1,1,...] for specific tensor-like fields."""
    for k in add_dims_for:
        if k in batched:
            v = batched[k]
            # Only for tensors; cpu_only fields like img_metas are lists of dicts and should stay so
            if isinstance(v, torch.Tensor):
                batched[k] = v.unsqueeze(1).unsqueeze(1)   # [N,1,1,...]
            # If some fields are lists of tensors, stack then unsqueeze:
            elif isinstance(v, (list, tuple)) and len(v) and isinstance(v[0], torch.Tensor):
                t = torch.stack(v, dim=0)
                batched[k] = t.unsqueeze(1).unsqueeze(1)
            else:
                batched[k] = add_dims_to_dc(batched[k])
    return batched

@RUNNERS.register_module()
class ReweightBasedRunner(IterBasedRunner):
    """Drop-in replacement for ReweightBasedRunner with only train() changed."""
    def train(self, data_loader, **kwargs):
        if "dataloader_flag" not in self.__dict__:
            demographic_group_weights = data_loader._dataloader.dataset.weights
            random_sampler = torch.utils.data.WeightedRandomSampler(demographic_group_weights, len(demographic_group_weights), replacement=True)
            batch_sampler = BatchSampler(random_sampler, batch_size=data_loader._dataloader.batch_sampler.batch_size, drop_last=False)
            loader_kwargs = {
                k: v for k, v in vars(data_loader._dataloader).items()
                if not k.startswith("_") and k not in ["dataset", "sampler", "batch_sampler", "shuffle", "batch_size", "_DataLoader__multiprocessing_context",
                                                    "_dataset_kind", "_DataLoader__initialized", "_IterableDataset_len_called", "_iterator", "pin_memory"]
            }
            new_data_loader = torch.utils.data.DataLoader(data_loader._dataloader.dataset, batch_sampler = batch_sampler, shuffle=False, pin_memory=False, **loader_kwargs)
            new_data_loader = IterLoader(new_data_loader)
            self.data_loader = new_data_loader

        if "dataloader_flag" not in self.__dict__:
            self.dataloader_flag = True

        self.model.train()
        self.mode = 'train'
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
