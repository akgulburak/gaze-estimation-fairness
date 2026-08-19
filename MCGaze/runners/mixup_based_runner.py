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
class MixupBasedRunner(IterBasedRunner):
    def mixup_data(self, x, y, alpha=1.0, use_cuda=True, clip=0.05):
        '''Returns mixed inputs, pairs of targets, and lambda'''
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        if use_cuda:
            index = torch.randperm(batch_size).cuda()
        else:
            index = torch.randperm(batch_size)
        if lam>=clip and lam<=1-clip:
            if lam>=0.5:
                lam = 1-clip
            else:
                lam = clip
        mixed_x = lam * x + (1 - lam) * x[index, :]

        permute_index = torch.tensor(list(range(batch_size)))
        permute_y = y
        if lam < 0.5:
            permute_index = index
            permute_y = [y[i] for i in index]
        return mixed_x, permute_y

    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._epoch = data_loader.epoch
        data_batch = next(data_loader)
        self.data_batch = data_batch
        self.call_hook('before_train_iter')
        # print(data_batch["img"])
        # print(data_batch.keys())
        # print(len(data_batch["gt_gazes"].data[0]))
        data_batch["img"].data[0], data_batch["gt_gazes"].data[0] = self.mixup_data(data_batch["img"].data[0], data_batch["gt_gazes"].data[0], alpha=1.0, use_cuda=False, clip=0.05)

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

    # def run(self,
    #     data_loaders: List[DataLoader],
    #     workflow: List[Tuple[str, int]],
    #     max_iters: Optional[int] = None,
    #     **kwargs) -> None:

    #     assert isinstance(data_loaders, list)
    #     assert mmcv.is_list_of(workflow, tuple)
    #     assert len(data_loaders) == len(workflow)
    #     if max_iters is not None:
    #         warnings.warn(
    #             'setting max_iters in run is deprecated, '
    #             'please set max_iters in runner_config', DeprecationWarning)
    #         self._max_iters = max_iters
    #     assert self._max_iters is not None, (
    #         'max_iters must be specified during instantiation')

    #     self.logger.info('Hooks will be executed in the following order:\n%s',
    #                         self.get_hook_info())
    #     self.logger.info('workflow: %s, max: %d iters', workflow,
    #                         self._max_iters)
    #     self.call_hook('before_run')

    #     iter_loaders = [IterLoader(x) for x in data_loaders]

    #     self.call_hook('before_epoch')

    #     while self.iter < self._max_iters:
    #         for i, flow in enumerate(workflow):
    #             self._inner_iter = 0
    #             mode, iters = flow
    #             if not isinstance(mode, str) or not hasattr(self, mode):
    #                 raise ValueError(
    #                     'runner has no method named "{}" to run a workflow'.
    #                     format(mode))
    #             iter_runner = getattr(self, mode),
    #             for _ in range(iters):
    #                 if mode == 'train' and self.iter >= self._max_iters:
    #                     break
    #                 iter_runner(iter_loaders[i], **kwargs)

    #     time.sleep(1)  # wait for some hooks like loggers to finish
    #     self.call_hook('after_epoch')
    #     self.call_hook('after_run')
