from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.distributed as dist
import numpy as np
import copy
import math
from torch.utils.data import IterableDataset

from longcat_image.utils.dist_utils import get_world_size, get_rank, get_local_rank

class MultiResolutionDistributedSampler(torch.utils.data.Sampler):
    def __init__(self,
                 batch_size: int,
                 dataset: IterableDataset,
                 data_resolution_infos: List,
                 bucket_info: dict,
                 num_replicas: int = None,
                 rank: int = None,
                 seed: int = 888,
                 epoch: int = 0,
                 shuffle: bool = True):

        if not dist.is_available():
            num_replicas = 1
            rank = 0
        else:
            num_replicas = get_world_size()
            rank = get_rank()

        self.len_items = len(dataset)
        bucket_info = {float(b): bucket_info[b] for b in bucket_info.keys()}
        self.aspect_ratios = np.array(sorted(list(bucket_info.keys())))
        self.resolutions = np.array([bucket_info[aspect] for aspect in self.aspect_ratios])

        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = epoch
        self.shuffle = shuffle
        self.seed = seed
        self.cur_rank_index = []
        self.rng = np.random.RandomState(seed+self.epoch)
        self.global_batch_size = batch_size*num_replicas
        self.data_resolution_infos = np.array(data_resolution_infos, dtype=np.float32)
        print(f'num_replicas {num_replicas}, cur rank {rank}!!!')

        self.split_to_buckets()
        self.num_samples = len(dataset)//num_replicas

    def split_to_buckets(self):
        self.buckets = {}
        self._buckets_bak = {}
        data_aspect_ratio = self.data_resolution_infos[:,0]*1.0/self.data_resolution_infos[:, 1]
        bucket_id = np.abs(data_aspect_ratio[:, None] - self.aspect_ratios).argmin(axis=1)
        for i in range(len(self.aspect_ratios)):
            self.buckets[i] = np.where(bucket_id == i)[0]
            self._buckets_bak[i] = np.where(bucket_id == i)[0]
        for k, v in self.buckets.items():
            print(f'bucket {k}, resolutions {self.resolutions[k]}, sampler nums {len(v)}!!!')

    def get_batch_index(self):
        success_flag = False
        while not success_flag:
            bucket_ids = list(self.buckets.keys())
            bucket_probs = [len(self.buckets[bucket_id]) for bucket_id in bucket_ids]
            bucket_probs = np.array(bucket_probs, dtype=np.float32)
            bucket_probs = bucket_probs / bucket_probs.sum()
            bucket_ids = np.array(bucket_ids, dtype=np.int64)
            chosen_id = int(self.rng.choice(bucket_ids, 1, p=bucket_probs)[0])
            if len(self.buckets[chosen_id]) < self.global_batch_size:
                del self.buckets[chosen_id]
                continue
            batch_data = self.buckets[chosen_id][:self.global_batch_size]
            batch_data = (batch_data, self.resolutions[chosen_id])
            self.buckets[chosen_id] = self.buckets[chosen_id][self.global_batch_size:]
            if len(self.buckets[chosen_id]) == 0:
                del self.buckets[chosen_id]
            success_flag = True
            assert bool(self.buckets), 'There is not enough data in the current epoch.'
        return batch_data

    def shuffle_bucker_index(self):
        self.rng = np.random.RandomState(self.seed+self.epoch)
        self.buckets = copy.deepcopy(self._buckets_bak)
        for bucket_id in self.buckets.keys():
            self.rng.shuffle(self.buckets[bucket_id])

    def __iter__(self):
        return self

    def __next__(self):
        try:
            if len(self.cur_rank_index) == 0:
                global_batch_index, target_resolutions = self.get_batch_index()
                self.cur_rank_index = list(map(
                    int, global_batch_index[self.batch_size*self.rank:self.batch_size*(self.rank+1)]))
                self.resolution = list(map(int, target_resolutions))
            data_index = self.cur_rank_index.pop(0)
            return (data_index, self.resolution)

        except Exception as e:
            self.epoch += 1
            self.shuffle_bucker_index()
            raise StopIteration

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class GroupedBatchSampler(torch.utils.data.Sampler):
    def __init__(self, type_to_indices, batch_size, drop_last=True, shuffle=True,
                 num_replicas=None, rank=None, seed=0, balancing_strategy="max"):

        self.type_to_indices = type_to_indices
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        if num_replicas is None:
            if dist.is_available() and dist.is_initialized():
                num_replicas = dist.get_world_size()
                rank = dist.get_rank()
            else:
                num_replicas = 1
                rank = 0
        self.num_replicas = num_replicas
        self.rank = rank
        self.balancing_strategy = balancing_strategy

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # 1. 预处理：先打乱每个任务的内部索引，并收集长度信息
        shuffled_indices_map = {}
        for e_type, indices in self.type_to_indices.items():
            indices_t = torch.tensor(indices)
            if self.shuffle:
                indices_t = indices_t[torch.randperm(len(indices_t), generator=g)]
            shuffled_indices_map[e_type] = indices_t.tolist()

        # 2. 决策：计算目标采样长度
        lengths = [len(idxs) for idxs in shuffled_indices_map.values()]
        if not lengths:
            return iter([])

        target_len = None
        if self.balancing_strategy == "max":
            target_len = max(lengths)
        elif self.balancing_strategy == "min":
            target_len = min(lengths)
        # "natural" 则 target_len 为 None，不做干预

        all_batches = []

        # 3. 生成 Batches：包含平衡逻辑
        for e_type, indices in shuffled_indices_map.items():
            current_len = len(indices)

            # --- 平衡逻辑开始 ---
            if target_len is not None:
                if current_len < target_len:
                    # [Over-sampling] 数据少于目标：重复填充
                    # 例如 [A, B] -> [A, B, A, B]
                    repeat_times = math.ceil(target_len / current_len)
                    indices = (indices * repeat_times)[:target_len]
                elif current_len > target_len:
                    # [Under-sampling] 数据多于目标：截断
                    indices = indices[:target_len]
            # --- 平衡逻辑结束 ---

            for i in range(0, len(indices), self.batch_size):
                batch = indices[i: i + self.batch_size]

                if len(batch) == self.batch_size:
                    all_batches.append(batch)
                elif not self.drop_last:
                    all_batches.append(batch)

        if self.shuffle:
            batch_indices = torch.randperm(len(all_batches), generator=g).tolist()
            all_batches = [all_batches[i] for i in batch_indices]

        num_batches = len(all_batches)
        num_batches_per_replica = num_batches // self.num_replicas
        total_batches_used = num_batches_per_replica * self.num_replicas

        all_batches = all_batches[:total_batches_used]

        my_batches = all_batches[self.rank: total_batches_used: self.num_replicas]

        return iter(my_batches)

    def __len__(self):
        # 计算逻辑需与 __iter__ 中的平衡逻辑一致，否则进度条会不准
        total_batches = 0

        lengths = [len(indices) for indices in self.type_to_indices.values()]
        if not lengths:
            return 0

        target_len = None
        if self.balancing_strategy == "max":
            target_len = max(lengths)
        elif self.balancing_strategy == "min":
            target_len = min(lengths)

        for length in lengths:
            # 如果有目标长度，则用目标长度计算；否则用原始长度
            calc_len = target_len if target_len is not None else length

            if self.drop_last:
                total_batches += calc_len // self.batch_size
            else:
                total_batches += (calc_len + self.batch_size - 1) // self.batch_size

        return total_batches // self.num_replicas

    def set_epoch(self, epoch):
        self.epoch = epoch