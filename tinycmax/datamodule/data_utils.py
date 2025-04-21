from itertools import islice
import random

from dotmap import DotMap
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import BatchSampler, DataLoader


def batched(iterable, n, drop_last=False):
    """
    https://docs.python.org/3/library/itertools.html#itertools.batched
    """

    iterator = iter(iterable)
    batches = []
    while batch := tuple(islice(iterator, n)):
        if len(batch) == n or not drop_last:
            batches.append(batch)
    return batches


class _RepeatSampler:
    """
    Sampler that repeats forever.
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)  # this shuffles


class InfiniteDataLoader(DataLoader):
    """
    Allows pre-fetching first batch of next epoch.
    Source: https://github.com/ultralytics/yolov5/blob/master/utils/dataloaders.py.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "batch_sampler", _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __iter__(self):
        for _ in range(len(self)):
            yield next(self.iterator)

    def __len__(self):
        return len(self.batch_sampler.sampler)


class ConcatBatchSampler(BatchSampler):
    """
    Batch sampler over a ConcatDataset.
    Shuffles, drops incomplete batches and cuts off longer sequences in a batch.
    """

    def __init__(self, concat_dataset, batch_size, shuffle=False):
        self.concat_dataset = concat_dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sequence_lengths = [len(dataset) for dataset in self.concat_dataset.datasets]
        self.idx_mapping = [
            list(range(cl - l, cl)) for l, cl in zip(self.sequence_lengths, self.concat_dataset.cumulative_sizes)
        ]
        self.reset()

    def reset(self):
        # shuffle
        random.shuffle(self.idx_mapping) if self.shuffle else None
        # batch
        self.batched_idx_mapping = batched(self.idx_mapping, self.batch_size, drop_last=True)
        # get length: zip so shortest
        self.length = sum(min(len(e) for e in batch) for batch in self.batched_idx_mapping)

    def __len__(self):
        return self.length

    def __iter__(self):
        # reset
        self.reset()

        # iterate over batches
        for batch in self.batched_idx_mapping:
            for idxs in zip(*batch):
                yield idxs


def time_first_collate(batch):
    collated_batch = DotMap(_dynamic=False)
    for key in batch[0]:
        if key in ["frames"]:
            collated_batch[key] = torch.stack([sample[key] for sample in batch], dim=1)
        elif key in ["auxs", "targets"]:
            collated_batch[key] = DotMap(_dynamic=False)
            for k in batch[0][key]:
                if k in ["events"]:
                    collated_batch[key][k] = pad_sequence([sample[key][k].transpose(0, 1) for sample in batch])
                    collated_batch[key][k] = collated_batch[key][k].transpose(0, 2).contiguous()
                else:
                    collated_batch[key][k] = torch.stack([sample[key][k] for sample in batch], dim=1)
        elif key in ["K_rect", "inv_K_rect"]:
            collated_batch[key] = torch.stack([sample[key] for sample in batch])  # constant over time
        elif key in ["eofs"]:
            collated_batch[key] = list(zip(*[sample[key] for sample in batch]))
        else:  # recording
            collated_batch[key] = [sample[key] for sample in batch]
    return collated_batch


def only_add_batch_dim(batch):
    for key in batch:
        if key in ["frames"]:
            batch[key] = batch[key].unsqueeze(1)
        elif key in ["auxs", "targets"]:
            for k in batch[key]:
                batch[key][k] = batch[key][k].unsqueeze(1)
        elif key in ["K_rect", "inv_K_rect"]:
            batch[key] = batch[key].unsqueeze(0)
        elif key in ["eofs"]:
            batch[key] = [[sample] for sample in batch[key]]
    return batch
