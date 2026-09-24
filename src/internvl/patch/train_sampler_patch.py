# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from typing import List, Optional

import torch
import transformers
from torch.utils.data import Dataset, Sampler, Subset
from transformers.tokenization_utils_base import BatchEncoding
from transformers.trainer import (LengthGroupedSampler, RandomSampler,
                                  has_length)
from transformers.trainer_pt_utils import logger


# copy from https://github.com/haotian-liu/LLaVA/blob/main/llava/train/llava_trainer.py#L38
def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float('inf')

    return chunks


# copy from https://github.com/haotian-liu/LLaVA/blob/main/llava/train/llava_trainer.py#L88
def get_length_grouped_indices(
    lengths, batch_size, world_size, generator=None, merge=True
):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [
        indices[i : i + megabatch_size].tolist()
        for i in range(0, len(lengths), megabatch_size)
    ]
    megabatches = [
        sorted(megabatch, key=lambda i: lengths[i], reverse=True)
        for megabatch in megabatches
    ]
    megabatches = [
        split_to_even_chunks(megabatch, lengths, world_size)
        for megabatch in megabatches
    ]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


# modified from https://github.com/haotian-liu/LLaVA/blob/main/llava/train/llava_trainer.py#L99
class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        dataset: Optional[Dataset] = None,
        lengths: Optional[List[int]] = None,
        model_input_name: Optional[str] = None,
        generator=None,
    ):
        if dataset is None and lengths is None:
            raise ValueError('One of dataset and lengths must be provided.')

        self.batch_size = batch_size
        if lengths is None:
            model_input_name = (
                model_input_name if model_input_name is not None else 'input_ids'
            )
            if (
                not (
                    isinstance(dataset[0], dict)
                    or isinstance(dataset[0], BatchEncoding)
                )
                or model_input_name not in dataset[0]
            ):
                raise ValueError(
                    'Can only automatically infer lengths for datasets whose items are dictionaries with an '
                    f"'{model_input_name}' key."
                )
            lengths = [len(feature[model_input_name]) for feature in dataset]
        elif isinstance(lengths, torch.Tensor):
            logger.info(
                'If lengths is a torch.Tensor, LengthGroupedSampler will be slow. Converting lengths to List[int]...'
            )
            lengths = lengths.tolist()
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        indices = get_length_grouped_indices(
            self.lengths, self.batch_size, self.world_size, generator=self.generator
        )
        return iter(indices)

def _get_lengths_for_group_by_length(dataset):
    """
    Collect per-sample lengths aligned with the dataset indices seen by DataLoader.

    Supports:
    1. original InternVL concat-style dataset with `.datasets`;
    2. torch.utils.data.Subset wrapping such a dataset;
    3. single dataset with `.length`.

    Important:
    For Subset, returned lengths must be reordered by subset.indices, because
    LengthGroupedSampler will sample local indices 0..len(subset)-1.
    """
    if isinstance(dataset, Subset):
        base_lengths = _get_lengths_for_group_by_length(dataset.dataset)
        return [base_lengths[int(i)] for i in dataset.indices]

    if hasattr(dataset, "length"):
        return list(dataset.length)

    if hasattr(dataset, "datasets"):
        lengths = []
        for sub_dataset in dataset.datasets:
            lengths.extend(_get_lengths_for_group_by_length(sub_dataset))
        return lengths

    raise AttributeError(
        "Cannot infer lengths for group_by_length sampler. "
        "Expected dataset to be a Subset, to have `.length`, "
        "or to have `.datasets`."
    )

# patch trainer
def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
    if self.train_dataset is None or not has_length(self.train_dataset):
        return None
    # Build the sampler.
    if self.args.group_by_length:
        lengths = _get_lengths_for_group_by_length(self.train_dataset)

        if len(lengths) != len(self.train_dataset):
            raise ValueError(
                "Length mismatch for group_by_length sampler: "
                f"len(lengths)={len(lengths)}, "
                f"len(train_dataset)={len(self.train_dataset)}. "
                "This usually means the dataset split/subset indices are not "
                "aligned with the length list."
            )

        model_input_name = (
            self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
        )

        return LengthGroupedSampler(
            self.args.train_batch_size,
            world_size=self.args.world_size * self.args.gradient_accumulation_steps,
            dataset=self.train_dataset,
            lengths=lengths,
            model_input_name=model_input_name,
        )
    else:
        return RandomSampler(self.train_dataset)


def replace_train_sampler():
    transformers.Trainer._get_train_sampler = _get_train_sampler
    # print('Replace train sampler!!')
