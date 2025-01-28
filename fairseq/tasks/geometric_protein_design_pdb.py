# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
import itertools
import json
import logging
import os
from typing import Optional, List
from argparse import Namespace
from omegaconf import II
import math
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from fairseq import metrics, utils
from fairseq.data import (
    AppendTokenDataset,
    ConcatDataset,
    ProteinDataset,
    ProteinPDBDataset,
    ProteinDatasetInference,
    PrependTokenDataset,
    StripTokenDataset,
    TruncateDataset,
    data_utils,
    encoders,
    indexed_dataset
)
from fairseq.data.indexed_dataset import get_available_dataset_impl
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.tasks import FairseqTask, register_task
from fairseq.models.esm_modules import Alphabet
from fairseq.models.geometric_protein_model import get_edges_batch


device = torch.device("cuda")


logger = logging.getLogger(__name__)


def load_protein_dataset(
    data_path,
    split,
    filter_conf,
    pdb_data_conf,
    src_dict,
    dataset_impl_source,
    dataset_impl_target,
    left_pad_source,
    left_pad_target,
    max_source_positions,
    max_target_positions,
    prepend_bos=False,
    truncate_source=False,
    num_buckets=0,
    shuffle=True,
    pad_to_multiple=1,
):

    dataset = ProteinPDBDataset(
        csv_path=data_path,
        filtering_cfg=filter_conf,
        pdb_data_cfg=pdb_data_conf,
        pad_idx=src_dict.padding_idx,
        split=split,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        align_dataset=None,
        src_dict=src_dict,
        num_buckets=num_buckets,
        shuffle=shuffle,
        pad_to_multiple=pad_to_multiple,
    )

    logger.info(
        "Loading {} {} {} examples".format(
            data_path, split, len(dataset)
        ))
    
    return dataset


@dataclass
class FilteringConfig(FairseqDataclass):
    max_len: int = field(
        default=512,
        metadata={"help": "Maximum sequence length"}
    )
    min_len: int = field(
        default=60,
        metadata={"help": "Minimum sequence length"}
    )
    subset: Optional[str] = field(
        default=None,
        metadata={"help": "Selects a subset of examples (e.g., for debugging)"}
    )
    allowed_oligomer: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Allowed oligomer types"}
    )
    max_helix_percent: float = field(
        default=1.0,
        metadata={"help": "Maximum helix percentage"}
    )
    max_loop_percent: float = field(
        default=0.5,
        metadata={"help": "Maximum loop percentage"}
    )
    min_beta_percent: float = field(
        default=-1.0,
        metadata={"help": "Minimum beta percentage"}
    )
    rog_quantile: float = field(
        default=0.96,
        metadata={"help": "Radius of gyration quantile"}
    )


@dataclass
class PDBDataConfig(FairseqDataclass):
    num_eval_lengths: int = field(
        default=10,
        metadata={"help": "Number of lengthes for validation"}
    )
    samples_per_eval_length: int = field(
        default=4,
        metadata={"help": "Number of validation sequences for each length"}
    )


@dataclass
class GeometricProteinDesignPDBConfig(FairseqDataclass):
    data: Optional[str] = field(
        default=None,
        metadata={
            "help": "colon separated path to data directories list, will be iterated upon during epochs "
            "in round-robin manner; however, valid and test data are always in the first directory "
            "to avoid the need for repeating them in all directories"
        },
    )
    protein_task: str = field(
        default="myoglobin",
        metadata={"help": "protein task name"}
    )
    left_pad_source: bool = field(
        default=False, metadata={"help": "pad the source on the left"}
    )
    left_pad_target: bool = field(
        default=False, metadata={"help": "pad the target on the left"}
    )
    max_source_positions: int = field(
        default=1024, metadata={"help": "max number of tokens in the source sequence"}
    )
    max_target_positions: int = field(
        default=1024, metadata={"help": "max number of tokens in the target sequence"}
    )
    truncate_source: bool = field(
        default=False, metadata={"help": "truncate source to max-source-positions"}
    )
    num_batch_buckets: int = field(
        default=0,
        metadata={
            "help": "if >0, then bucket source and target lengths into "
            "N buckets and pad accordingly; this is useful on TPUs to minimize the number of compilations"
        },
    )
    train_subset: str = II("dataset.train_subset")
    dataset_impl_source: Optional[ChoiceEnum(get_available_dataset_impl())] = field(
        default="raw", metadata={"help": "data format of source data"}
    )
    dataset_impl_target: Optional[ChoiceEnum(get_available_dataset_impl())] = field(
        default="coor", metadata={"help": "data format of target data"}
    )
    required_seq_len_multiple: int = II("dataset.required_seq_len_multiple")

    eval_aa_recovery: bool = field(
        default=False, metadata={
            "help": "evaluate amino acid recovery or not"
        }
    )

    filtering: FilteringConfig = field(
        default_factory=FilteringConfig,
        metadata={"help": "Configurations for filtering"}
    )

    pdb_data_cfg: PDBDataConfig = field(
        default_factory=PDBDataConfig,
        metadata={"help": "Configurations for filtering"}
    )


@register_task("geometric_protein_design_pdb", dataclass=GeometricProteinDesignPDBConfig)
class GeometricProteinDesignTaskPDB(FairseqTask):
    """
    Translate from one (source) language to another (target) language.

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.
    """

    cfg: GeometricProteinDesignPDBConfig

    def __init__(self, cfg: GeometricProteinDesignPDBConfig, src_dict):
        super().__init__(cfg)
        self.src_dict = src_dict
        self.mask_idx = self.src_dict.mask_idx

    @classmethod
    def setup_task(cls, cfg: GeometricProteinDesignPDBConfig, **kwargs):
        """Setup the task (e.g., load dictionaries).
        the dictionary is composed of amino acids

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """

        paths = utils.split_paths(cfg.data)
        assert len(paths) > 0

        # load dictionaries
        alphabet = Alphabet.from_architecture("ESM-1b")
        src_dict = alphabet

        return cls(cfg, src_dict)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        data_path = self.cfg.data

        filter_conf = self.cfg.filtering
        pdb_data_conf = self.cfg.pdb_data_cfg

        self.datasets[split] = load_protein_dataset(
            data_path,
            split,
            filter_conf,
            pdb_data_conf,
            self.src_dict,
            dataset_impl_source=self.cfg.dataset_impl_source,
            dataset_impl_target=self.cfg.dataset_impl_target,
            left_pad_source=self.cfg.left_pad_source,
            left_pad_target=self.cfg.left_pad_target,
            max_source_positions=self.cfg.max_source_positions,
            max_target_positions=self.cfg.max_target_positions,
            truncate_source=self.cfg.truncate_source,
            num_buckets=self.cfg.num_batch_buckets,
            shuffle=(split != "test"),
            pad_to_multiple=self.cfg.required_seq_len_multiple,
            prepend_bos=True,
        )

    def build_model(self, cfg):
        model = super().build_model(cfg)

        return model

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)

        if self.cfg.eval_aa_recovery:
            with torch.no_grad():
                source_input = sample["source_input"]
                src_tokens = source_input["src_tokens"]
                batch_size, n_nodes = src_tokens.size()[0], src_tokens.size()[1]

                target_input = sample["target_input"]
                motif = sample["motif"]
                output_mask = motif["output"]
                pdbs = sample["pdb"]
                ec = sample["ec"]
                centers = sample["center"]

                encoder_out, coords, _ = model(src_tokens,
                                            source_input["src_lengths"],
                                            target_input["target_coor"],
                                            motif, ec["ec1"], ec["ec2"], ec["ec3"], ec["ec4"])

                encoder_out[:, 1: -1, : 4] = -math.inf
                encoder_out[:, :, 24:] = -math.inf
                coords = coords.reshape(batch_size, -1, 3)
                target_coor = sample["target_input"]["target_coor"]
                rmsd = torch.sum(torch.sqrt(torch.sum(torch.square(coords - target_coor), dim=-1)) * motif["input"], dim=-1) / motif["input"].sum()
                # rmsd = torch.sqrt(torch.sum(torch.sum(torch.square(coords - target_coor), dim=-1) * motif["input"], dim=-1))
                aar = ((src_tokens == encoder_out.argmax(-1)) * motif["input"]).sum() / motif["input"].sum()

                coords = (output_mask.unsqueeze(-1) * coords + (output_mask.unsqueeze(-1) == 0).int() * target_coor)[:, 1: -1, :]
                coords = coords + centers.unsqueeze(1)

                indexes = torch.argmax(encoder_out, dim=-1)   # [batch, length]
                indexes = output_mask * indexes + (output_mask == 0).int() * source_input["src_tokens"]
                srcs = [model.encoder.alphabet.string(source_input["src_tokens"][i]) for i in range(source_input["src_tokens"].size(0))]
                strings = [model.encoder.alphabet.string(indexes[i]) for i in range(len(indexes))]
                return loss, sample_size, logging_output, strings, srcs, pdbs, coords, target_coor, rmsd, aar
        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.cfg.max_source_positions, self.cfg.max_target_positions)

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.src_dict

    @property
    def target_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return None

