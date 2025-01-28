# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
import math
from omegaconf import II
import numpy as np
from copy import deepcopy

import torch
import torch.nn.functional as F
from fairseq import metrics, modules, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from fairseq.data.openfold.utils import rigid_utils

@dataclass
class GeometricProteinDesignConfig(FairseqDataclass):
    tpu: bool = II("common.tpu")
    encoder_factor: float = field(
        default=1.0,
        metadata={"help": "the importance of the encoder loss"}
    )
    decoder_factor: float = field(
        default=1.0,
        metadata={"help": "the importance of the decoder loss"}
    )


@dataclass
class GeometricProteinDesignPDBConfig(FairseqDataclass):
    tpu: bool = II("common.tpu")
    aa_type_factor: float = field(
        default=1.0,
        metadata={"help": "the importance of the encoder loss"}
    )
    trans_factor: float = field(
        default=1.0,
        metadata={"help": "the importance of the decoder loss"}
    )
    rot_factor: float = field(
        default=1.0,
        metadata={"help": "the importance of the decoder loss"}
    )


def rotation_matrix_cosine_loss(R_pred, R_true):
    """
    Args:
        R_pred: (*, 3, 3).
        R_true: (*, 3, 3).
    Returns:
        Per-matrix losses, (*, ).
    """
    size = list(R_pred.shape[:-2])
    ncol = R_pred.numel() // 3

    RT_pred = R_pred.transpose(-2, -1).reshape(ncol, 3) # (ncol, 3)
    RT_true = R_true.transpose(-2, -1).reshape(ncol, 3) # (ncol, 3)

    ones = torch.ones([ncol, ], dtype=torch.long, device=R_pred.device)
    loss = F.cosine_embedding_loss(RT_pred, RT_true, ones, reduction='none')  # (ncol*3, )
    loss = loss.reshape(size + [3]).sum(dim=-1)    # (*, )
    return loss


@register_criterion("geometric_protein_loss", dataclass=GeometricProteinDesignConfig)
class GeometricProteinLoss(FairseqCriterion):
    """
    Implementation for the loss used in masked language model (MLM) training.
    """

    def __init__(self, cfg: GeometricProteinDesignConfig, task):
        super().__init__(task)
        self.tpu = cfg.tpu
        self.encoder_factor = cfg.encoder_factor
        self.decoder_factor = cfg.decoder_factor

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        source_input = sample["source_input"]
        target_input = sample["target_input"]
        motif = sample["motif"]
        output_mask = motif["output"]
        sample_size = output_mask.int().sum()
        ec = sample["ec"]

        encoder_out, decoder_out, center = model(source_input["src_tokens"],
                                         source_input["src_lengths"],
                                         target_input["target_coor"],
                                         motif, ec["ec1"], ec["ec2"], ec["ec3"], ec["ec4"])

        # encoder output should be logits
        loss_encoder = -torch.log(encoder_out.gather(dim=-1, index=source_input["src_tokens"].unsqueeze(-1)).squeeze(-1))
        # loss_encoder = torch.mean(torch.sum(loss_encoder * output_mask, dim=-1))
        loss_encoder = torch.mean(torch.sum(loss_encoder * output_mask, dim=-1) / (output_mask.sum(-1) + 1e-8))

        # decoder output should be the directly predicted mse loss
        target_coor = target_input["target_coor"]

        # apply same transformation for ground truth data
        # CoM
        target_coor = target_coor - center
        # scale coordinates
        target_coor = target_coor * model.coordinate_scaling

        # loss_decoder = torch.mean(torch.sum(torch.sum(torch.square(decoder_out - target_coor), dim=-1) * output_mask, dim=-1))
        loss_decoder = torch.mean(torch.sum(torch.sum(torch.square(decoder_out - target_coor), dim=-1) * output_mask, dim=-1) / (output_mask.sum(-1) + 1e-8))
        loss = self.encoder_factor * loss_encoder + self.decoder_factor * loss_decoder
        logging_output = {
            "loss": loss.data,
            "loss_encoder": loss_encoder.data,
            "loss_decoder": loss_decoder.data,
            "ntokens": sample_size,
            "nsentences": output_mask.size()[0]}
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        # batch_size = logging_outputs[0].get("nsentences")
        # loss = sum([log["loss"].cpu() * batch_size for log in logging_outputs])
        # loss_encoder = sum([log.get("loss_encoder").cpu() * batch_size for log in logging_outputs])
        # loss_decoder = sum([log.get("loss_decoder").cpu() * batch_size for log in logging_outputs])
        # sample_size = sum(log.get("ntokens", 0) for log in logging_outputs)

        # metrics.log_scalar(
        #     "loss", loss / sample_size / math.log(2), round=3
        # )
        # metrics.log_scalar(
        #     "sequence loss", loss_encoder / sample_size / math.log(2), sample_size, round=3
        # )
        # metrics.log_scalar(
        #     "coordinate loss", loss_decoder / sample_size, sample_size, round=3
        # )
        # metrics.log_scalar(
        #     "sample size", sample_size)

        loss = sum([log["loss"].cpu() for log in logging_outputs])
        loss_encoder = sum([log.get("loss_encoder").cpu() for log in logging_outputs])
        loss_decoder = sum([log.get("loss_decoder").cpu() for log in logging_outputs])
        sample_size = sum(log.get("ntokens", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss / math.log(2), round=3
        )
        metrics.log_scalar(
            "sequence loss", loss_encoder / math.log(2), round=3
        )
        metrics.log_scalar(
            "coordinate loss", loss_decoder, round=3
        )
        metrics.log_scalar(
            "sample size", sample_size)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True


@register_criterion("geometric_protein_pdb_loss", dataclass=GeometricProteinDesignPDBConfig)
class GeometricProteinPDBLoss(FairseqCriterion):
    """
    Implementation for the loss used in masked language model (MLM) training.
    """

    def __init__(self, cfg: GeometricProteinDesignPDBConfig, task):
        super().__init__(task)
        self.tpu = cfg.tpu
        self.aa_type_factor = cfg.aa_type_factor
        self.trans_factor = cfg.trans_factor
        self.rot_factor = cfg.rot_factor

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        aatype = sample['aatype']
        bb_rigids = rigid_utils.Rigid.from_tensor_7(sample['rigids_0'])
        rots = bb_rigids._rots.get_rot_mats()
        trans = bb_rigids.get_trans()
        mask = sample['input_mask']

        aa_mask = (aatype != model.encoder.alphabet.cls_idx) & (aatype != model.encoder.alphabet.eos_idx) & (aatype != model.encoder.alphabet.padding_idx)
        encoder_out, decoder_out, pred_rot = model(aatype, trans, mask, aa_mask)

        # encoder output should be logits
        loss_encoder = -torch.log(encoder_out.gather(dim=-1, index=aatype.unsqueeze(-1)).squeeze(-1))
        loss_encoder = torch.mean(torch.sum(loss_encoder * mask, dim=-1))

        # decoder output should be the directly predicted mse loss
        loss_decoder = torch.mean(torch.sum(torch.sum(torch.square(decoder_out - trans), dim=-1) * mask, dim=-1))

        # backbone rotation loss
        loss_rot = rotation_matrix_cosine_loss(pred_rot, rots)
        loss_rot = torch.mean(torch.sum(loss_rot * mask, dim=-1))

        loss = self.aa_type_factor * loss_encoder + self.trans_factor * loss_decoder + self.rot_factor * loss_rot

        logging_output = {
            "loss": loss.data,
            "loss_encoder": loss_encoder.data,
            "loss_decoder": loss_decoder.data,
            "loss_rot": loss_rot.data,
            "ntokens": mask.int().sum(),
            "nsentences": mask.size()[0]}
        return loss, aa_mask.sum(-1).float().mean(), logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        batch_size = logging_outputs[0].get("nsentences")
        loss = sum([log["loss"].cpu() * batch_size for log in logging_outputs])
        loss_encoder = sum([log.get("loss_encoder").cpu() * batch_size for log in logging_outputs])
        loss_decoder = sum([log.get("loss_decoder").cpu() * batch_size for log in logging_outputs])
        loss_rot = sum([log.get("loss_rot").cpu() * batch_size for log in logging_outputs])
        sample_size = sum(log.get("ntokens", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss / sample_size / math.log(2), round=3
        )
        metrics.log_scalar(
            "sequence loss", loss_encoder / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "coordinate loss", loss_decoder / sample_size, sample_size, round=3
        )
        metrics.log_scalar(
            "rotation loss", loss_rot / sample_size, sample_size, round=3
        )
        metrics.log_scalar(
            "masked tokens", sample_size)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
