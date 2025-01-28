# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os, re
from typing import Any, Dict, Optional, List
from pathlib import Path
import urllib
import warnings
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
# from torch_cluster import knn_graph

from fairseq import checkpoint_utils
from fairseq.data.legacy.masked_lm_dictionary import MaskedLMDictionary
from fairseq.models import register_model, register_model_architecture, FairseqDecoder
from fairseq.models.transformer import (
    TransformerModel,
    base_architecture as transformer_base_architecture,
)
from fairseq.models.esm import ESM2
from fairseq.models.esm_modules import Alphabet
from fairseq.models.egnn import EGNN, SubstrateEGNN

device = torch.device("cuda")
DEFAULT_MAX_SOURCE_POSITIONS = 1024


def load_hub_workaround(url):
    try:
        data = torch.hub.load_state_dict_from_url(url, progress=False, map_location="cpu")
    except RuntimeError:
        # Pytorch version issue - see https://github.com/pytorch/pytorch/issues/43106
        fn = Path(url).name
        data = torch.load(
            f"{torch.hub.get_dir()}/checkpoints/{fn}",
            map_location="cpu",
        )
    except urllib.error.HTTPError as e:
        raise Exception(f"Could not load {url}, check if you specified a correct model name?")
    return data


def _has_regression_weights(model_name):
    """Return whether we expect / require regression weights;
    Right now that is all models except ESM-1v and ESM-IF"""
    return not ("esm1v" in model_name or "esm_if" in model_name)


def load_regression_hub(model_name):
    url = f"https://dl.fbaipublicfiles.com/fair-esm/regression/{model_name}-contact-regression.pt"
    regression_data = load_hub_workaround(url)
    return regression_data


def _load_model_and_alphabet_core_v2(model_data):
    def upgrade_state_dict(state_dict):
        """Removes prefixes 'model.encoder.sentence_encoder.' and 'model.encoder.'."""
        prefixes = ["encoder.sentence_encoder.", "encoder."]
        pattern = re.compile("^" + "|".join(prefixes))
        state_dict = {pattern.sub("", name): param for name, param in state_dict.items()}
        return state_dict

    cfg = model_data["cfg"]["model"]
    state_dict = model_data["model"]
    state_dict = upgrade_state_dict(state_dict)
    alphabet = Alphabet.from_architecture("ESM-1b")
    model = ESM2(
        num_layers=cfg.encoder_layers,
        embed_dim=cfg.encoder_embed_dim,
        attention_heads=cfg.encoder_attention_heads,
        alphabet=alphabet,
        token_dropout=cfg.token_dropout,
    )
    return model, alphabet, state_dict


def load_from_pretrained_models(pretrained_model_name):
    def _download_model_and_regression_data(model_name):
        url = f"https://dl.fbaipublicfiles.com/fair-esm/models/{model_name}.pt"
        model_data = load_hub_workaround(url)
        if _has_regression_weights(model_name):
            regression_data = load_regression_hub(model_name)
        else:
            regression_data = None
        return model_data, regression_data

    model_data, regression_data = _download_model_and_regression_data(pretrained_model_name)

    if regression_data is not None:
        model_data["model"].update(regression_data["model"])

    # if pretrained_model_name.startswith("esm2"):
    #     model, alphabet, model_state = _load_model_and_alphabet_core_v2(model_data)
    # else:
    #     model, alphabet, model_state = _load_model_and_alphabet_core_v1(model_data)
    model, alphabet, model_state = _load_model_and_alphabet_core_v2(model_data)

    expected_keys = set(model.state_dict().keys())
    found_keys = set(model_state.keys())

    if regression_data is None:
        expected_missing = {"contact_head.regression.weight", "contact_head.regression.bias"}
        error_msgs = []
        missing = (expected_keys - found_keys) - expected_missing
        if missing:
            error_msgs.append(f"Missing key(s) in state_dict: {missing}.")
        unexpected = found_keys - expected_keys
        if unexpected:
            error_msgs.append(f"Unexpected key(s) in state_dict: {unexpected}.")

        if error_msgs:
            raise RuntimeError(
                "Error(s) in loading state_dict for {}:\n\t{}".format(
                    model.__class__.__name__, "\n\t".join(error_msgs)
                )
            )
        if expected_missing - found_keys:
            warnings.warn(
                "Regression weights not found, predicting contacts will not produce correct results."
            )
    model.load_state_dict(model_state, strict=regression_data is not None)
    return model, alphabet


def get_edges(n_nodes, k, indices):
    rows, cols = [], []

    for i in range(n_nodes):
        for j in range(k):
            rows.append(i)
            cols.append(indices[i][j+1])

    edges = [rows, cols]   # L * 30
    return edges


def get_edges_batch(n_nodes, batch_size, coords, k=30, aa_mask=None):
    rows, cols = [], []
    # batch = torch.tensor(range(batch_size)).reshape(-1, 1).expand(-1, n_nodes).reshape(-1).to(device)
    # edges = knn_graph(coords, k=k, batch=batch, loop=False)
    # edges = edges[[1, 0]]

    coords = torch.where( torch.isinf(coords), torch.full_like(coords, 0), coords)
    coords = torch.where( torch.isnan(coords), torch.full_like(coords, 0), coords)

    for i in range(batch_size):
        # k = min(k, len(coords[i]))
        coords_i = coords[i][aa_mask[i]]
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(coords_i)
        distances, indices = nbrs.kneighbors(coords_i)  # [N, 30]
        edges = get_edges(coords_i.size()[0], k, indices)  # [[N*N], [N*N]]
        edges = [torch.LongTensor(edges[0]), torch.LongTensor(edges[1])]
        rows.append(edges[0] + n_nodes * i + 1)  # every sample in batch has its own graph
        cols.append(edges[1] + n_nodes * i + 1)
    edges = [torch.cat(rows).to(device), torch.cat(cols).to(device)]  # B * L * 30
    return edges


# from diffab
def quaternion_1ijk_to_rotation_matrix(q):
    """
    (1 + ai + bj + ck) -> R
    Args:
        q:  (..., 3)
    """
    b, c, d = torch.unbind(q, dim=-1)
    s = torch.sqrt(1 + b**2 + c**2 + d**2)
    a, b, c, d = 1/s, b/s, c/s, d/s

    o = torch.stack(
        (
            a**2 + b**2 - c**2 - d**2,  2*b*c - 2*a*d,  2*b*d + 2*a*c,
            2*b*c + 2*a*d,  a**2 - b**2 + c**2 - d**2,  2*c*d - 2*a*b,
            2*b*d - 2*a*c,  2*c*d + 2*a*b,  a**2 - b**2 - c**2 + d**2,
        ),
        -1,
    )
    return o.reshape(q.shape[:-1] + (3, 3))


@register_model("geometric_protein_model_pdb")
class GeometricProteinPDBModel(TransformerModel):
    @staticmethod
    def add_args(parser):
        """
        Add model-specific arguments to the parser.
        """
        TransformerModel.add_args(parser)
        parser.add_argument(
            "--pretrained-esm-model",
            type=str,
            metavar="ESM",
            help="Pretrained protein language model",
        )
        parser.add_argument(
            "--egnn-mode",
            type=str,
            default="full",
            help="version of EGNN architectures, and values could be full, rm-node, rm-edge, rm-all",
        )
        parser.add_argument(
            "--knn",
            type=int,
            default=30,
            help="number of k nearest neighbors",
        )
        parser.add_argument(
            "--inter-layers",
            type=int,
            default=4,
            help="number of sequence block for each stru-seq interleaving module",
        )

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)
        self.encoder_layers = args.encoder_layers
        self.mask_index = self.encoder.alphabet.mask_idx
        self.k = args.knn
        self.inter_layers = args.inter_layers
        emb_dim = args.encoder_embed_dim
        self.eps_rot_net = nn.Sequential(
            nn.Linear(emb_dim, emb_dim), nn.ReLU(),
            nn.Linear(emb_dim, emb_dim), nn.ReLU(),
            nn.Linear(emb_dim, 3)
        )

    @classmethod
    def build_model(self, args, task, cls_dictionary=MaskedLMDictionary):
        assert hasattr(args, "pretrained_esm_model"), (
            "You must specify a path for --pretrained-esm-model to use "
            "--arch transformer_from_pretrained_xlm"
        )
        """Build a new model instance."""

        return super().build_model(args, task)

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        model, alphabet = load_from_pretrained_models(args.pretrained_esm_model)
        # for param in model.parameters():
        #     param.requires_grad = False
        return model

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = EGNN(in_node_nf=args.encoder_embed_dim, hidden_nf=args.encoder_embed_dim, out_node_nf=3,
                       in_edge_nf=0, device=device, n_layers=args.decoder_layers, attention=True, mode=args.egnn_mode)
        return decoder

    def forward(self, src_tokens, coords, input_mask, aa_mask):
        coords = coords.clone()
        need_head_weights = False
        return_contacts = False
        if return_contacts:
            need_head_weights = True

        # prepare encoder
        batch_size, n_nodes = src_tokens.size()[0], src_tokens.size()[1]
        tokens = input_mask * self.mask_index + (input_mask != 1) * src_tokens
        repr_layers = [self.encoder_layers]
        assert tokens.ndim == 2
        padding_mask = tokens.eq(self.encoder.padding_idx)  # B, T

        x = self.encoder.embed_scale * self.encoder.embed_tokens(tokens)

        if self.encoder.token_dropout:
            x.masked_fill_((tokens == self.encoder.mask_idx).unsqueeze(-1), 0.0)
            # x: B x T x C
            mask_ratio_train = 0.15 * 0.8
            src_lengths = (~padding_mask).sum(-1)
            mask_ratio_observed = (tokens == self.encoder.mask_idx).sum(-1).to(x.dtype) / src_lengths
            x = x * (1 - mask_ratio_train) / (1 - mask_ratio_observed)[:, None, None]

        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

        repr_layers = set(repr_layers)
        hidden_representations = {}
        if 0 in repr_layers:
            hidden_representations[0] = x

        if need_head_weights:
            attn_weights = []

        # (B, T, E) => (T, B, E)
        x = x.transpose(0, 1)

        if not padding_mask.any():
            padding_mask = None

        # prepare decoder
        for i in range(coords.size(0)):
            for j in range(coords.size(1)):
                if input_mask[i][j] == 1:
                    theta = np.random.uniform(0, np.pi)
                    phi = np.random.uniform(0, np.pi * 2)
                    coords[i][j][0] = coords[i][j - 1][0] + 3.75 * np.sin(theta) * np.sin(phi)
                    coords[i][j][1] = coords[i][j - 1][1] + 3.75 * np.sin(theta) * np.cos(phi)
                    coords[i][j][2] = coords[i][j - 1][2] + 3.75 * np.cos(theta)
        coords = coords.reshape(-1, coords.size()[-1])  # [batch * length, 3]

        for layer_idx, layer in enumerate(self.encoder.layers):
            x, attn = layer(
                x,
                self_attn_padding_mask=padding_mask,
                need_head_weights=need_head_weights,
            )
            if (layer_idx + 1) in repr_layers:
                hidden_representations[layer_idx + 1] = x.transpose(0, 1)
            if need_head_weights:
                # (H, B, T, T) => (B, H, T, T)
                attn_weights.append(attn.transpose(1, 0))

            if (layer_idx + 1) % self.inter_layers == 0:
                decoder_layer_idx = int(layer_idx/self.inter_layers)
                x = x.transpose(0, 1).reshape(-1, x.size()[-1])  # [batch * length, hidden]
                # x: B * L * dim; edges: B * L * 30; coords: B * L * 3
                coords = coords.view(batch_size, -1, coords.size()[-1])  # [batch * length, 3]
                edges = get_edges_batch(n_nodes, batch_size, coords.detach().cpu(), self.k, aa_mask)
                coords = coords.reshape(-1, coords.size()[-1])  # [batch * length, 3]
                # edges = get_edges_batch(n_nodes, batch_size)
                x, coords, _ = self.decoder._modules["gcl_%d" % int(decoder_layer_idx)](x, edges, coords,
                                                                                        edge_attr=None,
                                                                                        batch_size=batch_size, k=self.k,
                                                                                        mask=input_mask.reshape(-1))
                x = x.reshape(batch_size, -1, x.size()[-1]).transpose(0, 1)

        repr_output = x.transpose(0, 1)
        x = self.encoder.emb_layer_norm_after(x)
        x = x.transpose(0, 1)  # (T, B, E) => (B, T, E)

        # last hidden representation should have layer norm applied
        if (layer_idx + 1) in repr_layers:
            hidden_representations[layer_idx + 1] = x

        x = self.encoder.lm_head(x)

        encoder_prob = F.softmax(x, dim=-1)

        # pred rotation
        quat = self.eps_rot_net(repr_output)
        pred_rot = quaternion_1ijk_to_rotation_matrix(quat)

        return encoder_prob, coords.view(batch_size, -1, 3), pred_rot

    def max_positions(self):
        """Maximum length supported by the model."""
        return (self.encoder.max_positions(), self.encoder.max_positions())


@register_model_architecture("geometric_protein_model_pdb", "geometric_protein_model_pdb")
def base_architecture(args):
    transformer_base_architecture(args)


@register_model_architecture("geometric_protein_model_pdb", "geometric_protein_model_pdb_base")
def transformer_vae_base(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 512)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_layers = getattr(args, "encoder_layers", 3)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 256)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 512)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_layers = getattr(args, "decoder_layers", 3)
    base_architecture(args)


@register_model_architecture("geometric_protein_model_pdb", "geometric_protein_model_pdb_esm")
def transformer_vae_esm(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 320)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1280)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.decoder_layers = getattr(args, "decoder_layers", 4)
    base_architecture(args)

