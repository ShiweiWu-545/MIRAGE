import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.attention import CrossAttention, CrossAttentionDG, GAEncoder
from models.common import construct_3d_basis, get_pos_CA, get_pos_CB
from models.residue import PerResidueEncoder
from utils.model_features import introduce_contact_area, one_hot
from utils.protein import ATOM_C, ATOM_CA, ATOM_N


class ComplexEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cfg = config.model
        self.config = config
        self.relpos_embedding = nn.Embedding(self.cfg.max_relpos * 2 + 2, self.cfg.pair_sequence_feat_dim)
        self.residue_encoder = PerResidueEncoder(self.config, self.cfg.node_feat_dim)
        if self.config.feature.pass1:
            self.aa_pair1 = nn.Linear(21, self.cfg.pair_sequence_feat_dim, bias=False)
            self.aa_pair2 = nn.Linear(21, self.cfg.pair_sequence_feat_dim, bias=False)
        if self.config.feature.pass2:
            self.layer_norm_pair = nn.LayerNorm(self.cfg.pair_sequence_feat_dim)
            self.pair_distance_one_hot = nn.Linear(15, self.cfg.pair_sequence_feat_dim, bias=False)

        if self.cfg.geomattn is not None:
            self.ga_encoder = GAEncoder(
                config=self.config,
                node_feat_dim=self.cfg.node_feat_dim,
                pair_sequence_feat_dim=self.cfg.pair_sequence_feat_dim,
                num_layers=self.cfg.geomattn.num_layers,
                spatial_attn_mode=self.cfg.geomattn.spatial_attn_mode,
            )
        else:
            self.out_mlp = nn.Sequential(
                nn.Linear(self.cfg.node_feat_dim, self.cfg.node_feat_dim),
                nn.ReLU(),
                nn.Linear(self.cfg.node_feat_dim, self.cfg.node_feat_dim),
                nn.ReLU(),
                nn.Linear(self.cfg.node_feat_dim, self.cfg.node_feat_dim),
            )

    def forward(self, pos14, aa, seq, chain, prottrans, mask_atom):
        same_chain = chain[:, None, :] == chain[:, :, None]
        relpos = (seq[:, None, :] - seq[:, :, None]).clamp(
            min=-self.cfg.max_relpos,
            max=self.cfg.max_relpos,
        ) + self.cfg.max_relpos
        if self.config.feature.relpos_sparse:
            relpos = torch.where(
                torch.full_like(relpos, fill_value=64) == relpos,
                torch.full_like(relpos, fill_value=0),
                relpos,
            )
            relpos = torch.where(same_chain, relpos, torch.full_like(relpos, fill_value=0))
        else:
            relpos = torch.where(
                same_chain,
                relpos,
                torch.full_like(relpos, fill_value=self.cfg.max_relpos * 2 + 1),
            )

        pair_sequence_feat = self.relpos_embedding(relpos)
        if self.config.feature.pass1:
            aa_one_hot = one_hot(aa, bins=list(range(21))).to(pair_sequence_feat)
            pair_sequence_feat = (
                pair_sequence_feat + self.aa_pair1(aa_one_hot)[:, None, :, :] + self.aa_pair2(aa_one_hot)[:, :, None, :]
            ) / 16

        res_feat = self.residue_encoder(aa, pos14, prottrans, mask_atom)
        cb_pos = get_pos_CB(pos14, mask_atom)
        pair_distance_feat = torch.sqrt(((cb_pos.unsqueeze(2) - cb_pos.unsqueeze(1)) ** 2).sum(-1))

        if self.config.feature.pass2:
            pair_distance_feat_one_hot = one_hot(
                pair_distance_feat,
                bins=list(np.linspace(27.0 / 8.0, 171.0 / 8.0, 15)),
            )
            pair_sequence_feat = self.layer_norm_pair(pair_sequence_feat) + self.pair_distance_one_hot(
                pair_distance_feat_one_hot
            )

        t = pos14[:, :, ATOM_CA]
        r = construct_3d_basis(pos14[:, :, ATOM_CA], pos14[:, :, ATOM_C], pos14[:, :, ATOM_N])
        mask_residue = mask_atom[:, :, ATOM_CA]
        if self.config.feature.Spatial_distance == "CA":
            res_feat = self.ga_encoder(
                r,
                t,
                get_pos_CA(pos14, mask_atom),
                res_feat,
                pair_sequence_feat,
                pair_distance_feat,
                mask_residue,
            )
        if self.config.feature.Spatial_distance == "CB":
            res_feat = self.ga_encoder(
                r,
                t,
                get_pos_CB(pos14, mask_atom),
                res_feat,
                pair_sequence_feat,
                pair_distance_feat,
                mask_residue,
            )
        return res_feat


class DDGReadout(nn.Module):
    def __init__(self, config, feat_dim):
        super().__init__()
        contact_dim = 0
        self.config = config
        if self.config.feature.contact_area[0]:
            contact_dim = 1
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim * 2 + contact_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim),
        )

        if self.config.feature.Dropout[0]:
            self.dropout = nn.Dropout(self.config.feature.Dropout[1])

        if self.config.feature.CrossAttention:
            self.CrossAttention = CrossAttention()

        self.project = nn.Linear(feat_dim, 1, bias=False)

    def forward(self, node_feat_wt, node_feat_mut, contact_area, wt_prottrans, mut_prottrans, mask=None):
        feat_wm = torch.cat([node_feat_wt, node_feat_mut], dim=-1)
        feat_mw = torch.cat([node_feat_mut, node_feat_wt], dim=-1)

        if self.config.feature.contact_area[0]:
            feat_wm = introduce_contact_area(self.config, feat_wm, contact_area)
            feat_mw = introduce_contact_area(self.config, feat_mw, contact_area)

        feat_diff = self.mlp(feat_wm) - self.mlp(feat_mw)

        if self.config.feature.Dropout[0]:
            feat_diff = self.dropout(feat_diff)

        if self.config.feature.CrossAttention:
            feat_diff = self.CrossAttention(feat_diff, wt_prottrans, mut_prottrans)

        per_residue_ddg = self.project(feat_diff).squeeze(-1)
        if mask is not None:
            per_residue_ddg = per_residue_ddg * mask
        return per_residue_ddg.sum(dim=1)


class DGReadout(nn.Module):
    def __init__(self, config, feat_dim):
        super().__init__()
        contact_dim = 0
        self.config = config
        if self.config.feature.contact_area[0]:
            contact_dim = 1
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim + contact_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim),
        )

        if self.config.feature.Dropout[0]:
            self.dropout = nn.Dropout(self.config.feature.Dropout[1])

        if self.config.feature.CrossAttention:
            if self.config.model.mission == "DDG":
                self.CrossAttention = CrossAttention()
            elif self.config.model.mission == "DG":
                self.CrossAttention = CrossAttentionDG()

        self.project = nn.Linear(feat_dim, 1, bias=False)

    def forward(self, node_feat_wt, node_feat_mut, contact_area, wt_prottrans, mut_prottrans, mask=None):
        feat_wm = node_feat_wt

        if self.config.feature.contact_area[0]:
            feat_wm = introduce_contact_area(self.config, feat_wm, contact_area)

        feat_diff = self.mlp(feat_wm)

        if self.config.feature.Dropout[0]:
            feat_diff = self.dropout(feat_diff)

        if self.config.feature.CrossAttention:
            feat_diff = self.CrossAttention(feat_diff, wt_prottrans)

        per_residue_dg = self.project(feat_diff).squeeze(-1)
        if mask is not None:
            per_residue_dg = per_residue_dg * mask
        return per_residue_dg.sum(dim=1)


class DDGPredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        model_config = config.model
        self.config = config
        self.encoder = ComplexEncoder(self.config)
        self.ddG_readout = DDGReadout(self.config, model_config.node_feat_dim)
        self.dG_readout = DGReadout(self.config, model_config.node_feat_dim)

    def forward(self, complex_wt, complex_mut, contact_area=None, ddG_true=None):
        if self.config.model.mission == "DDG":
            mask_atom_wt = complex_wt["pos14_mask"].all(dim=-1)
            mask_atom_mut = complex_mut["pos14_mask"].all(dim=-1)

            feat_wt = self.encoder(
                complex_wt["pos14"],
                complex_wt["aa"],
                complex_wt["seq"],
                complex_wt["chain_seq"],
                complex_wt["prottrans"],
                mask_atom_wt,
            )
            feat_mut = self.encoder(
                complex_mut["pos14"],
                complex_mut["aa"],
                complex_mut["seq"],
                complex_mut["chain_seq"],
                complex_wt["prottrans"],
                mask_atom_mut,
            )

            mask_res = mask_atom_wt[:, :, ATOM_CA]
            ddG_pred = self.ddG_readout(
                feat_wt,
                feat_mut,
                contact_area,
                complex_wt["prottrans"],
                complex_mut["prottrans"],
                mask_res,
            )

            if ddG_true is None:
                return ddG_pred
            return {"ddG": F.mse_loss(ddG_pred, ddG_true)}, ddG_pred

        if self.config.model.mission == "DG":
            mask_atom_wt = complex_wt["pos14_mask"].all(dim=-1)

            feat_wt = self.encoder(
                complex_wt["pos14"],
                complex_wt["aa"],
                complex_wt["seq"],
                complex_wt["chain_seq"],
                complex_wt["res_prottrans"],
                mask_atom_wt,
            )

            mask_res = mask_atom_wt[:, :, ATOM_CA]
            dG_pred = self.dG_readout(
                feat_wt,
                None,
                contact_area,
                complex_wt["res_prottrans"],
                None,
                mask_res,
            )

            if ddG_true is None:
                return dG_pred
            return {"dG": F.mse_loss(dG_pred, ddG_true)}, dG_pred

        raise ValueError(f"Unsupported model mission: {self.config.model.mission}")
