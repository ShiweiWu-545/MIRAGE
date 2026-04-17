import torch
import torch.nn as nn

from models.common import PositionalEncoding, construct_3d_basis, global_to_local
from data.my_config import *


class PerResidueEncoder(nn.Module):

    def __init__(self, config, feat_dim):
        super().__init__()
        self.config = config
        self.aatype_embed = nn.Embedding(21, feat_dim)
        self.torsion_embed = PositionalEncoding()
        if self.config.feature.res_prottrans:
            self.mlp = nn.Sequential(

                nn.Linear(1024 + 882, feat_dim * 2),
                nn.ReLU(),
                nn.Linear(feat_dim * 2, feat_dim),
                nn.ReLU(),
                nn.Linear(feat_dim, feat_dim),
                nn.ReLU(),
                nn.Linear(feat_dim, feat_dim)
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(21 * self.config.feature.Atom[1] * 3 + feat_dim, feat_dim * 2),

                nn.ReLU(),
                nn.Linear(feat_dim * 2, feat_dim),
                nn.ReLU(),
                nn.Linear(feat_dim, feat_dim),
                nn.ReLU(),
                nn.Linear(feat_dim, feat_dim)
            )
        if self.config.feature.Dropout[0]:
            self.dropout = nn.Dropout(self.config.feature.Dropout[1])

    def forward(self, aa, pos14, prottrans, atom_mask):
        N, L = aa.size()


        R = construct_3d_basis(pos14[:, :, 1], pos14[:, :, 2], pos14[:, :, 0])
        t = pos14[:, :, 1]

        crd14 = global_to_local(R, t, pos14)

        crd14_mask = atom_mask[:, :, :, None].expand_as(crd14)



        crd14 = torch.where(crd14_mask, crd14, torch.zeros_like(crd14))


        aa_expand = aa[:, :, None, None, None].expand(N, L, 21, self.config.feature.Atom[1], 3)





        debug0 = torch.arange(0, 21)[None, None, :, None, None].expand(N, L, 21, self.config.feature.Atom[1], 3)
        rng_expand = torch.arange(0, 21)[None, None, :, None, None].expand(N, L, 21, self.config.feature.Atom[1], 3).to(aa_expand)

        place_mask = (aa_expand == rng_expand)

        crd_expand = crd14[:, :, None, :, :].expand(N, L, 21, self.config.feature.Atom[1], 3)



        crd_expand = torch.where(place_mask, crd_expand, torch.zeros_like(crd_expand))
        crd_feat = crd_expand.reshape(N, L, 21 * self.config.feature.Atom[1] * 3)

        aa_feat = self.aatype_embed(aa)


        debug1 = torch.cat([crd_feat, aa_feat], dim=-1)
        if self.config.feature.res_prottrans:

            out_feat = self.mlp(torch.cat([crd_feat, prottrans], dim=-1))
        else:
            out_feat = self.mlp(torch.cat([crd_feat, aa_feat], dim=-1))

        if self.config.feature.Dropout[0]:
            out_feat = self.dropout(out_feat)
        return out_feat

    def write_prottrans_embedding(self, feat, a):
        out_feat = self.mlp(feat)








