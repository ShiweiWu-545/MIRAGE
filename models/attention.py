import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.model_features import one_hot
from .common import mask_zero, global_to_local, local_to_global, normalize_vector


class SE(nn.Module):

    def __init__(self, in_chnls, ratio):
        super(SE, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.compress = nn.Conv2d(in_channels=in_chnls, out_channels=in_chnls * ratio, kernel_size=1)
        self.excitation = nn.Conv2d(in_channels=in_chnls * ratio, out_channels=in_chnls, kernel_size=1)

    def forward(self, x):
        out = self.squeeze(x)
        out = self.compress(out)
        out = F.relu(out)
        out = self.excitation(out)
        return torch.sigmoid(out)


def _alpha_from_logits(logits, mask, inf=1e5):
    N, L, _, _ = logits.size()
    mask_row = mask.view(N, L, 1, 1).expand_as(logits)
    mask_pair = mask_row * mask_row.permute(0, 2, 1, 3)

    logits = torch.where(mask_pair, logits, logits-inf)
    alpha = torch.softmax(logits, dim=2)
    alpha = torch.where(mask_row, alpha, torch.zeros_like(alpha))
    return alpha


def _heads(x, n_heads, n_ch):
    s = list(x.size())[:-1] + [n_heads, n_ch]
    return x.view(*s)


class GeometricAttention(nn.Module):

    def __init__(self, config, node_feat_dim, pair_sequence_feat_dim, spatial_attn_mode='CB', value_dim=16, query_key_dim=16, num_query_points=8, num_value_points=8, num_heads=12):
        super().__init__()
        self.config = config
        self.node_feat_dim = node_feat_dim
        self.pair_sequence_feat_dim = pair_sequence_feat_dim
        self.value_dim = value_dim
        self.query_key_dim = query_key_dim
        self.num_query_points = num_query_points
        self.num_value_points = num_value_points
        self.num_heads = num_heads

        assert spatial_attn_mode in ('CB', 'vpoint')
        self.spatial_attn_mode = spatial_attn_mode



        self.proj_query = nn.Linear(node_feat_dim, query_key_dim*num_heads, bias=False)
        self.proj_key = nn.Linear(node_feat_dim, query_key_dim*num_heads, bias=False)
        self.proj_value = nn.Linear(node_feat_dim, value_dim*num_heads, bias=False)


        self.proj_pair_bias = nn.Linear(pair_sequence_feat_dim, num_heads, bias=False)

        if self.config.feature.Gate.gate0:
            self.pair_sequence_gateInput = nn.Linear(pair_sequence_feat_dim, num_heads, bias=True)

            self.pair_sequence_gateInput.bias = nn.Parameter(torch.full([num_heads], fill_value=99.0, requires_grad=True))

        if self.config.feature.Gate.gate1:
            self.pair_sequence_gateValue = nn.Linear(pair_sequence_feat_dim, pair_sequence_feat_dim, bias=True)
            self.pair_sequence_gateValue.bias = nn.Parameter(torch.full([pair_sequence_feat_dim], fill_value=99.0, requires_grad=True))

        if self.config.feature.Gate.gate2:
            self.pair_sequence_gateOutput = nn.Linear(node_feat_dim, num_heads, bias=True)
            self.pair_sequence_gateOutput.bias = nn.Parameter(torch.full([num_heads], fill_value=99.0, requires_grad=True))


        self.spatial_coef = nn.Parameter(torch.full([1, 1, 1, self.num_heads], fill_value=np.log(np.exp(1.) - 1.)), requires_grad=True)

        if self.config.feature.Gate.gate3:
            self.pair_distance_gateInput = nn.Parameter(torch.full([1, 1, 1, self.num_heads], fill_value=99.0, requires_grad=True))

        if spatial_attn_mode == 'vpoint':
            self.proj_query_point = nn.Linear(node_feat_dim, num_query_points*num_heads*3, bias=False)
            self.proj_key_point = nn.Linear(node_feat_dim, num_query_points*num_heads*3, bias=False)
            self.proj_value_point = nn.Linear(node_feat_dim, num_value_points*num_heads*3, bias=False)


        if self.config.feature.pass3:
            self.distance_linear = nn.Linear(15, pair_sequence_feat_dim, bias=False)


        if spatial_attn_mode == 'CB':
            if self.config.feature.Side_chain_geometry:
                if self.config.feature.pass3:
                    self.out_transform = nn.Linear(
                        in_features=(num_heads*pair_sequence_feat_dim) + (num_heads*value_dim) + (num_heads*(3+3+1)) + (num_heads*pair_sequence_feat_dim),
                        out_features=node_feat_dim,
                    )
                else:
                    self.out_transform = nn.Linear(
                        in_features=(num_heads * pair_sequence_feat_dim) + (num_heads * value_dim) + (num_heads * (3 + 3 + 1)),
                        out_features=node_feat_dim,
                    )
            else:
                if self.config.feature.pass3:
                    self.out_transform = nn.Linear(
                        in_features=(num_heads*pair_sequence_feat_dim) + (num_heads*value_dim) + (num_heads*(3+3+1)) + (num_heads*pair_sequence_feat_dim),
                        out_features=node_feat_dim,
                    )
                else:
                    self.out_transform = nn.Linear(
                        in_features=(num_heads * pair_sequence_feat_dim) + (num_heads * value_dim),
                        out_features=node_feat_dim,
                    )
        elif spatial_attn_mode == 'vpoint':
            self.out_transform = nn.Linear(
                in_features = (num_heads*pair_sequence_feat_dim) + (num_heads*value_dim) + (num_heads*num_value_points*(3+3+1)),
                out_features = node_feat_dim,
            )
        self.layer_norm_node = nn.LayerNorm(node_feat_dim)
        if self.config.feature.LayerNorm:
            self.layer_norm_pair = nn.LayerNorm(pair_sequence_feat_dim)
            self.layer_norm_distance0 = nn.LayerNorm(pair_sequence_feat_dim)

        if self.config.feature.weight.SENet:
            self.se = SE(num_heads, 4)
        if self.config.feature.weight.Attention:
            self.weight_MLP = nn.Sequential(
                nn.Linear(node_feat_dim, node_feat_dim * 2),
                nn.ReLU(),
                nn.Linear(node_feat_dim * 2, node_feat_dim),
            )
            self.weight_abc = nn.Sequential(
                nn.Linear(self.num_heads * self.query_key_dim, node_feat_dim),
                nn.ReLU(),
                nn.Linear(node_feat_dim, 1),
                nn.Softplus()
            )
            self.weight_q = nn.Linear(node_feat_dim, self.num_heads * self.query_key_dim, bias=False)
            self.weight_k = nn.Linear(node_feat_dim, self.num_heads * self.query_key_dim, bias=False)
            self.weight_v = nn.Linear(node_feat_dim, self.num_heads * self.query_key_dim, bias=False)
        if self.config.feature.weight.Attention:
            self.weight_LayerNorm = nn.LayerNorm(self.num_heads * self.query_key_dim)

    def _node_logits(self, x):
        query_l = _heads(self.proj_query(x), self.num_heads, self.query_key_dim)
        key_l = _heads(self.proj_key(x), self.num_heads, self.query_key_dim)

        query_l = query_l.permute(0, 2, 1, 3)
        key_l = key_l.permute(0, 2, 3, 1)

        if self.config.feature.pass0:
            logits = 1 / np.sqrt(query_l.size()[-1]) * torch.matmul(query_l, key_l)
        else:
            logits = torch.matmul(query_l, key_l)
        logits = logits.permute(0, 2, 3, 1)

        return logits

    def _pair_logits(self, z):
        logits_pair_sequence = self.proj_pair_bias(z)
        if self.config.feature.Gate.gate0:
            pair_gateInput = torch.sigmoid(self.pair_sequence_gateInput(z))
            logits_pair_sequence = logits_pair_sequence * pair_gateInput
        return logits_pair_sequence

    def _distance_logits(self, d):

        if self.config.feature.LayerNorm:
            d = d / torch.max(d)

        gamma = F.softplus(self.spatial_coef)
        if self.config.feature.LayerNorm:
            d = d[:, :, :, None].expand(-1, -1, -1, self.num_heads)
        else:
            d = (d ** 2)[:, :, :, None].expand(-1, -1, -1, self.num_heads)

        logtis_distance = d * ((-1 * gamma * np.sqrt(2 / 9)) / 2)


        if self.config.feature.Gate.gate3:
            pair_distance_gateInput = torch.sigmoid(self.pair_distance_gateInput)
            logtis_distance = pair_distance_gateInput * logtis_distance

        return logtis_distance

    def _energy_logits(self, e):
        logits_energy = self.proj_energy_bias(e)
        return logits_energy

    def _beta_logits(self, R, t, p_CB):
        N, L, _ = t.size()
        qk = p_CB[:, :, None, :].expand(N, L, self.num_heads, 3)
        sum_sq_dist = ((qk.unsqueeze(2) - qk.unsqueeze(1)) ** 2).sum(-1)
        gamma = F.softplus(self.spatial_coef)
        logtis_beta = sum_sq_dist * ((-1 * gamma * np.sqrt(2 / 9)) / 2)
        return logtis_beta

    def _spatial_logits(self, R, t, x):
        N, L, _ = t.size()

        query_points = _heads(self.proj_query_point(x), self.num_heads*self.num_query_points, 3)
        query_points = local_to_global(R, t, query_points)
        query_s = query_points.reshape(N, L, self.num_heads, -1)

        key_points = _heads(self.proj_key_point(x), self.num_heads*self.num_query_points, 3)
        key_points = local_to_global(R, t, key_points)
        key_s = key_points.reshape(N, L, self.num_heads, -1)

        sum_sq_dist = ((query_s.unsqueeze(2) - key_s.unsqueeze(1)) ** 2).sum(-1)
        gamma = F.softplus(self.spatial_coef)
        logits_pair_distance = sum_sq_dist * ((-1 * gamma * np.sqrt(2 / (9 * self.num_query_points))) / 2)
        return logits_pair_distance


    def _pair_distance_aggregation(self, alpha, d):
        d = one_hot(d, bins=list(np.linspace(27.0 / 8.0, 171.0 / 8.0, 15)))
        d = self.distance_linear(d)
        if self.config.feature.LayerNorm:
            d = self.layer_norm_distance0(d)
        N, L = d.shape[:2]
        feat_p2n = alpha.unsqueeze(-1) * d.unsqueeze(-2)
        feat_p2n = feat_p2n.sum(dim=2)
        return feat_p2n.reshape(N, L, -1)

    def _pair_aggregation(self, alpha, z):

        if self.config.feature.Gate.gate1:
            pair_gateValue = torch.sigmoid(self.pair_sequence_gateValue(z))
            z = z * pair_gateValue

        N, L = z.shape[:2]
        feat_p2n = alpha.unsqueeze(-1) * z.unsqueeze(-2)
        feat_p2n = feat_p2n.sum(dim=2)

        if self.config.feature.Gate.gate2:
            z = z.permute(0, 1, 3, 2)
            pair_gateOutput = torch.sigmoid(self.pair_sequence_gateOutput(z)).permute(0, 1, 3, 2)
            feat_p2n = feat_p2n * pair_gateOutput

        return feat_p2n.reshape(N, L, -1)

    def _node_aggregation(self, alpha, x):
        N, L = x.shape[:2]
        value_l = _heads(self.proj_value(x), self.num_heads, self.query_key_dim)



        feat_node = alpha.unsqueeze(-1) * value_l.unsqueeze(1)
        feat_node = feat_node.sum(dim=2)
        return feat_node.reshape(N, L, -1)

    def _beta_aggregation(self, alpha, R, t, p_CB, x):
        N, L, _ = t.size()
        v = p_CB[:, :, None, :].expand(N, L, self.num_heads, 3)
        aggr = alpha.reshape(N, L, L, self.num_heads, 1) * v.unsqueeze(1)
        aggr = aggr.sum(dim=2)

        feat_points = global_to_local(R, t, aggr)
        feat_distance = feat_points.norm(dim=-1)
        feat_direction = normalize_vector(feat_points, dim=-1, eps=1e-4)

        feat_spatial = torch.cat([
            feat_points.reshape(N, L, -1),
            feat_distance.reshape(N, L, -1),
            feat_direction.reshape(N, L, -1),
        ], dim=-1)

        return feat_spatial

    def _spatial_aggregation(self, alpha, R, t, x):
        N, L, _ = t.size()
        value_points = _heads(self.proj_value_point(x), self.num_heads*self.num_value_points, 3)
        value_points = local_to_global(R, t, value_points.reshape(N, L, self.num_heads, self.num_value_points, 3))
        aggr_points = alpha.reshape(N, L, L, self.num_heads, 1, 1) * value_points.unsqueeze(1)
        aggr_points = aggr_points.sum(dim=2)

        feat_points = global_to_local(R, t, aggr_points)
        feat_distance = feat_points.norm(dim=-1)
        feat_direction = normalize_vector(feat_points, dim=-1, eps=1e-4)

        feat_spatial = torch.cat([
            feat_points.reshape(N, L, -1),
            feat_distance.reshape(N, L, -1),
            feat_direction.reshape(N, L, -1),
        ], dim=-1)

        return feat_spatial

    def forward_beta(self, R, t, p_CB, x, z, d, mask):
        if self.config.feature.LayerNorm:
            x = self.layer_norm_node(x)
            z = self.layer_norm_pair(z)



        logits_node = self._node_logits(x)
        logits_pair_sequence = self._pair_logits(z)
        logits_pair_distance = self._distance_logits(d)

        a, b, c = 1, 1, 1
        if self.config.feature.weight.Attention:
            a, b, c = self.weight_Attention(logits_node, logits_pair_sequence, logits_pair_distance)

        logits_sum = np.sqrt(1 / 3) * (a * logits_node + b * logits_pair_sequence + c * logits_pair_distance)

        if self.config.feature.weight.SENet:
            tem = logits_sum.permute(0, 3, 1, 2)
            se_weight = self.se(tem).permute(0, 2, 3, 1)
            logits_sum = logits_sum * se_weight

        alpha = _alpha_from_logits(logits_sum, mask)


        feat_p2n = self._pair_aggregation(alpha, z)
        feat_node = self._node_aggregation(alpha, x)
        feat_spatial = self._beta_aggregation(alpha, R, t, p_CB, x)
        if self.config.feature.pass3:
            feat_distance = self._pair_distance_aggregation(alpha, d)


        if self.config.feature.Side_chain_geometry:
            if self.config.feature.pass3:
                feat_all = self.out_transform(torch.cat([feat_p2n, feat_node, feat_spatial, feat_distance], dim=-1))
            else:
                feat_all = self.out_transform(torch.cat([feat_p2n, feat_node, feat_spatial], dim=-1))
        else:
            if self.config.feature.pass3:
                feat_all = self.out_transform(torch.cat([feat_p2n, feat_node, feat_distance], dim=-1))
            else:
                feat_all = self.out_transform(torch.cat([feat_p2n, feat_node], dim=-1))


        feat_all = mask_zero(mask.unsqueeze(-1), feat_all)
        if self.config.feature.LayerNorm:
            x_updated = x + feat_all
        else:
            x_updated = self.layer_norm_node(x + feat_all)
        return x_updated

    def weight_Attention(self, node, pair, distance):
        x = self.weight_process(node, pair, distance)
        q = _heads(self.weight_q(x), self.num_heads, self.value_dim)
        k = _heads(self.weight_k(x), self.num_heads, self.value_dim)
        v = _heads(self.weight_v(x), self.num_heads, self.value_dim)

        weight_x = 1 / np.sqrt(self.value_dim) * torch.matmul(q.permute(0, 2, 1, 3), k.permute(0, 2, 3, 1))
        weight_x = torch.softmax(weight_x, dim=2)
        y = torch.matmul(weight_x, v.permute(0, 2, 1, 3)).permute(0, 2, 1, 3).reshape(x.size()[0], x.size()[1], -1)
        y = self.weight_abc(self.weight_LayerNorm(y)).squeeze(dim=2)
        return torch.mean(y[:, 0]), torch.mean(y[:, 1]), torch.sigmoid(torch.mean(y[:, 2]))


    def weight_Attention4D(self, node, pair, distance, energy):
        x = self.weight_process4D(node, pair, distance, energy)
        q = _heads(self.weight_q(x), self.num_heads, self.value_dim)
        k = _heads(self.weight_k(x), self.num_heads, self.value_dim)
        v = _heads(self.weight_v(x), self.num_heads, self.value_dim)

        weight_x = 1 / np.sqrt(self.value_dim) * torch.matmul(q.permute(0, 2, 1, 3), k.permute(0, 2, 3, 1))
        weight_x = torch.softmax(weight_x, dim=2)
        y = torch.matmul(weight_x, v.permute(0, 2, 1, 3)).permute(0, 2, 1, 3).reshape(x.size()[0], x.size()[1], -1)
        y = self.weight_abc(self.weight_LayerNorm(y)).squeeze(dim=2)
        return torch.mean(y[:, 0]), torch.mean(y[:, 1]), torch.sigmoid(torch.mean(y[:, 2])), torch.mean(y[:, 3])

    def weight_process(self, node, pair, distance):
        node = node.sum(dim=3).sum(dim=2)[:, None, :]
        pair = pair.sum(dim=3).sum(dim=2)[:, None, :]
        distance = distance.sum(dim=3).sum(dim=2)[:, None, :]
        return torch.cat([node, pair, distance], dim=1)

    def weight_process4D(self, node, pair, distance, energy):
        node = node.sum(dim=3).sum(dim=2)[:, None, :]
        pair = pair.sum(dim=3).sum(dim=2)[:, None, :]
        distance = distance.sum(dim=3).sum(dim=2)[:, None, :]
        energy = energy.sum(dim=3).sum(dim=2)[:, None, :]
        return torch.cat([node, pair, distance, energy], dim=1)

    def forward_vpoint(self, R, t, p_CB, x, z, d, mask):

        logits_node = self._node_logits(x)
        logits_pair_sequence = self._pair_logits(z)
        logits_pair_distance = self._spatial_logits(R, t, x)

        logits_sum = logits_node + logits_pair_sequence + logits_pair_distance
        alpha = _alpha_from_logits(logits_sum * np.sqrt(1 / 3), mask)


        feat_p2n = self._pair_aggregation(alpha, z)
        feat_node = self._node_aggregation(alpha, x)
        feat_spatial = self._spatial_aggregation(alpha, R, t, x)


        feat_all = self.out_transform(torch.cat([feat_p2n, feat_node, feat_spatial], dim=-1))
        feat_all = mask_zero(mask.unsqueeze(-1), feat_all)
        x_updated = self.layer_norm_node(x + feat_all)
        return x_updated

    def forward(self, R, t, p_CB, x, z, d, mask):
        if self.spatial_attn_mode == 'CB':
            return self.forward_beta(R, t, p_CB, x, z, d, mask)
        else:
            return self.forward_vpoint(R, t, p_CB, x, z, d, mask)


class CrossAttention(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1024, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 128)
        )

        self.value_dim = 16
        self.num_heads = 12
        self.weight_q = nn.Linear(128, self.num_heads * self.value_dim, bias=False)
        self.weight_k = nn.Linear(128, self.num_heads * self.value_dim, bias=False)
        self.weight_v = nn.Linear(128, self.num_heads * self.value_dim, bias=False)

        self.weight = nn.Linear(self.num_heads * self.value_dim, 128)

    def forward(self, x, wt_prottrans, mut_prottrans):
        feat = self.mlp(wt_prottrans) - self.mlp(mut_prottrans)
        q = _heads(self.weight_q(feat), self.num_heads, self.value_dim)
        k = _heads(self.weight_k(x), self.num_heads, self.value_dim)
        v = _heads(self.weight_v(x), self.num_heads, self.value_dim)

        weight_x = 1 / np.sqrt(self.value_dim) * torch.matmul(q.permute(0, 2, 1, 3), k.permute(0, 2, 3, 1))
        weight_x = torch.softmax(weight_x, dim=2)
        y = torch.matmul(weight_x, v.permute(0, 2, 1, 3)).permute(0, 2, 1, 3).reshape(x.size()[0], x.size()[1], -1)
        y = self.weight(y)
        return y


class CrossAttentionDG(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1024, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 128)
        )

        self.value_dim = 16
        self.num_heads = 12
        self.weight_q = nn.Linear(128, self.num_heads * self.value_dim, bias=False)
        self.weight_k = nn.Linear(128, self.num_heads * self.value_dim, bias=False)
        self.weight_v = nn.Linear(128, self.num_heads * self.value_dim, bias=False)

        self.weight = nn.Linear(self.num_heads * self.value_dim, 128)

    def forward(self, x, wt_prottrans):
        feat = self.mlp(wt_prottrans)
        q = _heads(self.weight_q(feat), self.num_heads, self.value_dim)
        k = _heads(self.weight_k(x), self.num_heads, self.value_dim)
        v = _heads(self.weight_v(x), self.num_heads, self.value_dim)

        weight_x = 1 / np.sqrt(self.value_dim) * torch.matmul(q.permute(0, 2, 1, 3), k.permute(0, 2, 3, 1))
        weight_x = torch.softmax(weight_x, dim=2)
        y = torch.matmul(weight_x, v.permute(0, 2, 1, 3)).permute(0, 2, 1, 3).reshape(x.size()[0], x.size()[1], -1)
        y = self.weight(y)
        return y


class GAEncoder(nn.Module):

    def __init__(self, config, node_feat_dim, pair_sequence_feat_dim, num_layers, spatial_attn_mode='CB'):
        super().__init__()
        self.config = config
        self.blocks = nn.ModuleList([
            GeometricAttention(self.config, node_feat_dim, pair_sequence_feat_dim, spatial_attn_mode=spatial_attn_mode)
            for _ in range(num_layers)
        ])

    def GeometricAttention_blocks(self, R, t, p_CB, x, z, d, mask, output):
        x = x + output
        for block in self.blocks:
            x = block(R, t, p_CB, x, z, d, mask)
        return x

    def recycle(self, R, t, p_CB, x, z, d, mask, output):
        with torch.no_grad():
            for num_cycle in range(self.config.feature.Recycling[1]):
                output = self.GeometricAttention_blocks(R, t, p_CB, x, z, d, mask, output)


        x = self.GeometricAttention_blocks(R, t, p_CB, x, z, d, mask, output)
        x = 1 / (self.config.feature.Recycling[1] + 1) * x
        return x

    def forward(self, R, t, p_CB, x, z, d, mask):
        output = 0
        if self.config.feature.Recycling[0]:
            x = self.recycle(R, t, p_CB, x, z, d, mask, output)
        else:
            x = self.GeometricAttention_blocks(R, t, p_CB, x, z, d, mask, output)
        return x
