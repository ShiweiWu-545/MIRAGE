import torch


def one_hot(x, bins):
    if x.size()[-1] != 1:
        x = x.unsqueeze(-1)
    bin_tensor = torch.tensor(bins, dtype=x.dtype, device=x.device)
    if len(x.size()) == 3:
        i, j, _ = x.size()
        out = torch.zeros((i, j, len(bins)), dtype=x.dtype, device=x.device)
        indices = torch.argmin(torch.abs(x - bin_tensor[None, None, :]), dim=2)
        return out.scatter_(2, indices.unsqueeze(2), 1)
    if len(x.size()) == 4:
        i, j, k, _ = x.size()
        out = torch.zeros((i, j, k, len(bins)), dtype=x.dtype, device=x.device)
        indices = torch.argmin(torch.abs(x - bin_tensor[None, None, None, :]), dim=3)
        return out.scatter_(3, indices.unsqueeze(3), 1)
    raise ValueError(f"Unsupported tensor rank for one_hot: {len(x.size())}")


def introduce_contact_area(config, residue_data, contact_area):
    if config.feature.contact_area[1] == "Residue level":
        _, length, _ = residue_data.size()
        contact_area = contact_area[:, None, None].expand(-1, length, -1)
        return torch.cat((residue_data, contact_area), dim=-1)
    return residue_data
