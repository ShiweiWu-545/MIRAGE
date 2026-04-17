import torch
import torch.nn.functional as F
import numpy as np



from typing import Optional

from torch.nn import functional as F

def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    r, i, j, k = torch.unbind(quaternions, -1)

    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = angles * 0.5
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )


    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions

def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))

def get_ang(a, b, c):
    v = a - b
    w = c - b
    v /= torch.norm(v, dim=-1, keepdim=True)
    w /= torch.norm(w, dim=-1, keepdim=True)
    vw = torch.sum(v * w, dim=-1)

    return torch.acos(vw)


def get_dih(a, b, c, d):
    b0 = a - b
    b1 = c - b
    b2 = d - c

    b1 /= torch.norm(b1, dim=-1, keepdim=True)

    v = b0 - torch.sum(b0 * b1, dim=-1, keepdim=True) * b1
    w = b2 - torch.sum(b2 * b1, dim=-1, keepdim=True) * b1

    x = torch.sum(v * w, dim=-1)
    y = torch.sum(torch.cross(b1, v, dim=-1) * w, dim=-1)

    return torch.atan2(y, x)


def backbone_torsion(N, Ca, C, O):

    omega = get_dih(Ca[:-1], C[:-1], N[1:], Ca[1:])
    phi = get_dih(C[:-1], N[1:], Ca[1:], C[1:])
    psi = get_dih(N[:-1], Ca[:-1], C[:-1], O[:-1])

    omega = F.pad(omega, (0, 1))
    phi = F.pad(phi, (1, 0))
    psi = F.pad(psi, (0, 1))

    omega = omega % (2 * np.pi)
    phi = phi % (2 * np.pi)
    pspsi_af2i = psi % (2 * np.pi)

    return omega, phi, psi


def get_rotation(p1, o, p2):
    v1 = p1 - o
    v2 = p2 - o
    e1 = v1 / np.linalg.norm(v1 + 1e-12, ord=2)
    e2 = v2 - v2.dot(e1) * e1
    e2 = e2 / np.linalg.norm(e2 + 1e-12, ord=2)
    e3 = np.cross(e1, e2)
    r = np.stack([e1, e2, e3]).T
    return r


def get_batch_rotation(p1, o, p2):
    v1 = p1 - o
    v2 = p2 - o
    e1 = v1 / np.linalg.norm(v1 + 1e-12, ord=2, keepdims=True, axis=-1)
    e2 = v2 - (v2 * e1).sum(axis=-1, keepdims=True) * e1
    e3 = np.cross(e1, e2)
    e = np.stack([e1, e2, e3]).transpose(1, 2, 0)
    return e


def normalize_coord_by_first_res(coords_norm, idx_ca=1, idx_c=2):
    Ca0 = coords_norm[0, idx_ca]
    C = coords_norm[0, idx_c]
    Ca1 = coords_norm[1, idx_ca]
    r = get_rotation(Ca1, Ca0, C)
    if np.isnan(r).any():
        r = np.eye(3)
    t = Ca0
    coords_norm = coords_norm - Ca0
    coords_norm = coords_norm @ r
    return coords_norm


def get_local_rotatation(src, dst):
    zeros = torch.zeros_like(src)
    angles = get_ang(src, zeros, dst)

    n = torch.cross(src, dst)
    n = F.normalize(n, p=2, dim=-1)

    axis_angle = angles[..., None] * -n
    r_t = pytorch3d.transforms.axis_angle_to_matrix(axis_angle).float()
    is_collinear = torch.logical_or(torch.isnan(angles), torch.abs(angles) < 1e-4)
    is_collinear = is_collinear[..., None, None].expand_as(r_t)
    eye = torch.eye(3, device=r_t.device).unsqueeze(0).expand_as(r_t)

    r_t = torch.where(is_collinear, eye, r_t)
    return r_t


def apply_rigid(x: torch.Tensor, rotation: torch.Tensor, translation: Optional[torch.Tensor] = None):
    if translation is None:
        translation = torch.zeros_like(x)
    assert len(x.shape) == len(translation.shape)
    x = x[..., None, :]
    translation = translation[..., None, :]
    out = x @ rotation + translation
    out = out.squeeze(-2)
    return out


def inv_rigid(rotation, translation):
    translation = translation[..., None, :]
    rotation_new = rotation.transpose(-1, -2)
    translation_new = -translation @ rotation_new
    translation_new = translation_new.squeeze(-2)

    return rotation_new, translation_new


def get_frame_from_coords(coords):
    N = coords[:, 0]
    Ca = coords[:, 1]
    C = coords[:, 2]
    if not torch.is_tensor(N):
        N = torch.tensor(N)
        Ca = torch.tensor(Ca)
        C = torch.tensor(C)
    frames_rotation = get_batch_rotation_torch(C, Ca, N)
    frames_rotation = frames_rotation.transpose(-1, -2)
    frames_translation = Ca
    r_global_to_local, t_global_to_local = inv_rigid(frames_rotation, frames_translation)

    return r_global_to_local, t_global_to_local


def multi_rigid(rotation1, translation1, rotation2, translation2):
    translation1 = translation1[..., :, None]
    translation2 = translation2[..., :, None]
    translation = rotation1 @ translation2 + translation1
    rotation = rotation1 @ rotation2
    translation = translation.squeeze(-1)
    return rotation, translation


def get_batch_rotation_torch(p1, o, p2):
    v1 = p1 - o
    v2 = p2 - o
    e1 = F.normalize(v1 + 1e-12, p=2, dim=-1)
    e2 = v2 - (v2 * e1).sum(dim=-1, keepdim=True) * e1
    e2 = F.normalize(e2 + 1e-12, p=2, dim=-1)
    e3 = torch.cross(e1, e2)
    e = torch.stack([e1, e2, e3]).permute(1, 2, 0)
    return e


def get_frame_from_coords_batch(coords):
    b = coords.shape[0]
    rs = []
    ts = []
    for i in range(b):
        r, t = get_frame_from_coords(coords[i])
        rs += [r]
        ts += [t]
    r_global_to_local = torch.stack(rs)
    t_global_to_local = torch.stack(ts)
    return r_global_to_local, t_global_to_local
