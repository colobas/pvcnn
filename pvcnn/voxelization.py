import torch
import torch.nn as nn

import pvcnn.functional as F

__all__ = ['Voxelization']


class Voxelization(nn.Module):
    def __init__(self, rx, ry=1, rz=1, normalize=True, eps=0):
        super().__init__()
        self.rx = rx
        self.ry = ry
        self.rz = rz
        self.normalize = normalize
        self.eps = eps

    def forward(self, features, coords):
        coords = coords.detach()
        norm_coords = coords - coords.mean(2, keepdim=True)
        if self.normalize:
            norm_coords = norm_coords / (norm_coords.norm(dim=1, keepdim=True).max(dim=2, keepdim=True).values * 2.0 + self.eps) + 0.5
        else:
            norm_coords = (norm_coords + 1) / 2.0
        breakpoint()

        norm_coords[0] = torch.clamp(norm_coords[0] * self.rx, 0, self.rx - 1)
        norm_coords[1] = torch.clamp(norm_coords[1] * self.ry, 0, self.ry - 1)
        norm_coords[2] = torch.clamp(norm_coords[2] * self.rz, 0, self.rz - 1)

        vox_coords = torch.round(norm_coords).to(torch.int32)
        return F.avg_voxelize(features, vox_coords, self.rx, self.ry, self.rz), norm_coords

    def extra_repr(self):
        s = f"resolution={(self.rx, self.ry, self.rz)}"
        if self.normalize:
            s += f", normalized eps = {self.eps}"
        return s
