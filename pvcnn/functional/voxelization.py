from torch.autograd import Function

from pvcnn.functional.backend import _backend

__all__ = ['avg_voxelize']


class AvgVoxelization(Function):
    @staticmethod
    def forward(ctx, features, coords, rx, ry=None, rz=None):
        """
        :param ctx:
        :param features: Features of the point cloud, FloatTensor[B, C, N]
        :param coords: Voxelized Coordinates of each point, IntTensor[B, 3, N]
        :param resolution: Voxel resolution
        :return:
            Voxelized Features, FloatTensor[B, C, rx, ry, rz]
        """
        if ry is None:
            ry = rx
        if rz is None:
            rz = rx

        features = features.contiguous()
        coords = coords.int().contiguous()
        b, c, _ = features.shape
        out, indices, counts = _backend.avg_voxelize_forward(features, coords, rx, ry, rz)
        ctx.save_for_backward(indices, counts)
        return out.view(b, c, rx, ry, rz)

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx:
        :param grad_output: gradient of output, FloatTensor[B, C, R, R, R]
        :return:
            gradient of inputs, FloatTensor[B, C, N]
        """
        b, c = grad_output.shape[:2]
        indices, counts = ctx.saved_tensors
        grad_features = _backend.avg_voxelize_backward(grad_output.contiguous().view(b, c, -1), indices, counts)
        return grad_features, None, None


avg_voxelize = AvgVoxelization.apply
