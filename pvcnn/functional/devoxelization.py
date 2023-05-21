from torch.autograd import Function

from pvcnn.functional.backend import _backend

__all__ = ['trilinear_devoxelize']


class TrilinearDevoxelization(Function):
    @staticmethod
    def forward(ctx, features, coords, rx, ry=None, rz=None, is_training=True):
        """
        :param ctx:
        :param coords: the coordinates of points, FloatTensor[B, 3, N]
        :param features: FloatTensor[B, C, Rx, Ry, Rz]
        :param resolution: int, the voxel resolution
        :param is_training: bool, training mode
        :return:
            FloatTensor[B, C, N]
        """
        if ry is None:
            ry = rx
        if rz is None:
            rz = rx

        B, C = features.shape[:2]
        features = features.contiguous().view(B, C, -1)
        coords = coords.contiguous()
        outs, inds, wgts = _backend.trilinear_devoxelize_forward(rx, ry, rz, is_training, coords, features)
        if is_training:
            ctx.save_for_backward(inds, wgts)
            ctx.rx = rx
            ctx.ry = ry
            ctx.rz = rz
        return outs

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx: 
        :param grad_output: gradient of outputs, FloatTensor[B, C, N]
        :return:
            gradient of inputs, FloatTensor[B, C, R, R, R]
        """
        inds, wgts = ctx.saved_tensors
        grad_inputs = _backend.trilinear_devoxelize_backward(
            grad_output.contiguous(), inds, wgts, ctx.rx, ctx.ry, ctx.rz)
        return (
            grad_inputs.view(grad_output.size(0), grad_output.size(1),
            ctx.rx, ctx.ry, ctx.rz), None, None, None
        )


trilinear_devoxelize = TrilinearDevoxelization.apply
