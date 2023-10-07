import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseLoss:

    def __init__(self, get_params, lam, lam0=None, crop=0, detach_target=False):
        self._get_params = get_params
        self._lam = lam
        self._lam0 = lam0
        self._crop = crop
        self._detach_target = detach_target

    def __call__(self, layer_outputs):
        # Layer losses
        layer_loss = 0
        for i, (output, target, hidden) in enumerate(layer_outputs):
            if i == 0 and self._crop > 0:
                output = output[:, :, :, self._crop:-self._crop, self._crop:-self._crop]
                target = target[:, :, :, self._crop:-self._crop, self._crop:-self._crop]

            layer_loss = layer_loss + self.layer_loss(output, target.detach() if self._detach_target else target, hidden)

        # Regulirisation losses
        reg_loss = 0
        for i, param in enumerate(self._get_params()):
            lam = self._lam
            if i == 0 and self._lam0 is not None:
                lam = self._lam0
            reg_loss = reg_loss + lam * torch.norm(param, p=1)

        total_loss = layer_loss + reg_loss

        return total_loss

    def layer_loss(self, output, target, hidden, lam):
        raise NotImplementedError


class PredictionLoss(BaseLoss):

    def __init__(self, get_params, lam, lam0=None, crop=0, detach_target=False, pred_steps=1):
        super().__init__(get_params, lam, lam0, crop, detach_target)
        self._pred_steps = pred_steps

    def layer_loss(self, output, target, hidden):
        t_offset = target.shape[2] - output.shape[2]

        return F.mse_loss(output[:, :, :-self._pred_steps], target[:, :, t_offset+self._pred_steps:])


class SparseCompressionLoss(BaseLoss):

    def __init__(self, get_params, lam, lam0=None, crop=0, detach_target=False, lam_activity=10**-7):
        super().__init__(get_params, lam, lam0, crop, detach_target)
        self._lam_activity = lam_activity

    def layer_loss(self, output, target, hidden):
        t_offset = target.shape[2] - output.shape[2]

        return F.mse_loss(output, target[:, :, t_offset:]) + self._lam_activity * torch.norm(hidden, p=1)


class SlownessLoss(BaseLoss):

    def __init__(self, get_params, lam, lam0=None, crop=0, detach_target=False):
        super().__init__(get_params, lam, lam0, crop, detach_target)
        self._norms = {}

    def layer_loss(self, output, target, hidden):
        if self._norms.get(hidden.shape[1]) is None:
            self._norms[hidden.shape[1]] = nn.BatchNorm3d(hidden.shape[1], affine=False).cuda()
        hidden = self._norms.get(hidden.shape[1])(hidden)
        flatten_hidden = hidden.permute(1, 2, 3, 4, 0)
        flatten_hidden = flatten_hidden.flatten(1, 4)

        slowness = F.mse_loss(hidden[:, :, :-1], hidden[:, :, 1:])
        noise = 0.0001 * torch.rand(flatten_hidden.shape).cuda()
        decorelation = torch.corrcoef(flatten_hidden+noise)
        ii, jj = torch.triu_indices(decorelation.shape[0], decorelation.shape[0], offset=1)
        decorelation = torch.nan_to_num(decorelation[ii, jj].abs(), nan=0.0, posinf=0, neginf=0).mean()

        return slowness + decorelation
