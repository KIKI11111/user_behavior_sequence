import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ApiidReconstructionKLDivergenceLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.KL_loss = nn.KLDivLoss(reduction='batchmean')

    def forward(self, apiid_output, label_apiid_dist):
        return self.KL_loss(F.log_softmax(apiid_output, dim=1), label_apiid_dist)


class IntervalReconstructionKLDivergenceLoss(nn.Module):

    def __init__(self):
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')

    def forward(self, interval_output, label_interval_dist):
        return self.Kl_loss(F.log_softmax(interval_output, dim=1), label_interval_dist)


class GaussianNllLoss(nn.Module):

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, mu, sigma, target):
        # 确保sigma非常小的值不会引发问题
        sigma = torch.clamp(sigma, min=self.eps)
        loss = 0.5*torch.log(2*math.pi*sigma**2) + (target - mu)**2/(2*sigma**2)

        return torch.mean(loss)


class IntervalReconstructionGaussianNllLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.nll_loss = GaussianNllLoss()

    def forward(self, interval_output, target):
        mu, sigma = interval_output[:, 0], interval_output[:,1]
        return self.nll_loss(mu, sigma, target)


class Wasserstein1DLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.epsilon = 1e-14

    def forward(self, x, target):
        x = x/(torch.sum(x, dim=-1, keepdim=True) + self.epsilon)
        target = target/(torch.sum(target, dim=-1, kppedim=True) + self.epsilon)

        cdf_x = torch.cumsum(x, dim=-1)
        cdf_target = torch.cumsum(target, dim=-1)
        cdf_distance = torch.sum(torch.abs((cdf_x - cdf_target)), dim=-1)

        return cdf_distance.mean()


class IntervalReconstructionWassersteinLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.wassersteinLoss = Wasserstein1DLoss()

    def forward(self, interval_output, target):
        return self.wassersteinLoss(interval_output, target)


