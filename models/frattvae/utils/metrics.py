import torch
import torch.nn as nn
import torch.nn.functional as F


def batched_kl_divergence(mu: torch.Tensor, ln_var: torch.Tensor):
    return (-0.5 * torch.sum(1 + ln_var - mu.pow(2) - ln_var.exp(), dim=-1)).mean(dim=0)


def cosine_matrix(a: torch.Tensor, b: torch.Tensor, p: int = 2, dim: int = 1):
    return torch.matmul(F.normalize(a, p=p, dim=dim), F.normalize(b, p=p, dim=dim).T)


def cosine_similarity(a: torch.Tensor, b: torch.Tensor, p: int = 2, dim: int = 1):
    return cosine_matrix(a, b, p, dim).diag()


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        loss = torch.sqrt(self.mse(input, target) + self.eps)
        return loss


CRITERION = {
    'mse': nn.MSELoss,
    'mae': nn.L1Loss,
    'bce': nn.BCEWithLogitsLoss,
    'crs': nn.CrossEntropyLoss,
    'rmse': RMSELoss,
}


def euclid_distance(a: torch.Tensor, b: torch.Tensor, p: int = 2, dim: int = 1):
    return torch.norm(a - b, p=p, dim=dim)
