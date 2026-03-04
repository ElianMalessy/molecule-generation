import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer('mask', torch.ones(out_features, in_features))

    def set_mask(self, mask):
        self.mask.data.copy_(mask)

    def forward(self, x):
        return F.linear(x, self.weight * self.mask, self.bias)


class MADE(nn.Module):
    """2-layer MADE that outputs (mean, log_scale) autoregressively."""

    def __init__(self, dim: int, hidden_dim: int = 256):
        super().__init__()
        self.lin1 = MaskedLinear(dim, hidden_dim)
        self.lin2 = MaskedLinear(hidden_dim, dim * 2)

        input_order  = torch.arange(1, dim + 1)
        hidden_order = torch.arange(hidden_dim) % max(dim - 1, 1) + 1
        output_order = input_order.repeat(2)

        mask1 = (hidden_order.unsqueeze(1) >= input_order.unsqueeze(0)).float()
        mask2 = (output_order.unsqueeze(1) > hidden_order.unsqueeze(0)).float()
        self.lin1.set_mask(mask1)
        self.lin2.set_mask(mask2)

    def forward(self, z):
        h = F.relu(self.lin1(z))
        m, s = self.lin2(h).chunk(2, dim=-1)
        return m, s


class IAFStep(nn.Module):
    def __init__(self, dim: int, hidden_dim: int = 256):
        super().__init__()
        self.made = MADE(dim, hidden_dim)

    def forward(self, z):
        m, s = self.made(z)
        gate = torch.sigmoid(s)
        z_new = gate * z + (1.0 - gate) * m
        log_det = torch.log(gate + 1e-8).sum(dim=-1)
        return z_new, log_det


class InverseAutoregressiveFlow(nn.Module):
    """Stacked IAF steps that map z0 → zK with tractable log-determinant.

    The prior is p(zK) = N(0, I).  During sampling, draw z ~ N(0, I) and
    decode directly — do NOT pass through the flow.
    """

    def __init__(self, dim: int, num_flows: int = 4, hidden_dim: int = 256):
        super().__init__()
        self.flows = nn.ModuleList(
            [IAFStep(dim, hidden_dim) for _ in range(num_flows)]
        )

    def forward(self, z: torch.Tensor):
        """Returns (zK, sum_log_det) where sum_log_det is (B,)."""
        sum_log_det = torch.zeros(z.size(0), device=z.device, dtype=z.dtype)
        for flow in self.flows:
            z, ld = flow(z)
            sum_log_det = sum_log_det + ld
        return z, sum_log_det
