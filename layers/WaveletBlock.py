import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleWaveletBlock(nn.Module):
    """
    True multi-level Haar DWT decomposition.
    Returns full-resolution bands for fusion:
      [D1_up, D2_up, ..., Dk_up, A_k_up]
    """

    def __init__(self, levels=2, dropout=0.1):
        super().__init__()
        self.levels = max(1, int(levels))
        self.dropout = nn.Dropout(dropout)
        coeff = 1.0 / (2.0 ** 0.5)
        self.register_buffer("haar_low", torch.tensor([coeff, coeff], dtype=torch.float32).view(1, 1, 2))
        self.register_buffer("haar_high", torch.tensor([coeff, -coeff], dtype=torch.float32).view(1, 1, 2))

    def _dwt_step(self, x):
        # x: [B, C, T]
        if x.shape[-1] % 2 == 1:
            x = F.pad(x, (0, 1), mode='replicate')
        c = x.shape[1]
        low_filter = self.haar_low.repeat(c, 1, 1).to(dtype=x.dtype, device=x.device)
        high_filter = self.haar_high.repeat(c, 1, 1).to(dtype=x.dtype, device=x.device)
        low = F.conv1d(x, low_filter, stride=2, groups=c)
        high = F.conv1d(x, high_filter, stride=2, groups=c)
        return low, high

    def decompose(self, x):
        """
        x: [B, T, C]
        returns (approx, details) where:
          approx: [B, C, T/2^k]
          details: list of [B, C, T/2^i] from i=1..k
        """
        cur = x.transpose(1, 2).contiguous()  # [B, C, T]
        details = []
        for _ in range(self.levels):
            cur, detail = self._dwt_step(cur)
            details.append(detail)
        return cur, details

    def forward(self, x):
        """
        x: [B, T, C]
        returns full-resolution stacked bands: [B, T, C, levels+1]
        """
        t = x.shape[1]
        approx, details = self.decompose(x)
        bands = []
        for detail in details:
            up = F.interpolate(detail, size=t, mode='linear', align_corners=False)  # [B,C,T]
            bands.append(up)
        approx_up = F.interpolate(approx, size=t, mode='linear', align_corners=False)
        bands.append(approx_up)
        stacked = torch.stack(bands, dim=-1).permute(0, 2, 1, 3).contiguous()  # [B,T,C,num_bands]
        return self.dropout(stacked)
