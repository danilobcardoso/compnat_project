import torch
import torch.nn.functional as F
import torch.nn as nn


class ST_GCN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.t_kernel_size = 5
        self.t_padding = 2
        self.conv = nn.Conv2d(in_channels,
                              out_channels * kernel_size,
                              kernel_size=(self.t_kernel_size, 1),
                              padding=(self.t_padding, 0),
                              stride=(1, 1),
                              dilation=(1, 1),
                              bias=True)

    def forward(self, x, A):
        assert A.size(0) == self.kernel_size

        x = self.conv(x)

        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))

        return x.contiguous(), A