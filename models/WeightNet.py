# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch.nn as nn
import torch.nn.functional as F
import torch


class WNet(nn.Module):
    def __init__(self, input=1, hidden=100):
        super(WNet, self).__init__()
        self.linear1 = nn.Linear(input, hidden)
        self.gelu1 = nn.GELU()
        self.linear2 = nn.Linear(hidden, 1)
        self.linear_o = nn.Linear(1, 1, bias=False)
        self.linear_g = nn.Linear(1, 1, bias=False)

    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu1(x)
        x = self.linear2(x)
        self.linear_o.weight.data = torch.clip(self.linear_o.weight.data, min=1e-3)
        omega = F.sigmoid(self.linear_o(x))
        self.linear_g.weight.data = torch.clip(self.linear_g.weight.data, max=-1e-3)
        gamma = F.sigmoid(self.linear_g(x))
        return omega, gamma
