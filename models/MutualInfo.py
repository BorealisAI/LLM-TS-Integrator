# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn


class NodeEmb(nn.Module):
    def __init__(self, input_emb_size=4096, node_emb_size=64):
        super(NodeEmb, self).__init__()
        self.L1 = nn.Linear(input_emb_size, node_emb_size)

    def forward(self, node_feat):
        return self.L1(node_feat)


class MutualInfo(nn.Module):
    def __init__(self, input_emb_size=64, llm_emb_size=4096):
        super(MutualInfo, self).__init__()
        self.llm_emb_layer = NodeEmb(input_emb_size=llm_emb_size)
        self.num_emb_layer = NodeEmb(input_emb_size=input_emb_size)
        self.softplus = nn.Softplus()

    def forward(self, llm_emb, num_emb, prob_mutual=None):
        if prob_mutual is None:
            prob_mutual = torch.ones(llm_emb.shape[0]) / llm_emb.shape[0]
            prob_mutual = prob_mutual.to(llm_emb.device)
        llm_emb = self.llm_emb_layer(llm_emb)
        num_emb = self.num_emb_layer(num_emb)
        distance = torch.mm(llm_emb, num_emb.t())
        prob_ij = torch.mm(prob_mutual.reshape(-1, 1), prob_mutual.reshape(1, -1))
        diag = torch.diag(distance)
        diag_loss = torch.sum(prob_mutual.squeeze() * self.softplus(-diag))
        undiag = (
            distance.flatten()[:-1]
            .view(distance.shape[0] - 1, distance.shape[0] + 1)[:, 1:]
            .flatten()
        )
        prob_ij_undiag = (
            prob_ij.flatten()[:-1]
            .view(prob_ij.shape[0] - 1, prob_ij.shape[0] + 1)[:, 1:]
            .flatten()
        )
        prob_ij_undiag = prob_ij_undiag / torch.sum(prob_ij_undiag)
        undiag_loss = torch.sum(self.softplus(undiag) * prob_ij_undiag)
        loss = diag_loss + undiag_loss
        mutualinfo = -loss
        return mutualinfo
