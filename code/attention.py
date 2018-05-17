import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm

class Attention(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid=512, n_glimpse=2, p_drop=0.5):
        super(Attention, self).__init__()
        self.v_dim = v_dim
        self.q_dim = q_dim
        self.num_hid = num_hid
        self.n_glimpse = n_glimpse
        self.p_drop = p_drop

        self.conv_1 = nn.Conv2d(self.v_dim, self.num_hid, 1, bias=False)
        self.ques_linear = nn.Linear(self.q_dim, self.num_hid)

        self.conv_2 = nn.Conv2d(self.num_hid, self.n_glimpse, 1)
        self.dropout = nn.Dropout(self.p_drop)

    def forward(self, v, q):
        """
        v: [batch, n_objects, v_dim]
        q: [batch, q_dim]
        """
        # print(v.size())
        # v = v.permute(0, 2, 1) # [batch, v_dim, n_objects]
        # print(v.size())
        # v = v.unsqueeze(2) # [batch, v_dim, 1, n_objects]
        # print(v.size())
        v_a = self.conv_1(self.dropout(v)) # [batch, num_hid, 1, n_objects]

        n_objects = v.size(3)

        q = self.ques_linear(q) # [batch, num_hid]

        q = q.unsqueeze(-1).unsqueeze(-1) # [batch, num_hid, 1, 1]
        q = q.expand(v.size(0), q.size(1), q.size(2), n_objects) # [batch, num_hid, 1, n_objects]

        # x = torch.cat((v_a, q), dim=1) # [batch, (num_hid + num_hid), 1, n_objects]
        x = F.relu(v_a + q) - (v_a - q)**2 # [batch, num_hid, 1, n_objects]

        x = self.conv_2(self.dropout(x)) # [batch, n_glimpse, 1, n_objects]
        attn_op = x # to be used by count module

        x = x.view(-1, x.size(3)) # [batch * n_glimpse, n_objects]
        attn = F.softmax(x, dim=1) # [batch * n_glimpse, n_objects]

        attn = attn.view(v.size(0), self.n_glimpse, 1, n_objects).expand(v.size(0), self.n_glimpse, self.v_dim, n_objects) # [batch, n_glimpse, v_dim, n_objects]

        v.contiguous()
        v = v.view(v.size(0), 1, self.v_dim, n_objects).expand(v.size(0), self.n_glimpse, self.v_dim, n_objects) # [batch, n_glimpse, v_dim, n_objects]
        
        v_weighted = v * attn # [batch, n_glimpse, v_dim, n_objects]
        v_weighted = v_weighted.sum(dim=3) # [batch, n_glimpse, v_dim]
        v_weighted = v_weighted.view(v.size(0), -1)  # [batch, (n_glimpse * v_dim)]

        return v_weighted, attn_op


class NewAttention(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid, dropout=0.2):
        super(NewAttention, self).__init__()

        self.v_proj = FCNet([v_dim, num_hid])
        self.q_proj = FCNet([q_dim, num_hid])
        self.dropout = nn.Dropout(dropout)
        self.linear = weight_norm(nn.Linear(q_dim, 1), dim=None)

    def forward(self, v, q):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        logits = self.logits(v, q)
        w = nn.functional.softmax(logits, 1)
        return w

    def logits(self, v, q):
        batch, k, _ = v.size()
        v_proj = self.v_proj(v) # [batch, k, qdim]
        q_proj = self.q_proj(q).unsqueeze(1).repeat(1, k, 1)
        joint_repr = v_proj * q_proj
        joint_repr = self.dropout(joint_repr)
        logits = self.linear(joint_repr)
        return logits
