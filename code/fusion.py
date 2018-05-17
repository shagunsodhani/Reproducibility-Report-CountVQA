import torch
import torch.nn.functional as F

class Fusion(torch.nn.Module):
    def __init__(self):
        super(Fusion, self).__init__()

    def forward(self, data):
        v, q = data
        return F.relu(v+q) - torch.pow((v-q), 2)
