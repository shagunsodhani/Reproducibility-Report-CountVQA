import torch
from torch.autograd import Variable
from torch.nn import functional as F
from torch import nn


class CountModule(torch.nn.Module):

    def __init__(self, num_proposals):
        super(CountModule, self).__init__()
        self.num_proposals = num_proposals
        self.f = nn.ModuleList([PiecewiseLinear(16) for _ in range(8)])
        self.num_proposals = 10 # consider only top-k objects

    def forward(self, data):
        (boxes, attention) = data
        boxes, attention = self.filter_most_important(self.num_proposals, boxes, attention)

        # print(boxes.size())
        # print(attention.size())

        attention = F.sigmoid(attention)

        # equation 1
        A = self.outer_product(attention)

        # equation 2
        D = 1 - self.iou(boxes, boxes)

        # equation 3
        A_tilde = self.f[0](A) * self.f[1](D)

        # equation 4
        X = self.f[3](A) * self.f[4](D)

        # equation 5
        s_stranspose_inv, s_i_inv = self.deduplicate(X, attention)

        score = A_tilde / s_stranspose_inv

        # equation 6
        correction = self.f[0](attention * attention) / s_i_inv
        mod_E = score.sum(dim=2).sum(dim=1, keepdim=True) + correction.sum(dim=1, keepdim=True)

        # equation 7

        c = (mod_E + 1e-20).sqrt()

        # equation 8
        o = self.to_k_hot(c)
        # o = self.to_k_hot(score)

        # equation 9
        p_a = (self.f[5](attention) - 0.5).abs()

        # equation 10
        p_d = (self.f[6](D) - 0.5).abs()

        # equation 11

        o_tilde = self.f[7](p_a.mean(dim=1, keepdim=True)
                            + p_d.mean(dim=2).mean(dim=1, keepdim=True)) * o

        return o_tilde

    def filter_most_important(self, n, boxes, attention):
        """ Only keep top-n object proposals, scored by attention weight """
        attention, idx = attention.topk(n, dim=1, sorted=False)
        idx = idx.unsqueeze(dim=1).expand(boxes.size(0), boxes.size(1), idx.size(1))
        boxes = boxes.gather(2, idx)
        return boxes, attention

    def to_k_hot(self, c):
        c = c.clamp(min=0, max=self.num_proposals)
        integer_part = c.long().data
        fraction_part = c.frac()
        integer_on_left = c.data.new(c.size(0), self.num_proposals + 1).fill_(0)
        integer_on_left.scatter_(dim=1, index=integer_part.clamp(max=self.num_proposals), value=1)

        integer_on_right = c.data.new(c.size(0), self.num_proposals + 1).fill_(0)
        integer_on_right.scatter_(dim=1, index=(integer_part+1).clamp(max=self.num_proposals), value=1)

        # print(type(integer_on_left))
        # print(type(integer_on_right))
        # print(type(fraction_part))

        return (1-fraction_part)*Variable(integer_on_left) + fraction_part*Variable(integer_on_right)

    def outer_product(self, x):
        # code taken from https://discuss.pytorch.org/t/easy-way-to-compute-outer-products/720/3
        new_size = [*x.size()] + [x.size()[1]]
        return x.unsqueeze(2).expand(*new_size) * x.unsqueeze(-2).expand(*new_size)

    def iou(self, a, b):
        # boxes are assumed to be of the shap[e (batchx4xum_proposals)
        eps = 1e-12
        intersection_area = self.intersection_area(a, b)
        area_a = self.area(a).unsqueeze(2).expand_as(intersection_area)
        area_b = self.area(b).unsqueeze(1).expand_as(intersection_area)
        return intersection_area / (area_a + area_b - intersection_area + eps)

    def area(self, box):
        # Clamping should not be needed, given the way the bb is defined
        height = (box[:, 2, :] - box[:, 0, :]).clamp(min=0)
        width = (box[:, 3, :] - box[:, 1, :]).clamp(min=0)
        return height * width

    def intersection_area(self, a, b):
        size = (a.size(0), 2, a.size(2), a.size(2))
        pairwise_min_point = torch.max(
            a[:, :2, :].unsqueeze(dim=3).expand(*size),
            b[:, :2, :].unsqueeze(dim=2).expand(*size),
        )
        pairwise_max_point = torch.min(
            a[:, 2:, :].unsqueeze(dim=3).expand(*size),
            b[:, 2:, :].unsqueeze(dim=2).expand(*size),
        )
        intersection_points = (pairwise_max_point - pairwise_min_point).clamp(min=0)
        area = intersection_points[:, 0, :, :] * intersection_points[:, 1, :, :]
        return area

    def deduplicate(self, X, att):
        # using outer-diffs
        att_diff = self.outer_diff(att)
        score_diff = self.outer_diff(X)
        sim_ij = self.f[2](1 - att_diff) * self.f[2](1 - score_diff).prod(dim=1)
        # similarity for each row
        s_i = sim_ij.sum(dim=2)
        # similarity for each entry
        s_stranspose = self.outer_product(s_i)
        return s_stranspose, s_i

    def outer_diff(self, x):
        new_size = [*x.size()] + [x.size()[1]]
        return (x.unsqueeze(2).expand(*new_size) - x.unsqueeze(-2).expand(*new_size)).abs()

class PiecewiseLinear(torch.nn.Module):
    def __init__(self, d=16):
        super(PiecewiseLinear, self).__init__()
        self.d = d
        self.weights = self.init_weights()

    def init_weights(self):
        weights = torch.nn.Parameter(torch.ones(self.d + 1))
        weights.data[0] = 0
        return weights

    def forward(self, x):
        weights = self.weights

        # male monotonic
        abs_weights = weights.abs()

        normalized_weights = abs_weights / abs_weights.sum()

        new_size = [self.d + 1] + [1] * x.dim()

        # We want it to be expanded for each dim in x

        expanded_weights = normalized_weights.view(new_size)

        cumulative_sum = expanded_weights.cumsum(dim=0)

        new_size = [self.d + 1] + [*x.size()]

        broadcasted_cumsum = cumulative_sum.expand(new_size)
        broadcasted_weight = expanded_weights.expand_as(broadcasted_cumsum)

        intger_part = Variable(
            (x.unsqueeze(0) * self.d)
                .long()
                .data
        )

        fraction_part = (x.unsqueeze(0) * self.d).frac()

        output = broadcasted_cumsum.gather(0, intger_part.clamp(max=self.d))
        output += fraction_part * broadcasted_weight.gather(0, (intger_part + 1).clamp(max=self.d))

        return output.squeeze(0)
