from collections import namedtuple

import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
from fusion import Fusion


class SimpleClassifier(nn.Module):
    # Not using this any more
    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super(SimpleClassifier, self).__init__()
        layers = [
            weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
            nn.ReLU(),
            nn.Dropout(dropout, inplace=True),
            weight_norm(nn.Linear(hid_dim, out_dim), dim=None)
        ]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.main(x)
        return logits


class ModifiedClassifier(nn.Module):
    def __init__(self, classifier_config):
        super(ModifiedClassifier, self).__init__()
        self.dropout = nn.Dropout(classifier_config.dropout_prob)
        self.relu = nn.ReLU()
        self.fusion_op = Fusion()
        self.wv = nn.Linear(classifier_config.input_dim_v, classifier_config.mid_dim)
        self.wq = nn.Linear(classifier_config.input_dim_q, classifier_config.mid_dim)
        self.w = nn.Linear(classifier_config.mid_dim, classifier_config.output_dim)
        self.wc = nn.Linear(classifier_config.input_dim_c, classifier_config.mid_dim)
        self.bnc = nn.BatchNorm1d(classifier_config.mid_dim)
        self.bn2 = nn.BatchNorm1d(classifier_config.mid_dim)

    def forward(self, x):
        (v, q, c) = x
        fused_features = self.fusion_op(
            (
            self.wv(
                self.dropout(
                    v
                )
            ),
            self.wq(
                self.dropout(
                    q
                )
            )
            )
        )
        count_features = self.bnc(self.relu(
            self.wc(c)
        ))
        combined_features = fused_features + count_features
        return self.w(
            self.dropout(
                self.bn2(
                    combined_features
                )
            )
        )


ClassifierConfig = namedtuple("ClassifierConfig",
                              ["input_dim_v", "input_dim_q", "input_dim_c",
                               "mid_dim", "output_dim", "dropout_prob"])
