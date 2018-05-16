import os
import sys

import torch
import torch.nn as nn
from torch.autograd import Variable

from attention import Attention, NewAttention
from classifier import ClassifierConfig
from classifier import ModifiedClassifier as Classifier
from language_model import WordEmbedding, QuestionEmbedding, WordEmbeddingConfig, QuestionEmbeddingConfig
from Counter import CountModule

sys.path.append(os.getcwd())



class BaseModel(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, counter, classifier, v_dim):
        super(BaseModel, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.counter = counter
        self.classifier = classifier
        self.v_dim = v_dim
        self.init_weights()

    def init_weights(self):
        for m in self.named_parameters():
            if 'v_att' in m[0] or 'classifier' in m[0]:
                print(m[0])
                if 'bnc' in m[0] or 'bn2' in m[0]:
                    if 'weight' in m[0]:
                        nn.init.constant(m[1], 1)
                    elif 'bias' in m[0]:
                        nn.init.constant(m[1], 0)
                else:
                    if 'weight' in m[0]:
                        torch.nn.init.xavier_uniform(m[1])
                    elif 'bias' in m[0]:
                        m[1].data.zero_()

    def forward(self, v, b, q, q_len, labels):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        # Changing this logic would require a change in the dataloader
        # len_q = [14]*q.shape[0]
        # print(len_q)
        
        # w_emb = self.w_emb(q)

        w_emb = nn.utils.rnn.pack_padded_sequence(self.w_emb(q), lengths=list(q_len.data), batch_first=True)
        q_emb = self.q_emb(w_emb)  # [batch, q_dim]

        v = v / (v.norm(p=2, dim=1, keepdim=True) + 1e-12).expand_as(v)
        
        v, att = self.v_att(v, q_emb)
        att_c = att[:, 0, :, :].contiguous().view(att.size(0), -1)

        count_feat = self.counter((b, att_c))
        # count_feat = Variable(torch.cuda.FloatTensor(v.size(0), 11).fill_(0))
        # print(count_feat.size())

        logits = self.classifier((v, q_emb, count_feat))

        return logits


def build_baseline0(n_dict_token, hid_dim, v_dim, n_objects, max_answers, n_glimpse):
    w_emb = build_wordembedding_model(n_dict_token)
    q_emb = build_questionembedding_model(hid_dim)
    v_att = Attention(v_dim, q_emb.hidden_dim, hid_dim)
    counter = CountModule(num_proposals=n_objects)

    classifier = build_classifier_model(input_dim_v = v_dim * n_glimpse, input_dim_q = q_emb.hidden_dim,
                                        input_dim_c = n_objects+1,
                                        num_candidate_ans=max_answers)
    return BaseModel(w_emb, q_emb, v_att, counter, classifier, v_dim)


def build_wordembedding_model(n_dict_token):
    wordembedding_config = WordEmbeddingConfig(n_dict_token, 300, 0.0)
    w_emb = WordEmbedding(wordembedding_config)
    return w_emb


def build_questionembedding_model(num_hid):
    questionembedding_config = QuestionEmbeddingConfig(
        input_dim=300,
        hidden_dim=num_hid,
        num_layers=1,
        is_bidirect=False,
        dropout_prob=0.5,
        rnn_type="GRU"
    )
    return QuestionEmbedding(questionembedding_config)

def build_classifier_model(input_dim_v, input_dim_q, input_dim_c, num_candidate_ans):

    classifier_config = ClassifierConfig(
        input_dim_v=input_dim_v,
        input_dim_q=input_dim_q,
        input_dim_c=input_dim_c,
        mid_dim=1024,
        output_dim=num_candidate_ans,
        dropout_prob=0.5
    )
    # print(classifier_config)
    return Classifier(classifier_config)
