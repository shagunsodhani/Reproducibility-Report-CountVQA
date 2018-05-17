from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


class WordEmbedding(nn.Module):
    """Word Embedding Model
    """

    def __init__(self, wordembedding_config):
        super(WordEmbedding, self).__init__()
        self.emb = nn.Embedding(wordembedding_config.padding_idx + 1, wordembedding_config.emb_dim,
                                padding_idx=wordembedding_config.padding_idx)
        self.dropout = nn.Dropout(wordembedding_config.dropout_prob)
        self.ntoken = wordembedding_config.padding_idx
        self.emb_dim = wordembedding_config.emb_dim
        self.tanh = nn.Tanh()

    def init_embedding(self, np_file):
        weight_init = torch.from_numpy(np.load(np_file))
        assert weight_init.shape == (self.ntoken, self.emb_dim)
        self.emb.weight.data[:self.ntoken] = weight_init

    def forward(self, x):
        emb = self.emb(x)
        emb = self.dropout(emb)
        emb = self.tanh(emb)
        return emb


class QuestionEmbedding(nn.Module):
    def __init__(self, questionembedding_config):
        """Module for question embedding
        """
        super(QuestionEmbedding, self).__init__()
        assert questionembedding_config.rnn_type == 'LSTM' or questionembedding_config.rnn_type == 'GRU'
        rnn_cls = nn.LSTM if questionembedding_config.rnn_type == 'LSTM' else nn.GRU

        self.rnn = rnn_cls(
            input_size=questionembedding_config.input_dim,
            hidden_size=questionembedding_config.hidden_dim,
            num_layers=questionembedding_config.num_layers,
            bidirectional=questionembedding_config.is_bidirect,
            dropout=questionembedding_config.dropout_prob,
            batch_first=True)
        
        self._init_rnn(self.rnn.weight_ih_l0)
        self._init_rnn(self.rnn.weight_hh_l0)
        self.rnn.bias_ih_l0.data.zero_()
        self.rnn.bias_hh_l0.data.zero_()
                
        self.config = questionembedding_config


    def _init_rnn(self, weight):
        for w in weight.chunk(4, 0):
            nn.init.xavier_uniform(w)

    @property
    def hidden_dim(self):
        return self.config.hidden_dim

    def init_hidden(self, batch):
        # just to get the type of tensor
        weight = next(self.parameters()).data
        num_directions = int(self.config.is_bidirect) + 1
        hid_shape = (self.config.num_layers * num_directions, batch, self.config.hidden_dim)
        if(self.config.rnn_type == "LSTM"):
            return (Variable(weight.new(*hid_shape).zero_()),
                    Variable(weight.new(*hid_shape).zero_()))
        elif(self.config.rnn_type == "GRU"):
            return Variable(weight.new(*hid_shape).zero_())
        return None

    def forward(self, x):
        # x: [batch, sequence, in_dim]
        # batch = x.size(0)
        # hidden = self.init_hidden(batch)
        self.rnn.flatten_parameters()

        _, c = self.rnn(x)
        # _, (_, c) = self.rnn(x)
        # z, y = self.rnn(x)
        # print(type(z.data))
        # print(type(y.data))
        return c.squeeze(0)

        # commenting out this code for now
        # forward_ = output[:, -1, :self.config.hidden_dim]
        # backward_ = output[:, 0, self.config.hidden_dim:]
        # return torch.cat((forward_, backward_), dim=1)


WordEmbeddingConfig = namedtuple("WordEmbeddingConfig", ["padding_idx", "emb_dim", "dropout_prob"])

QuestionEmbeddingConfig = namedtuple("QuestionEmbeddingConfig", ["input_dim", "hidden_dim", "num_layers", "is_bidirect",
                                                                 "dropout_prob", "rnn_type"])
