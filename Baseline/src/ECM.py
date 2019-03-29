
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn


class ECMWrapper(nn.Module):
    def __init__(self, hidden_size, state_size, emo_size, num_emotion, embedding, emotion_embedding, gru):
        '''
        hidden_size: hidden input dimension
        state_size: state vector size
        emo_size: emotional embedding size
        num_emotion: number of emotion categories
        '''
        super(ECMWrapper, self).__init__()
        self.hidden_size = hidden_size
        self.state_size = state_size
        self.emo_size = emo_size
        self.num_emotion = num_emotion
        # read gate dimensions (word_embedding + hidden_input + context_input)
        self.read_g = nn.Linear(self.hidden_size + self.hidden_size + self.hidden_size, self.emo_size)
        # write gate
        self.write_g = nn.Linear(self.state_size, self.emo_size)
        # GRU output input dimensions = state_last + context + emotion emb + internal memory
        self.gru = gru
        self.emotion_embedding = emotion_embedding
        self.embedding = embedding

    def forward(self, word_input, emotion_input, last_hidden, context_input, M_emo):
        '''
        Last hidden == prev_cell_state
        last word embedding = word_input
        last hidden input = h
        '''
        # get embedding of input word and emotion
        context_input = context_input.unsqueeze(dim=0)
        emo_embedding = self.emotion_embedding(emotion_input)
        emo_embedding = emo_embedding.unsqueeze(dim=0)
        last_word_embedding = self.embedding(word_input)
        # sum bidirectional hidden input
        last_hidden_sum = torch.sum(last_hidden, dim=0).unsqueeze(dim=0)
        read_inputs = torch.cat((last_word_embedding, last_hidden_sum, context_input), dim=-1)
        # compute read input
        read_inputs = self.read_g(read_inputs)
        M_read = torch.sigmoid(read_inputs)
        # write to emotion embedding
        emo_embedding = emo_embedding * M_read
        # pass everything to GRU
        X = torch.cat([last_word_embedding, last_hidden_sum, context_input, emo_embedding], dim=-1)
        rnn_output, hidden = self.gru(X, last_hidden)
        # write input
        M_write = torch.sigmoid(self.write_g(rnn_output))
        # write to emotion embedding
        new_M_emo = emo_embedding * M_write
        return rnn_output, hidden, new_M_emo
