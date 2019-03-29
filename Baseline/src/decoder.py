
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn


class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, embedding,emotion_embedding, hidden_size, output_size, n_layers=1, dropout=0.1,num_emotions = 7):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.num_emotions = num_emotions
        # Define layers
        self.embedding = embedding
        # define emotion embedding
        self.emotion_embedding = emotion_embedding
        self.embedding_dropout = nn.Dropout(dropout)
        #self.emotion_embedding_dropout = nn.Dropout(dropout)
        # dimension
        self.gru = nn.GRU(hidden_size + hidden_size + hidden_size + hidden_size , hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.attn = Attn(attn_model, hidden_size)
        self.internal_memory = ECMWrapper(hidden_size,hidden_size,
                                          hidden_size,self.num_emotions,
                                          self.embedding,self.emotion_embedding,self.gru)
    def forward(self, input_step,input_step_emotion, last_hidden
                ,input_context, last_int_memory,encoder_outputs):
        '''
        First input_context will be a random vectors
        '''
        rnn_output, hidden, new_M_emo = self.internal_memory(input_step,input_step_emotion,
                                                            last_hidden,input_context,
                                                             last_int_memory)
        # Calculate attention weights from the current GRU output
        attn_weights = self.attn(rnn_output, encoder_outputs)
        # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # Concatenate weighted context vector and GRU output using Luong eq. 5
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        # Predict next word using Luong eq. 6
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)
        # Return output and final hidden state
        return output, hidden, new_M_emo, context