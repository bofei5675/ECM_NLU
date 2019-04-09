from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import pandas as pd
import csv
import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools
import math
import time
import json


class EncoderRNN(nn.Module):
    '''
    Regular encoder from torch tutorial
    '''
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        # Initialize GRU; the input_size and hidden_size params are both set to 'hidden_size'
        #   because our input size is a word embedding with number of features == hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=True)

    def forward(self, input_seq, input_lengths, hidden=None):
        # Convert word indexes to embeddings
        embedded = self.embedding(input_seq)
        # Pack padded batch of sequences for RNN module
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        # Forward pass through GRU
        outputs, hidden = self.gru(packed, hidden)
        # Unpack padding
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        # Sum bidirectional GRU outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]
        # Return output and final hidden state
        return outputs, hidden
# Luong attention layer
class Attn(torch.nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = torch.nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = torch.nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = torch.nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # Transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t()

        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)


class ECMWrapper(nn.Module):
    '''
    This module is for internal memory
    '''
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

    def forward(self, word_input, emotion_input, last_hidden, context_input):
        '''
        Last hidden == prev_cell_state
        last word embedding = word_input
        last hidden input = h
        '''
        # get embedding of input word and emotion
        context_input = context_input.unsqueeze(dim=0)
        last_word_embedding = self.embedding(word_input)
        # sum bidirectional hidden input
        last_hidden_sum = torch.sum(last_hidden, dim=0).unsqueeze(dim=0)
        read_inputs = torch.cat((last_word_embedding, last_hidden_sum, context_input), dim=-1)
        # compute read input
        read_inputs = self.read_g(read_inputs)
        M_read = torch.sigmoid(read_inputs)
        # write to emotion embedding
        emotion_input = emotion_input * M_read
        # pass everything to GRU
        X = torch.cat([last_word_embedding, last_hidden_sum, context_input, emotion_input], dim=-1)
        rnn_output, hidden = self.gru(X, last_hidden)
        # write input
        M_write = torch.sigmoid(self.write_g(rnn_output))
        # write to emotion embedding
        new_M_emo = emotion_input * M_write
        return rnn_output, hidden, new_M_emo


class LuongAttnDecoderRNN(nn.Module):
    '''
    Oritinal from torch tutorial
    Decode modified to contain external memory and internal memory module
    '''
    def __init__(self, attn_model, embedding, emotion_embedding, hidden_size, output_size, ememory=None, n_layers=1,
                 dropout=0.1, num_emotions=7):
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
        # self.emotion_embedding_dropout = nn.Dropout(dropout)
        # dimension
        self.gru = nn.GRU(hidden_size + hidden_size + hidden_size + hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.attn = Attn(attn_model, hidden_size)
        self.internal_memory = ECMWrapper(hidden_size, hidden_size,
                                          hidden_size, self.num_emotions,
                                          self.embedding, self.emotion_embedding, self.gru)
        # read external from outside
        self.external_memory = ememory
        # emotional output linear layer
        self.emotion_word_output_layer = nn.Linear(self.hidden_size, output_size)
        # emotional gate/ choice layer
        self.alpha_layer = nn.Linear(output_size, 1)

    def forward(self, input_step, input_step_emotion, last_hidden
                , input_context, encoder_outputs):
        '''
        First input_context will be a random vectors
        '''
        if not torch.is_floating_point(input_step_emotion):
            input_step_emotion = self.emotion_embedding(input_step_emotion)
        rnn_output, hidden, new_M_emo = self.internal_memory(input_step, input_step_emotion,
                                                             last_hidden, input_context)
        # Calculate attention weights from the current GRU output
        attn_weights = self.attn(rnn_output, encoder_outputs)
        # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # Concatenate weighted context vector and GRU output using Luong eq. 5
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        if self.external_memory is not None:
            # Predict next word using Luong eq. 6
            output = self.out(concat_output)
            # external memory gate
            g = torch.sigmoid(self.alpha_layer(output))
            # splice tensor based on ememory
            output_e = output[:, self.external_memory == 1]
            output_g = output[:, self.external_memory == 0]
            # get indices of emotion word and genric word
            idx_e = (self.external_memory == 1).nonzero().reshape(-1)
            idx_g = (self.external_memory == 0).nonzero().reshape(-1)
            # compute softmax output
            output_e = F.softmax(output_e, dim=1) * (g)
            output_g = F.softmax(output_g, dim=1) * (1 - g)
            output = torch.cat((output_e, output_g), dim=1)
            idx = torch.cat((idx_e, idx_g), dim=0)
            idx_sort, _ = torch.sort(idx, dim=0, descending=False)
            output = output[:, idx_sort]
        else:
            # Predict next word using Luong eq. 6
            output = self.out(concat_output)
            # generic output
            output = F.softmax(output, dim=1)

        # Return output and final hidden state
        return output, hidden, new_M_emo, context


class GreedySearchDecoder(nn.Module):
    '''
    Gready search decode
    '''
    def __init__(self, encoder, decoder,num_word = None):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq,target_emotions,input_length, max_length):
        # Forward input through encoder model
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden[:decoder.n_layers]
        # Initialize decoder input with SOS_token
        decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * SOS_token
        # Initialize tensors to append decoded words to
        all_tokens = torch.zeros([0], device=device, dtype=torch.long)
        all_scores = torch.zeros([0], device=device)
        # Set initial context value,last_rnn_output, internal_memory
        context_input = torch.FloatTensor(1,hidden_size)
        context_input = context_input.to(device)
        # last_rnn_output = torch.FloatTensor(hidden_size)
        internal_memory = torch.FloatTensor(batch_size,hidden_size)
        internal_memory = internal_memory.to(device)
        # Iteratively decode one word token at a time
        for _ in range(max_length):
            # Forward pass through decoder
            decoder_output, decoder_hidden,internal_memory,context_input = decoder(
                decoder_input,target_emotions, decoder_hidden,
                context_input, internal_memory,encoder_outputs
            )
            # Obtain most likely word token and its softmax score
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            # Record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        # Return collections of word tokens and scores
        return all_tokens, all_scores


class BeamSearchDecoder(nn.Module):
    '''
    Beam search decode
    '''
    def __init__(self, encoder, decoder,num_word,device):
        '''

        :param encoder: torch nn
        :param decoder: torch nn
        :param num_word: int vocabulary size
        '''
        super(BeamSearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.num_word = num_word
        self.device = device

    def forward(self, input_seq,target_emotions,input_length, max_length):
        SOS_token = 1
        # Forward input through encoder model
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]
        # Initialize decoder input with SOS_token
        decoder_input = torch.ones(1, 1, device=self.device, dtype=torch.long) * SOS_token
        # Initialize tensors to append decoded words to
        all_tokens = torch.zeros([0], device=self.device, dtype=torch.long)
        all_words_order = torch.zeros((1,self.num_word),device=self.device,dtype=torch.long)
        all_scores = torch.zeros([0], device=self.device)
        all_scores_array = torch.zeros((1,self.num_word),device=self.device,dtype=torch.float)
        # Set initial context value,last_rnn_output, internal_memory
        context_input = torch.ones(1,self.decoder.hidden_size,dtype=torch.float)
        context_input = context_input.to(self.device)
        # last_rnn_output = torch.FloatTensor(hidden_size)
        # Iteratively decode one word token at a time
        for _ in range(max_length):
            # Forward pass through decoder
            decoder_output, decoder_hidden,target_emotions,context_input = self.decoder(
                decoder_input,target_emotions, decoder_hidden,
                context_input, encoder_outputs
            )
            # Obtain most likely word token and its softmax score
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            decoder_input_order = torch.argsort(decoder_output,dim=1,descending=True)
            # Record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            all_scores_array = torch.cat((all_scores_array,decoder_output),dim = 0)
            all_words_order = torch.cat((all_words_order,decoder_input_order), dim=0)
            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        # Return collections of word tokens and scores
        sequences = self.beam_search(all_scores_array,3)
        return sequences
    def beam_search(self,array,k):
        array = array.tolist()
        sequences = [[list(), 1.0]]
        # walk over each step in sequence
        for row in array:
            all_candidates = list()
            # expand each current candidate
            for i in range(len(sequences)):
                seq, score = sequences[i]
                for j in range(len(row)):
                    candidate = [seq + [j], score * -np.log(row[j] + 1e-8)]
                    all_candidates.append(candidate)
            # order all candidates by score
            ordered = sorted(all_candidates, key=lambda tup:tup[1])
            # select k best
            sequences = ordered[:k]
        return sequences


def maskNLLLoss_IMemory(inp, target, mask,M_emo,device):
    '''
    Compute loss based on output and internal memory
    :param inp: decoder output
    :param target: target label
    :param mask: mask to record real size of sentence
    :param M_emo: internal memory
    :return: tensor that has loss
    '''
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).sum() + torch.norm(M_emo)
    loss = loss.to(device)
    return loss, nTotal.item()