

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn


class GreedySearchDecoder(nn.Module):
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
    def __init__(self, encoder, decoder,num_word):
        super(BeamSearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.num_word = num_word

    def forward(self, input_seq,target_emotions,input_length, max_length):
        # Forward input through encoder model
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden[:decoder.n_layers]
        # Initialize decoder input with SOS_token
        decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * SOS_token
        # Initialize tensors to append decoded words to
        all_tokens = torch.zeros([0], device=device, dtype=torch.long)
        all_words_order = torch.zeros((1,self.num_word),device=device,dtype=torch.long)
        all_scores = torch.zeros([0], device=device)
        all_scores_array = torch.zeros((1,self.num_word),device=device,dtype=torch.float)
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