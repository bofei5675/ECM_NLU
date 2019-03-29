from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim

from src.decoder import *
from src.encoder import *
from src.ECM import *
from src.attention import *
from src.searcher import *



import random

import os

from preprocessing_dailydialogue import *

# Define constant
# Default word tokens
#
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token
MAX_LENGTH = 20  # Maximum sentence length to consider
MIN_COUNT = 3  # Minimum word count threshold for trimming
save_dir = os.path.join("data", "save")
def maskNLLLoss_IMemory(inp, target, mask,M_emo):
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).sum() + torch.norm(M_emo)
    loss = loss.to(device)
    return loss, nTotal.item()


def train(input_variable, lengths, target_variable, target_variable_emotion,
          mask, max_target_len, encoder, decoder,encoder_optimizer, decoder_optimizer,
          batch_size, clip):
    # Zero gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # Set device options
    input_variable = input_variable.to(device)
    lengths = lengths.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)
    target_variable_emotion = target_variable_emotion.to(device)
    # Initialize variables
    loss = 0
    print_losses = []
    n_totals = 0

    # Forward pass through encoder
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

    # Create initial decoder input (start with SOS tokens for each sentence)
    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)

    # Set initial decoder hidden state to the encoder's final hidden state
    decoder_hidden = encoder_hidden[:decoder.n_layers]
    # Set initial context value,last_rnn_output, internal_memory
    context_input = torch.FloatTensor(batch_size, hidden_size)
    context_input = context_input.to(device)
    # last_rnn_output = torch.FloatTensor(hidden_size)
    internal_memory = torch.randn(batch_size, hidden_size, dtype=torch.float)
    internal_memory = internal_memory.to(device)
    # Determine if we are using teacher forcing this iteration
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    # Forward batch of sequences through decoder one time step at a time
    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output, decoder_hidden, internal_memory, context_input = decoder(
                decoder_input, target_variable_emotion, decoder_hidden,
                context_input, internal_memory, encoder_outputs
            )
            # Teacher forcing: next input is current target
            decoder_input = target_variable[t].view(1, -1)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss_IMemory(decoder_output, target_variable[t], mask[t], internal_memory)
            loss += mask_loss
            print_losses.append(mask_loss.item())  # print average loss
            n_totals += nTotal
    else:
        for t in range(max_target_len):
            decoder_output, decoder_hidden, internal_memory, context_input = decoder(
                decoder_input, target_variable_emotion, decoder_hidden,
                context_input, internal_memory, encoder_outputs
            )
            # No teacher forcing: next input is decoder's own current output
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss_IMemory(decoder_output, target_variable[t], mask[t], internal_memory)
            loss += mask_loss
            print_losses.append(mask_loss.item())  # print average loss
            n_totals += nTotal

    # Perform backpropatation
    loss.backward()

    # Clip gradients: gradients are modified in place
    _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # Adjust model weights
    encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(print_losses) / n_totals


def trainIters(model_name, voc, pairs,pairs_emotion,
               encoder, decoder, encoder_optimizer,
               decoder_optimizer,
               encoder_n_layers, decoder_n_layers, save_dir,
               n_iteration, batch_size, print_every, save_every,
               clip,corpus_name):

    loadFilename=None
    # Load batches for each iteration
    #training_batches = [batch2TrainData(voc, [random.choice(pairs) for _ in range(batch_size)])
                      #for _ in range(n_iteration)]
    print('Loading Training data ...')
    training_batches = [batch2TrainData(voc, [random.choice(list(range(len(pairs)))) for _ in range(batch_size)],
                                       pairs,pairs_emotion) for _ in range(n_iteration)]
    # Initializations
    print('Initializing ...')
    start_iteration = 1
    print_loss = 0
    if loadFilename:
        start_iteration = checkpoint['iteration'] + 1

    # Training loop
    print(train)
    print("Training...")
    for iteration in range(start_iteration, n_iteration + 1):
        training_batch = training_batches[iteration - 1]
        # Extract fields from batch
        input_variable,input_variable_emotion, lengths, target_variable,target_variable_emotion, mask, max_target_len = training_batch

        # Run a training iteration with batch
        loss = train(input_variable, lengths, target_variable,target_variable_emotion,
                     mask, max_target_len, encoder,
                     decoder,
                     encoder_optimizer, decoder_optimizer,
                     batch_size, clip)
        print_loss += loss

        # Print progress
        if iteration % print_every == 0 or iteration == 1:
            print_loss_avg = print_loss / print_every
            print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(iteration, iteration / n_iteration * 100, print_loss_avg))
            print_loss = 0

        # Save checkpoint
        if (iteration % save_every == 0):
            directory = os.path.join(save_dir, model_name, corpus_name, '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size))
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'iteration': iteration,
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
                'loss': loss,
                'voc_dict': voc.__dict__,
                'embedding': embedding.state_dict()
            }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))


def evaluate(encoder, decoder, searcher, voc, sentence, emotions, max_length=MAX_LENGTH, beam_search=False):
    emotions = int(emotions)
    emotions = torch.LongTensor([emotions])
    ### Format input sentence as a batch
    # words -> indexes
    indexes_batch = [indexesFromSentence(voc, sentence)]
    # Create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # Transpose dimensions of batch to match models' expectations
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    # Use appropriate device
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)
    emotions = emotions.to(device)

    # indexes -> words
    if beam_search:
        sequences = searcher(input_batch, emotions, lengths, max_length)
        decoded_words = beam_decode(sequences, voc)
    else:
        # Decode sentence with searcher
        tokens, scores = searcher(input_batch, emotions, lengths, max_length)
        decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words


def beam_decode(sequences, voc):
    for each in sequences:
        for idxs in each:
            return [voc.index2word[idx] for idx in idxs[:-1]]


def evaluateInput(encoder, decoder, searcher, voc, emotion_dict):
    input_sentence = ''
    while (1):
        try:
            # Get input sentence
            input_sentence = input('> ')
            for emotion in range(len(emotion_dict)):
                # Check if it is quit case
                if input_sentence == 'q' or input_sentence == 'quit': break
                # Normalize sentence
                input_sentence = normalizeString(input_sentence)
                # Evaluate sentence
                output_words = evaluate(encoder, decoder, searcher, voc, input_sentence, emotion, beam_search=True)
                # Format and print response sentence
                output = []
                for word in output_words:
                    if word == 'PAD':
                        continue
                    elif word == 'EOS':
                        break
                    else:
                        output.append(word)
                # output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
                print('Bot({}):'.format(emotion_dict[emotion]), ' '.join(output))
        except KeyError:
            print("Error: Encountered unknown word.")
if __name__ == '__main__':

    # read data
    data, voc, pairs, pairs_emotion = get_data()

    #build model

    # Configure models
    model_name = 'emotion_model'
    corpus_name = 'dailydialogue'
    attn_model = 'dot'
    # attn_model = 'general'
    # attn_model = 'concat'
    hidden_size = 500
    encoder_n_layers = 2
    decoder_n_layers = 2
    dropout = 0.4
    batch_size = 64
    # number of emotion
    num_emotions = 7

    # Set checkpoint to load from; set to None if starting from scratch
    loadFilename = None
    checkpoint_iter = 120
    training = True
    if loadFilename:
        training = False
    # loadFilename = os.path.join(save_dir, model_name, corpus_name,
    #                            '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
    #                            '{}_checkpoint.tar'.format(checkpoint_iter))

    # Load model if a loadFilename is provided
    if loadFilename:
        # If loading on same machine the model was trained on
        checkpoint = torch.load(loadFilename, map_location='cpu')
        # If loading a model trained on GPU to CPU
        # checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
        encoder_sd = checkpoint['en']
        decoder_sd = checkpoint['de']
        encoder_optimizer_sd = checkpoint['en_opt']
        decoder_optimizer_sd = checkpoint['de_opt']
        embedding_sd = checkpoint['embedding']
        voc.__dict__ = checkpoint['voc_dict']

    print('Building encoder and decoder ...')
    # Initialize word embeddings
    embedding = nn.Embedding(voc.num_words, hidden_size)
    emotion_embedding = nn.Embedding(num_emotions, hidden_size)
    if loadFilename:
        embedding.load_state_dict(embedding_sd)
    # Initialize encoder & decoder models
    encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
    decoder = LuongAttnDecoderRNN(attn_model, embedding, emotion_embedding, hidden_size, voc.num_words,
                                  decoder_n_layers, dropout)
    if loadFilename:
        encoder.load_state_dict(encoder_sd)
        decoder.load_state_dict(decoder_sd)
    # Use appropriate device
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    print('Models built and ready to go!')

    # Configure training/optimization
    clip = 50.0
    teacher_forcing_ratio = 1.0
    learning_rate = 0.0001
    decoder_learning_ratio = 5.0
    n_iteration = 100
    print_every = 10
    save_every = 40

    # Ensure dropout layers are in train mode
    encoder.train()
    decoder.train()

    # Initialize optimizers
    print('Building optimizers ...')
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
    if loadFilename:
        encoder_optimizer.load_state_dict(encoder_optimizer_sd)
        decoder_optimizer.load_state_dict(decoder_optimizer_sd)

    # Run training iterations

    if training:
        print("Starting Training!")
        trainIters(model_name, voc, pairs, pairs_emotion, encoder, decoder, encoder_optimizer, decoder_optimizer,
                 encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size,
                   print_every, save_every, clip, corpus_name)
    validation = False
    if loadFilename:
        validation = True
    if validation:
        # emotion dictionary
        emo_dict = {0: 'no emotion', 1: 'anger', 2: 'disgust',
                    3: 'fear', 4: 'happiness',
                    5: 'sadness', 6: 'surprise'}
        # Set dropout layers to eval mode

        encoder.eval()
        decoder.eval()

        # Initialize search module
        searcher = GreedySearchDecoder(encoder, decoder)
        searcher2 = BeamSearchDecoder(encoder, decoder, voc.num_words)
        # Begin chatting (uncomment and run the following line to begin)
        evaluateInput(encoder, decoder, searcher2, voc, emo_dict)
