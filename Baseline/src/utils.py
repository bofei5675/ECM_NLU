from src.preprocessing import *
import torch
from src.model import *

def train(input_variable, lengths, target_variable, target_variable_emotion,
          mask, max_target_len, encoder, decoder, embedding, emotion_embedding,
          encoder_optimizer, decoder_optimizer, batch_size, clip,device,
          teacher_forcing_ratio, hidden_size):
    '''
    This function run signle train step

    :param input_variable:
    :param lengths:
    :param target_variable:
    :param target_variable_emotion:
    :param mask:
    :param max_target_len:
    :param encoder:
    :param decoder:
    :param embedding:
    :param emotion_embedding:
    :param encoder_optimizer:
    :param decoder_optimizer:
    :param batch_size:
    :param clip:
    :param max_length:
    :return:
    '''
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
    # Determine if we are using teacher forcing this iteration
    if random.random() < teacher_forcing_ratio:
        use_teacher_forcing = True
    else:
        use_teacher_forcing = False

    # Forward batch of sequences through decoder one time step at a time
    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output, decoder_hidden, target_variable_emotion, context_input = decoder(
                decoder_input, target_variable_emotion, decoder_hidden,
                context_input, encoder_outputs
            )
            # Teacher forcing: next input is current target
            decoder_input = target_variable[t].view(1, -1)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss_IMemory(decoder_output, target_variable[t], mask[t],
                                                    target_variable_emotion,device)
            loss += mask_loss
            print_losses.append(mask_loss.item())  # print average loss
            n_totals += nTotal
    else:
        for t in range(max_target_len):
            decoder_output, decoder_hidden, target_variable_emotion, context_input = decoder(
                decoder_input, target_variable_emotion, decoder_hidden,
                context_input, encoder_outputs
            )
            # No teacher forcing: next input is decoder's own current output
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss_IMemory(decoder_output, target_variable[t], mask[t],
                                                    target_variable_emotion,device)
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
               decoder_optimizer, embedding,emotion_embedding,
               encoder_n_layers, decoder_n_layers, save_dir,
               n_iteration, batch_size, print_every, save_every,
               clip,corpus_name,external_memory,device,
               teacher_forcing_ratio, hidden_size):
    '''
    This function accept hyperparameter and run training

    :param model_name:
    :param voc:
    :param pairs:
    :param pairs_emotion:
    :param encoder:
    :param decoder:
    :param encoder_optimizer:
    :param decoder_optimizer:
    :param embedding:
    :param emotion_embedding:
    :param encoder_n_layers:
    :param decoder_n_layers:
    :param save_dir:
    :param n_iteration:
    :param batch_size:
    :param print_every:
    :param save_every:
    :param clip:
    :param corpus_name:
    :param external_memory:
    :return:
    '''
    loadFilename=None
    # Load batches for each iteration
    #training_batches = [batch2TrainData(voc, [random.choice(pairs) for _ in range(batch_size)])
                      #for _ in range(n_iteration)]
    print('Loading Training data ...')
    length_pairs = len(pairs)
    training_batches = [batch2TrainData(voc, [random.choice(range(length_pairs)) for _ in range(batch_size)],
                                       pairs,pairs_emotion) for _ in range(n_iteration)]
    # Initializations
    print('Initializing ...')
    start_iteration = 1
    print_loss = 0
    if loadFilename:
        start_iteration = checkpoint['iteration'] + 1

    # Training loop
    print("Training...")
    for iteration in range(start_iteration, n_iteration + 1):
        training_batch = training_batches[iteration - 1]
        # Extract fields from batch
        input_variable,input_variable_emotion, lengths, target_variable,target_variable_emotion, mask, max_target_len = training_batch

        # Run a training iteration with batch
        loss = train(input_variable, lengths, target_variable,target_variable_emotion,
                     mask, max_target_len, encoder,
                     decoder, embedding,emotion_embedding,
                     encoder_optimizer, decoder_optimizer,
                     batch_size, clip,device,teacher_forcing_ratio,
                     hidden_size)
        print_loss += loss

        # Print progress
        if iteration % print_every == 0 or iteration == 1:
            if iteration == 1:
                print_loss_avg = print_loss / 1
            else:
                print_loss_avg = print_loss / print_every
            perplexity = compute_perplexity(print_loss_avg)
            print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}; Perplexity: {:.2f}".format(iteration, iteration / n_iteration * 100, print_loss_avg,perplexity))
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
                'embedding': embedding.state_dict(),
                'external_memory':external_memory
            }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))


def evaluate(searcher, voc, sentence, emotions, beam_search,device,max_length):
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


def evaluateInput(searcher, voc, emotion_dict,device):
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
                output_words = evaluate(searcher, voc, input_sentence, emotion, True,device,voc.max_length)
                # Format and print response sentence
                output = []
                for word in output_words:
                    if word == 'PAD':
                        continue
                    elif word == 'EOS':
                        break
                    else:
                        output.append(word)
                print('Bot({}):'.format(emotion_dict[emotion]), ' '.join(output))

        except KeyError:
            print("Error: Encountered unknown word.")
