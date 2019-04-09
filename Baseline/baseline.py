from src.utils import *
from src.preprocessing import *

from src.model import *

# Define constant
# Default word tokens
#
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token
MAX_LENGTH = 10  # Maximum sentence length to consider
MIN_COUNT = 1    # Minimum word count threshold for trimming
save_dir = os.path.join("data", "save")

emo_dict = { 0: 'no emotion', 1: 'anger', 2: 'disgust',
            3: 'fear', 4: 'happiness',
            5: 'sadness', 6: 'surprise'}

emo_group = {
    0:0,
    1:1,
    2:1,
    3:1,
    4:2,
    5:1,
    6:2
}
if __name__ == '__main__':
    # preprocessing
    voc, pairs, pairs_emotion = get_data('data/ijcnlp_dailydialog',min_count=MIN_COUNT, max_length=MAX_LENGTH, drop_num=6000)

    pairs_emotion = group_emotions(pairs_emotion, emo_group)
    print(len(pairs), len(pairs_emotion))
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
    # load external memory based vocab.
    emotion_words = get_ememory('jupyter/ememory.txt', voc)
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
        emotion_words = checkpoint['external_memory']

    print('Building encoder and decoder ...')
    # Initialize word embeddings
    embedding = nn.Embedding(voc.num_words, hidden_size)
    emotion_embedding = nn.Embedding(num_emotions, hidden_size)
    if loadFilename:
        embedding.load_state_dict(embedding_sd)
    # Initialize encoder & decoder models
    encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
    decoder = LuongAttnDecoderRNN(attn_model, embedding, emotion_embedding, hidden_size,
                                  voc.num_words, emotion_words, decoder_n_layers, dropout)
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
    n_iteration = 10
    print_every = 1
    save_every = 2

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
                   embedding, emotion_embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size,
                   print_every, save_every, clip, corpus_name, emotion_words,device,teacher_forcing_ratio, hidden_size)