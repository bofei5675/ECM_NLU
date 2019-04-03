from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
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
import numpy as np
'''
CONSTANT
'''
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token

def printLines(file, n=5):
    with open(file, 'rb') as datafile:
        lines = datafile.readlines()
    for line in lines[:n]:
        print(line)


# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s


def extractSentencePairs(conversations, is_emotions=True):
    qa_pairs = []
    i = 0
    while i < len(conversations) - 1:
        if is_emotions:
            qa_pairs.append([conversations[i], conversations[i + 1]])
        else:
            qa_pairs.append([normalizeString(conversations[i]), normalizeString(conversations[i + 1])])
        i += 1

    return qa_pairs


def loadLines(fileName):
    lines = {}
    with open(fileName, 'r') as f:
        for idx, line in enumerate(f):
            conversations = [i.strip() for i in line.strip().split('__eou__')[:-1]]
            qa_pairs = extractSentencePairs(conversations, is_emotions=False)
            lines[idx] = qa_pairs
    return lines


def loadEmotions(fileName):
    lines = {}
    with open(fileName, 'r') as f:
        for idx, line in enumerate(f):
            emotions = [int(i) for i in line.strip().split(' ')]

            emotions_pair = extractSentencePairs(emotions, is_emotions=True)
            lines[idx] = emotions_pair

    return lines
class Voc:
    def __init__(self, name,min_count,max_length):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3  # Count SOS, EOS, PAD
        self.min_count = min_count
        self.max_length = max_length
    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    # Remove words below a certain count threshold
    def trim(self):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= self.min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.word2index = {'PAD':0, 'SOS':1,'EOS':2}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3 # Count default tokens

        for word in keep_words:
            self.addWord(word)


# Read query/response pairs and return a voc object
import string


def flatten(data):
    pairs = []
    for key, values in data.items():
        for each in values:
            pairs.append([each[0], each[1]])

    return pairs


def readVocs(corpus_data, emotions_data,min_count,max_length):
    print("Reading lines...")
    # Read the file and split into lines
    conversations = loadLines(corpus_data)
    emotions = loadEmotions(emotions_data)

    voc = Voc('Movie_Dialogue',min_count,max_length)
    return voc, flatten(conversations), flatten(emotions)


# Returns True iff both sentences in a pair 'p' are under the MAX_LENGTH threshold
def filterPair(p,max_length):
    # Input sequences need to preserve the last word for EOS token
    return len(p[0].split(' ')) < max_length and len(p[1].split(' ')) < max_length


# Filter pairs using filterPair condition
def filterPairs(pairs,pairs_emotion,max_length):
    keep_pairs,keep_pairs_emotion = [], []
    for pair,pair_emotion in zip(pairs,pairs_emotion):
        if filterPair(pair,max_length):
            keep_pairs.append(pair)
            keep_pairs_emotion.append(pair_emotion)
    return keep_pairs,keep_pairs_emotion


# Using the functions defined above, return a populated voc object and pairs list
def loadPrepareData(corpus_data, emotions_data,min_count,max_length,drop_num):
    print("Start preparing training data ...")
    voc, pairs, pairs_emotion = readVocs(corpus_data, emotions_data,min_count,max_length)
    # flatten the pairs of sentences
    print("Read {!s} sentence pairs".format(len(pairs)))
    pairs,pairs_emotion = filterPairs(pairs,pairs_emotion,voc.max_length)
    print("Trimmed to {!s} sentence pairs".format(len(pairs)))
    print('Emotions left {}'.format(len(pairs_emotion)))
    print('Under sample one categories:')
    pairs, pairs_emotion = under_sample(pairs, pairs_emotion, 0, drop_num)
    print("Counting words...")
    for pair in pairs:
        voc.addSentence(pair[0])
        voc.addSentence(pair[1])
    print("Counted words:", voc.num_words)
    return voc, pairs, pairs_emotion

def trimRareWords(voc, pairs,pairs_emotion, min_count):
    # Trim words used under the MIN_COUNT from the voc

    voc.trim()
    # Filter out pairs with trimmed words
    keep_pairs = []
    keep_emotions = []
    for pair,pair_emotion in zip(pairs,pairs_emotion):
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True
        # Check input sentence
        for word in input_sentence.split(' '):
            if word not in voc.word2index:
                keep_input = False
                break
        # Check output sentence
        for word in output_sentence.split(' '):
            if word not in voc.word2index:
                keep_output = False
                break

        # Only keep pairs that do not contain trimmed word(s) in their input or output sentence
        if keep_input and keep_output:
            keep_pairs.append(pair)
            keep_emotions.append(pair_emotion)

    print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))
    return keep_pairs,keep_emotions


def indexesFromSentence(voc, sentence):
    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]


def zeroPadding(l, fillvalue=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))

def binaryMatrix(l, value=PAD_token):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m

# Returns padded input sequence tensor and lengths
def inputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths

# Returns padded target sequence tensor, padding mask, and max target length
def outputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.ByteTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len
# return an emotion tensor
def emotion_tensor(input_list):
    return torch.LongTensor(input_list)
# Returns all items for a given batch of pairs
def batch2TrainData(voc, index,pairs,pairs_emotion):
    pair_batch = [pairs[idx] for idx in index]
    pair_batch_emotions =[pairs_emotion[idx] for idx in index]
    keys = [len(x[0].split(' ')) for x in pair_batch]
    sorted_index = np.argsort(keys)[::-1]
    input_batch, output_batch = [], []
    input_batch_emotion, output_batch_emotion = [],[]
    for idx in sorted_index:
        input_batch.append(pair_batch[idx][0])
        input_batch_emotion.append(pair_batch_emotions[idx][0])
        output_batch.append(pair_batch[idx][1])
        output_batch_emotion.append(pair_batch_emotions[idx][1])
    inp, lengths = inputVar(input_batch, voc)
    output, mask, max_target_len = outputVar(output_batch, voc)
    input_batch_emotion = emotion_tensor(input_batch_emotion)
    output_batch_emotion = emotion_tensor(output_batch_emotion)
    return inp,input_batch_emotion, lengths, output,output_batch_emotion, mask, max_target_len


def get_data(data_path = '../data/ijcnlp_dailydialog', corpus_name = 'dialogues_text.txt', emotions_file = 'dialogues_emotion.txt',min_count = 1,max_length= 10,drop_num = 30000):
    DATA_PATH = data_path 
    corpus = os.path.join(DATA_PATH, corpus_name)
    emotions = os.path.join(DATA_PATH, emotions_file)

    voc, pairs, pairs_emotion = loadPrepareData(corpus, emotions,min_count,max_length,drop_num)
    # Print some pairs to validate
    # Trim voc and pairs
    pairs, pairs_emotion = trimRareWords(voc, pairs, pairs_emotion, min_count)
    '''
    dataset = []
    for idx, qa in enumerate(pairs):
        question, response = qa
        emotions = pairs_emotion[idx]
        question_emotion, response_emotion = emotions
        question = [question] + [question_emotion] * 2
        response = [[response] + [response_emotion] * 2]
        dataset.append([question, response])

    train = dataset
    '''
    return voc,pairs,pairs_emotion

def under_sample(pairs,pairs_emotion,emotions_cate = 0,drop_num = 30000):
    '''
    Under sample the response that has emotion category 0 (no emotion)
    :param pairs:
    :param pairs_emotion:
    :param emotions_cate:
    :param drop_num:
    :return:
    '''
    drop_no_emotion = True
    keep_pairs = []
    keep_pairs_emotion = []
    drop_count = 0
    for conversation,emotion in zip(pairs,pairs_emotion):
        post,response = conversation
        if emotion[1] == emotions_cate and drop_no_emotion:
            drop_count += 1
            if drop_count == drop_num:
                drop_no_emotion = False
            continue
        else:
            keep_pairs.append(conversation)
            keep_pairs_emotion.append(emotion)
    print('Under sample non-emotion to {} samples'.format(len(keep_pairs)))
    return keep_pairs,keep_pairs_emotion


def get_ememory(file_path, voc):
    '''
    Get external memory from file. And generate category embedding based on the
    current vocabulary

    :param file_path:
    :param voc:
    :return:
    '''
    emotion_words = [0] * voc.num_words
    count = 0
    with open(file_path, 'r') as f:
        for each in f:
            each = each.rstrip()
            if each in voc.word2index:
                count += 1
                emotion_words[voc.word2index[each]] = 1
    print('Emotion word counts:', count)
    return torch.ByteTensor(emotion_words)
