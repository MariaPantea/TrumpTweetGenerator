import pandas as pd
# from langdetect import detect
from copy import deepcopy
from collections import defaultdict
import re
import numpy as np

MAX_LENGTH = 25
OCCURENCE_LIMIT = 2
NUM_TWEETS = 2000


def is_english(text):
    try:
        lang = detect(text)
        return lang == 'en'
    except:
        return False

def save_only_english_tweets():
    df = pd.read_csv('data/twitter_data_without_RT.csv')

    # filter tweets which are not in english
    df = df[df.apply(lambda row:is_english(row['text']), axis=1)]
    df.to_csv("data/filtered_data.csv", index=False)


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def get_word2vecs(word2inx):
    embedding_size = 50
    embedding_dict = {}
    with open('data/glove/glove.6B.{}d.txt'.format(embedding_size), 'r') as f:
        mappings = f.readlines()
    
    for line in mappings:
        l = line.split()
        if l[0] in word2inx:
            embedding_dict[l[0]] = [float(i) for i in l[1:]]

    embedding_matrix = np.zeros(shape=(len(word2inx), embedding_size))
    for word, inx in word2inx.items():
        if word in embedding_dict:
            embedding_matrix[inx] = embedding_dict[word]
        else:
            embedding_matrix[inx] = np.random.uniform(-0.25,0.25,embedding_size)

    return embedding_matrix

def load_data():
    # df = pd.read_csv("data/filtered_data.csv")
    # df = df[['text']].copy()
    
    # df = df[~df.text.str.contains("http")]
    # df = df[~df.text.str.contains("www")]
    # df = df[~df.text.str.contains("@")]
    # df.text = df.text.apply(lambda x: clean_str(x))
    # df = df[:20]

    # df.to_csv("data/cleaned_data.csv", index=False)
    df = pd.read_csv('data/cleaned_data.csv')
    df = df[:NUM_TWEETS]

    # Join all the sentences together and extract the unique words from the combined sentences
    text = df.text.to_list()
    words = ''.join(text).split()

    vocabulary = defaultdict(int)
    for word in words:
        vocabulary[word] += 1
    
    # Removes words that occure less than OCCURENCE_LIMIT times
    words = [i for i in vocabulary if vocabulary[i] >= OCCURENCE_LIMIT]
    dict_size = len(words)
    print('dict size: ', dict_size)
    
    # Create dicts
    word2inx = dict(zip(words, range(1,dict_size+1)))
    inx2word = dict(zip(range(1, dict_size+1), words))
    
    # Add unknown word
    word2inx['<UNK>'] = 0
    inx2word[0] = '<UNK>'

    num_tweets = len(text)
    input_seqs = np.zeros(shape=(num_tweets, MAX_LENGTH), dtype=int)
    target_seqs = np.zeros(shape=(num_tweets, MAX_LENGTH), dtype=int)

    # Convert sentences to inx
    text = [tweet.split()[:MAX_LENGTH] for tweet in text]

    for i, tweet in enumerate(text):
        for j, word in enumerate(tweet):
            # print(word)
            input_seqs[i][j] = word2inx.get(word, 0)

    
    target_seqs = deepcopy(input_seqs)
    input_seqs = np.array([seq[:-1] for seq in input_seqs])
    target_seqs = np.array([seq[1:] for seq in target_seqs])

    word2vecs = get_word2vecs(word2inx)

    return input_seqs, target_seqs, inx2word, word2inx, word2vecs
