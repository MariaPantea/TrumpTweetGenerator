import pandas as pd
# from langdetect import detect
from copy import deepcopy
from collections import defaultdict
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import re
import numpy as np
import torch

MAX_LENGTH = 25
OCCURENCE_LIMIT = 2
batch_size = 2000


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

    # Train test split of data
    input_seq_train, input_seq_test, target_seq_train, target_seq_test = train_test_split(
        input_seqs, 
        target_seqs, 
        test_size=0.05, 
        random_state=1337,
        shuffle=True)

    # Construct dataloaders
    dataloader_train = dataloader_constructor(input_seq_train, target_seq_train, batch_size)
    dataloader_test = dataloader_constructor(input_seq_test, target_seq_test, batch_size)

    word2vecs = get_word2vecs(word2inx)

    return dataloader_train, dataloader_test, inx2word, word2inx, word2vecs, batch_size



def dataloader_constructor(input_vector, target_vector, batch_size):
        '''
        Constructs a dataloader from two numpy arrays. 
        input_vector is a numpy array of one_hot encoded product orders, with size (orders_in_data, num_products_per_order, product_vocabulary_size)
        target_vector is a numpy array of the sequentially next product_id purchased for the corresponding order in input_vector, with size (orders_in_data, num_products_per_order)
        batch_size is an integer value determining the number of data samples to extract from dataloader per iteration

        Thus, the one_hot_encoded version of product in position (x, y, :) in input_vector should be found as a product id in position (x, y-1) in target_vector
        '''

        input_tensor = torch.from_numpy(input_vector)
        target_tensor = torch.from_numpy(target_vector)

        dataset_for_dataloader = TensorDataset(input_tensor, target_tensor)
        
        return DataLoader(dataset_for_dataloader, batch_size=batch_size, shuffle=True)
