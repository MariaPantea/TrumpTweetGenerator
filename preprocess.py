import pandas as pd
from langdetect import detect
from copy import deepcopy



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


def clean_text(text):
    # remove links
    if 'http' in text:
        text = "".join(list(filter(lambda x :  'http' not in x, text.split())))
    return text


def load_data():
    df = pd.read_csv("data/filtered_data.csv")
    df = df[:10] 
    df = df[['text']].copy()
    
    df.text = df.text.apply(lambda x: clean_text(x))
    df.to_csv("data/cleaned_data.csv", index=False)

    # Join all the sentences together and extract the unique characters from the combined sentences
    text = df.text.to_list()
    chars = set(''.join(text))

    # Creating a dictionary that maps integers to the characters and vice versa
    int2char = dict(enumerate(chars))
    char2int = {char: ind for ind, char in int2char.items()}

    # Add whitespaces until the length of the sentence matches the length of the longest sentence
    maxlen = 50

    for i in range(len(text)):
        l = len(text[i])
        if maxlen > l:
            num_whitespaces = maxlen - l
            text[i] += ' ' * num_whitespaces
        else:
            text[i] = text[i][:maxlen]
    
    # Creating lists that will hold our input and target sequences
    input_seq = deepcopy(text)
    target_seq = deepcopy(text)

    for i in range(len(text)):
        # Remove last character for input sequence and the first character for target sequence
        input_seq[i] = [char2int[character] for character in text[i][:-1]]
        target_seq[i] = [char2int[character] for character in text[i][1:]]
    
    return input_seq, target_seq, int2char, char2int