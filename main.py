from preprocess import load_data
import argparse
import numpy as np
import torch
from torch import nn
import json
import pickle

from model import RNN


def get_index(word, word2inx):
    return word2inx.get(word, 0)


# This function takes in the model and character as arguments and returns the next character prediction and hidden state
def predict(model, word, device, inx2word, word2inx):
    # One-hot encoding our input to fit into the model
    word = np.array([[get_index(w, word2inx) for w in word]])
    # word = one_hot_encode(word, dict_size, word.shape[1], 1)
    word = torch.from_numpy(word).type(torch.LongTensor)
    word = word.to(device)
    
    out, hidden = model(word)
    
    prob = nn.functional.softmax(out[-1], dim=0).data
    # Taking the class with the highest probability score from the output
    word_ind = torch.max(prob, dim=0)[1].item()

    return inx2word[word_ind], hidden

# This function takes the desired output length and input characters as arguments, returning the produced sentence
def sample(out_len, tweet):
    with open('models/word2vec.p', 'rb') as f:
        word2vec = pickle.load(f)
        word2vec = torch.tensor(word2vec)

    with open('models/translators.p', 'rb') as f:
        translators = pickle.load(f)

    inx2word = {int(k): v for k, v in translators['inx2word'].items()}
    word2inx = {k: int(v) for k, v in translators['word2inx'].items()}

    dict_size = len(inx2word)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = RNN(embedding_matrix=word2vec, dict_size=dict_size, hidden_dim=100, n_layers=1)
    model.load_state_dict(torch.load('models/rnn'))
    model.eval()
    model = model.to(device)

    size = out_len - len(tweet)
    # Now pass in the previous characters and get a new one
    for _ in range(size):
        word, h = predict(model, tweet, device, inx2word, word2inx)
        if word != '<UNK>':
            tweet.append(word)
        h = h.to(device)

    return ' '.join(tweet)



def train():
    
    train_seq, test_seq, inx2word, word2inx, word2vec, batch_size = load_data()

    translators = {'inx2word' : inx2word, 'word2inx': word2inx}
    with open('models/translators.p', 'wb') as f:
        pickle.dump(translators, f)
    with open('models/word2vec.p', 'wb') as f:
        pickle.dump(word2vec, f)

    dict_size = len(word2inx)

    word2vec = torch.tensor(word2vec)

    # check for GPU
    is_cuda = torch.cuda.is_available()

    if is_cuda:
        device = torch.device("cuda")
        print("GPU is available")
    else:
        device = torch.device("cpu")
        print("GPU not available, CPU used")

    # Instantiate the model with hyperparameters
    model = RNN(embedding_matrix=word2vec, dict_size=dict_size, hidden_dim=100, n_layers=1)
    model.to(device)

    # Define hyperparameters
    batch_size = 2000
    n_epochs = 100
    lr = 0.01

    # Define Loss, Optimizer
    lossfunction = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)


    # Training Run
    for epoch in range(1, n_epochs + 1):
        epoch_loss = 0
        for _, (input_seq, target_seq) in enumerate(train_seq):

            optimizer.zero_grad() # Clears existing gradients from previous epoch
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)
            output, h = model(input_seq)
            h = h.to(device)
            loss = lossfunction(output, target_seq.view(-1).long())
            loss.backward() # Does backpropagation and calculates gradients
            optimizer.step()  # Updates the weights accordingly
            epoch_loss += loss.item()
            
        if epoch % 10 == 0 or epoch == 1:
            loss_test_total = 0
            for input_test, target_test in test_seq:
                input_test = input_test.to(device)
                target_test = target_test.to(device)

                output_test, _ = model(input_test)
                loss_test = lossfunction(output_test, target_test.view(-1).long())
                loss_test_total += loss_test.item()

            norm_loss = epoch_loss / (len(train_seq) * batch_size)
            norm_loss_test = loss_test_total / (len(test_seq) * batch_size)

            print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
            print("Train loss: {:.4f}".format(norm_loss), end=' | ')
            print("Test loss: {:.4f}".format(norm_loss_test))
            torch.save(model.state_dict(), 'models/rnn')
    
    print('Training done')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--start', 
        type=str,
        nargs='+',
        default=['mexico'], 
        help='Start of the sentence')

    args=parser.parse_args()

    #train()
    print('\n\n==> ' + sample(25, args.start) + '...')
    print('\n\n***')
