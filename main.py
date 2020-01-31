from preprocess import load_data
import numpy as np
import torch
from torch import nn

from model import RNN

def embedd(sequence, embeddings):
    # Creating a multi-dimensional array of zeros with the desired output shape
    embedded_seq = np.zeros((len(sequence), len(sequence[0]), len(embeddings[0])), dtype=np.float32)


    for i, sentence in enumerate(sequence):
        for j, word in enumerate(sentence):
            embedded_seq[i, j] = embeddings[word]
    return embedded_seq

def one_hot_encode(sequence, dict_size, seq_len, batch_size):
    # Creating a multi-dimensional array of zeros with the desired output shape
    features = np.zeros((batch_size, seq_len, dict_size), dtype=np.float32)
    
    # Replacing the 0 at the relevant character index with a 1 to represent that character
    for i in range(batch_size):
        for u in range(seq_len):
            features[i, u, sequence[i][u]] = 1
    return features

def get_index(word):
    return word2inx.get(word, 0)


# This function takes in the model and character as arguments and returns the next character prediction and hidden state
def predict(model, word):
    # One-hot encoding our input to fit into the model
    word = np.array([[get_index(w) for w in word]])
    # word = one_hot_encode(word, dict_size, word.shape[1], 1)
    word = torch.from_numpy(word).type(torch.LongTensor)
    word = word.to(device)
    
    out, hidden = model(word)
    
    prob = nn.functional.softmax(out[-1], dim=0).data
    # Taking the class with the highest probability score from the output
    char_ind = torch.max(prob, dim=0)[1].item()

    return inx2word[char_ind], hidden

# This function takes the desired output length and input characters as arguments, returning the produced sentence
def sample(model, out_len, tweet=['impeachment']):
    model.eval() # eval mode

    size = out_len - len(tweet)
    # Now pass in the previous characters and get a new one
    for ii in range(size):
        word, h = predict(model, tweet)
        tweet.append(word)
        h = h.to(device)

    return ' '.join(tweet)


if __name__ == "__main__":
    
    input_seq, target_seq, inx2word, word2inx, word2vec = load_data()

    embedding_size = len(word2vec[0])
    dict_size = len(word2inx)

    # Input shape --> (Batch Size, Sequence Length, One-Hot Encoding Size)
    # input_seq = one_hot_encode(input_seq, dict_size, len(input_seq[0]), len(input_seq))

    # input_seq = embedd(input_seq, word2vec)

    input_seq = torch.from_numpy(input_seq).type(torch.LongTensor)
    target_seq = torch.torch.LongTensor(target_seq)
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
    model = RNN(embedding_matrix=word2vec, embedding_size=50, dict_size=dict_size, input_size=dict_size, output_size=dict_size, hidden_dim=100, n_layers=1)
    model.to(device)

    # Define hyperparameters
    n_epochs = 700
    lr=0.01

    # Define Loss, Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)


    # Training Run
    for epoch in range(1, n_epochs + 1):
        optimizer.zero_grad() # Clears existing gradients from previous epoch
        input_seq = input_seq.to(device)
        target_seq = target_seq.to(device)
        output, h = model(input_seq)
        h = h.to(device)
        loss = criterion(output, target_seq.view(-1).long())
        loss.backward() # Does backpropagation and calculates gradients
        optimizer.step() # Updates the weights accordingly
        
        if epoch%10 == 0 or epoch == 1:
            print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
            print("Loss: {:.4f}".format(loss.item()))
    
    print(sample(model, 15, ['republicans']) + '!')
