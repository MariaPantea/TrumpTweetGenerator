from preprocess import load_data
import numpy as np
import torch
from torch import nn

from model import RNN


def one_hot_encode(sequence, dict_size, seq_len, batch_size):
    # Creating a multi-dimensional array of zeros with the desired output shape
    features = np.zeros((batch_size, seq_len, dict_size), dtype=np.float32)
    
    # Replacing the 0 at the relevant character index with a 1 to represent that character
    for i in range(batch_size):
        for u in range(seq_len):
            features[i, u, sequence[i][u]] = 1
    return features

# This function takes in the model and character as arguments and returns the next character prediction and hidden state
def predict(model, character):
    # One-hot encoding our input to fit into the model
    character = np.array([[char2int[c] for c in character]])
    character = one_hot_encode(character, dict_size, character.shape[1], 1)
    character = torch.from_numpy(character)
    character.to(device)
    
    out, hidden = model(character)

    prob = nn.functional.softmax(out[-1], dim=0).data
    # Taking the class with the highest probability score from the output
    char_ind = torch.max(prob, dim=0)[1].item()

    return int2char[char_ind], hidden

# This function takes the desired output length and input characters as arguments, returning the produced sentence
def sample(model, out_len, start='hey'):
    model.eval() # eval mode
    start = start.lower()
    # First off, run through the starting characters
    chars = [ch for ch in start]
    size = out_len - len(chars)
    # Now pass in the previous characters and get a new one
    for _ in range(size):
        char, _ = predict(model, chars)
        chars.append(char)

    return ''.join(chars)


if __name__ == "__main__":
    
    input_seq, target_seq, int2char, char2int = load_data()
    
    dict_size = len(char2int)
    seq_len = len(input_seq[0])
    batch_size = len(input_seq)


    # Input shape --> (Batch Size, Sequence Length, One-Hot Encoding Size)
    input_seq = one_hot_encode(input_seq, dict_size, seq_len, batch_size)

    input_seq = torch.from_numpy(input_seq)
    target_seq = torch.Tensor(target_seq)

    # check for GPU
    is_cuda = torch.cuda.is_available()

    if is_cuda:
        device = torch.device("cuda")
        print("GPU is available")
    else:
        device = torch.device("cpu")
        print("GPU not available, CPU used")

    # Instantiate the model with hyperparameters
    model = RNN(input_size=dict_size, output_size=dict_size, hidden_dim=12, n_layers=1)
    model.to(device)

    # Define hyperparameters
    n_epochs = 100
    lr=0.01

    # Define Loss, Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)


    # Training Run
    for epoch in range(1, n_epochs + 1):
        optimizer.zero_grad() # Clears existing gradients from previous epoch
        input_seq = input_seq.to(device)
        target_seq = target_seq.to(device)
        output, _ = model(input_seq)
        loss = criterion(output, target_seq.view(-1).long())
        loss.backward() # Does backpropagation and calculates gradients
        optimizer.step() # Updates the weights accordingly
        
        if epoch%10 == 0 or epoch == 1:
            print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
            print("Loss: {:.4f}".format(loss.item()))
    
    print(sample(model, 15, 'I') + '!')