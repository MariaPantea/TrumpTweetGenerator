from torch import nn
import torch

class RNN(nn.Module):
    def __init__(self, embedding_matrix, embedding_size, dict_size, input_size, output_size, hidden_dim, n_layers):
        super(RNN, self).__init__()
        self.embedding_size = embedding_size
        self.embedding_matrix = embedding_matrix

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        #Defining the layers
        self.embedding = nn.Embedding(dict_size, embedding_size)
        # RNN Layer
        self.rnn = nn.RNN(embedding_size, hidden_dim, n_layers, batch_first=True)   
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_size)
    
    def forward(self, x):
        
        # embedded_words = self.embedding(x)
        batch_size = x.size(0)
        embedding = self.embedding(x)
        embedding.weight = nn.Parameter(self.embedding_matrix)
        embedding.weight.requires_grad = False

        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(embedding, hidden)
        
        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
        
        return out, hidden
    
    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        is_cuda = torch.cuda.is_available()

        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        hidden = hidden.to(device)
        return hidden
