from typing import ClassVar
import torch
import torch.nn as nn
import torch.nn.functional as F

# Initialize weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class MLP(nn.Module):
    #### F_j (s) with shared first layers
    def __init__(self, 
                inputs_dim, 
                outputs_dim, 
                hidden_dim):
        super(MLP, self).__init__()

        self.linear1 = nn.Linear(inputs_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, outputs_dim)
        self.apply(weights_init_)

    def forward(self, inputs):
        x = F.relu(self.linear1(inputs))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class MMLP(nn.Module):
    """
    distribution to vector
    """

    def __init__(self, 
                batch_size, 
                sample_size, 
                features_dim,
                hidden_dim, 
                out_dim, 
                cuda):
        super(MMLP, self).__init__()
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.number_features = features_dim

        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.nonlinearity = F.relu
        self.device = torch.device("cuda" if cuda else "cpu")
        # modules
        self.F = MLP(features_dim,out_dim, hidden_dim)
        self.b = torch.zeros(out_dim, requires_grad=True, device=self.device)

    def forward(self, x):
        e = self.F(x)
        e = self.pool(e)
        e = self.nonlinearity(e + self.b)
        return e

    def pool(self, e):
        e = e.view(self.batch_size, self.sample_size, self.out_dim)
        e = e.mean(1).view(self.batch_size, self.out_dim)
        return e


# Recurrent neural network (many-to-one)
class RNN(nn.Module):
    def __init__(self, 
                batch_size, 
                features_dim, 
                hidden_dim,
                out_dim,
                cuda, 
                num_layers=2 
                ):
        super(RNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(features_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, out_dim)
        self.device = torch.device("cuda" if cuda else "cpu")
    
    def forward(self, x):
        # Set initial hidden and cell states 
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(self.device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(self.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_dim)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out


class Classify(nn.Module):
    """
    2layer-MLP
    """

    def __init__(self, 
                batch_size, 
                out_dim, 
                hidden_dim, 
                class_size):
        super(Classify, self).__init__()
        self.batch_size = batch_size
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.class_size = class_size

        self.nonlinearity = F.relu

        self.F = MLP(out_dim, class_size, hidden_dim)

    def forward(self, x):
        e = self.F(x)
        e = F.softmax(e, dim=0)
        return e


    

