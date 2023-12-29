import torch
import torch.nn as nn
from nets import Classify  
from nets import MMLP
from nets import MLP
from nets import RNN

class ClassifyMMLP(nn.Module):

    def __init__( self, 
                batch_size, 
                sample_size, 
                features_dim, 
                hidden_dim, 
                out_dim, 
                class_size, 
                cuda):
        super(ClassifyMMLP, self).__init__()
        self.mmlp = MMLP(batch_size, sample_size, features_dim, hidden_dim, out_dim, cuda)
        self.calssify = Classify(batch_size, out_dim, hidden_dim, class_size)
        self.loss_func = torch.nn.CrossEntropyLoss()
    def forward(self, x):
        e = self.mmlp(x)
        e = self.calssify(e)
        return e 
    
    def loss(self, output, label):
        loss = self.loss_func(output, label.long())  # debug on computer
        return loss, output, label

    def step(self, batch, label, optimizer):
        output = self.forward(batch)
        loss, output, label = self.loss(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss, output, label

    def save(self, optimizer, path):
        torch.save({
            'mnn_model_state': self.state_dict(),
            'mnn_optimizer_state': optimizer.state_dict()
        }, path)


class ClassifyMLP(nn.Module):

    def __init__( self, 
                batch_size, 
                sample_size, 
                features_dim, 
                hidden_dim, 
                out_dim, 
                class_size, 
                cuda):
        super(ClassifyMLP, self).__init__()
        in_MLP = sample_size*features_dim
        self.batch_size = batch_size
        self.mlp = MLP(in_MLP, out_dim, hidden_dim)
        self.calssify = Classify(batch_size, out_dim, hidden_dim, class_size)
        self.loss_func = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        x = x.view(self.batch_size,-1)
        e = self.mlp(x)
        e = self.calssify(e)
        return e 
    
    def loss(self, output, label):
        loss = self.loss_func(output, label.long())
        return loss, output, label

    def step(self, batch, label, optimizer):
        output = self.forward(batch)
        loss, output, label = self.loss(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss, output, label

    def save(self, optimizer, path):
        torch.save({
            'nn_model_state': self.state_dict(),
            'nn_optimizer_state': optimizer.state_dict()
        }, path)  


class ClassifyLSTM(nn.Module):

    def __init__(self, 
                batch_size, 
                sample_size, 
                features_dim, 
                hidden_dim, 
                out_dim, 
                class_size, 
                cuda):
        super(ClassifyLSTM, self).__init__()

        self.batch_size = batch_size
        self.lstm = RNN(batch_size, features_dim, hidden_dim, out_dim, cuda)
        self.calssify = Classify(batch_size, out_dim, hidden_dim, class_size)
        self.loss_func = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        e = self.lstm(x)
        e = self.calssify(e)
        return e 
    
    def loss(self, output, label):
        loss = self.loss_func(output, label.long())
        return loss, output, label

    def step(self, batch, label, optimizer):
        output = self.forward(batch)
        loss, output, label = self.loss(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss, output, label

    def save(self, optimizer, path):
        torch.save({
            'lstm_model_state': self.state_dict(),
            'lstm_optimizer_state': optimizer.state_dict()
        }, path)  


