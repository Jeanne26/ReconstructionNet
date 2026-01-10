import torch
import torch.nn as nn
from math import log2

class ReconstructionNet(nn.Module):
    def __init__(self,input_dim,num_class):
        super().__init__()
        self.input_dim= input_dim
        self.num_class = num_class
        #create num_class autoencoders
        self.autoencoders = nn.ModuleList([AutoEncoder(input_dim) for _ in range(num_class)])
        
        #create num_class dense layers
        self.dense_layers = nn.ModuleList([])
        for i in range(num_class):
            self.dense_layers.append(nn.Linear(input_dim,1))

    def forward(self,x):
        #reconstructed list
        r_list = []
        #weighted reconstruction error list
        wre_list = []
        for i in range(self.num_class):
            x_copy = x.clone() 
            reconstructed = self.autoencoders[i](x_copy)
            reconstruction_error = torch.sqrt((reconstructed-x_copy)**2)
            weighted_reconstruction_error = self.dense_layers[i](reconstruction_error)

            r_list.append(reconstruction_error)
            wre_list.append(weighted_reconstruction_error)
        
        wre = torch.cat(wre_list, dim=1)  # batch_size , num_class
        rec_error = torch.cat(r_list, dim=1 ) #batch size, input dim
        logits = nn.Softmax(dim=1)(-wre)  #batch, num_calss


        return logits, rec_error, wre



class Encoder(nn.Module):
    def __init__(self,input_dim):
        super().__init__()
        num_layers = compute_num_layers(input_dim)
        self.layers = nn.Sequential()
        for i in range(num_layers-1):
            self.layers.append(nn.Linear(input_dim,input_dim//2))
            self.layers.append(nn.ReLU())
            input_dim = input_dim//2

        self.layers.append(nn.Linear(input_dim,input_dim//2))
    
    def forward(self,x):
        return self.layers(x)

class Decoder(nn.Module):
    def __init__(self,input_dim):
        super().__init__()
        num_layers = compute_num_layers(input_dim)
        start = input_dim // (2 ** num_layers)
        self.layers = nn.Sequential()
        for i in range(num_layers-1):
            self.layers.append(nn.Linear(start,start*2))
            self.layers.append(nn.ReLU())
            start*=2

        self.layers.append(nn.Linear(start,input_dim))
    
    def forward(self,x):
        return self.layers(x)
    
class AutoEncoder(nn.Module):
    def __init__(self,input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.encoder = Encoder(input_dim)
        self.decoder = Decoder(input_dim)
    
    def forward(self,x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def compute_num_layers(input_size):
    return int(log2(input_size))