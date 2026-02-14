import torch
import torch.nn as nn
from math import log2

class ReconstructionNet(nn.Module):
    def __init__(self,input_dim,num_class,is_image):
        super().__init__()
        self.input_dim= input_dim
        self.num_class = num_class
        self.is_image = is_image
        
        #si on gere des images on utilise un encoder commun et des decodeurs propres à chaque classe
        if self.is_image:
            c, h, w = input_dim
            self.flat_dim = c * h * w
            self.encoder = EncoderImage(input_dim)

            self.decoders = nn.ModuleList([DecoderImage(input_dim) for _ in range(num_class)])     
        #si cas tablaire on utilise des autoencoders séparés pour chaque classe 
        else:
            self.flat_dim = input_dim
            self.autoencoders = nn.ModuleList([AutoEncoder(input_dim) for _ in range(num_class)])

        #couche de poids pour wre
        self.weights = nn.Parameter(torch.ones(num_class, self.flat_dim))

    def forward(self,x):
        #reconstructed list
        r_list = []
        #weighted reconstruction error list
        wre_list = []
        
        #si image on encode une fois et on decode pour chaque classe 
        if self.is_image:
            z = self.encoder(x)

            for j in range(self.num_class):
                
                reconstructed = self.decoders[j](z)
                reconstruction_error = (reconstructed - x)**2
                r_list.append(reconstructed)

                error_flat = reconstruction_error.view(x.size(0),-1)

                weighted_error = torch.sum(
                    error_flat * self.weights[j],
                    dim=1,
                    keepdim=True
                )
                wre_list.append(weighted_error)


        #pour les données tab j'ai gardé la meme chose
        else:   
            for i in range(self.num_class):
                x_copy = x.clone() 
                reconstructed = self.autoencoders[i](x_copy)
                reconstruction_error = (reconstructed-x_copy)**2 # removed sqrt

                #weighted_reconstruction_error = self.dense_layers[i](reconstruction_error)
                r_list.append(reconstruction_error)

                error_flat = reconstruction_error

                weighted_error = torch.sum(error_flat * self.weights[i], dim=1, keepdim=True)
                wre_list.append(weighted_error)
        
        wre = torch.cat(wre_list, dim=1)  # batch_size , num_class

        logits = -wre 
        return logits, r_list, wre

        



class EncoderImage(nn.Module):
    def __init__(self,input_dim):
        super().__init__()
        c, h, w = input_dim

        #encoder pour 
        self.model = nn.Sequential(
            nn.Conv2d(c, 8, 3, stride=2,padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2,padding=1),
            nn.ReLU()
        )

    def forward(self,x):
        return self.model(x)

class DecoderImage(nn.Module):
    def __init__(self,input_dim):
        super().__init__()
        c,h,w = input_dim
        self.model = nn.Sequential(
            nn.ConvTranspose2d(16, 8, 3, stride=2,padding=1,output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, c, 3, stride=2,padding=1,output_padding=1),
            nn.Sigmoid()
            )
        
    def forward(self,x):
        return self.model(x)
    
#ancion encoder fonctionne pour tabulaire (et les images aussi mais on preferea l'encoder specifique EncoderImage)
class Encoder(nn.Module):
    def __init__(self,input_dim):
        super().__init__()
        if isinstance(input_dim, tuple):
            self.encoder = EncoderImage(input_dim)
        else:
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
        start = input_dim // (2**num_layers)
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


# class ConvAutoEncoder(nn.Module):
#     def __init__(self, input_shape):
#         super().__init__()
#         c, h, w = input_shape

#         self.encoder = nn.Sequential(
#             nn.Conv2d(c, 16, 3, stride=2, padding=1), nn.ReLU(), nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(), nn.Conv2d(32, 64, 7))

#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(64, 32, 7),nn.ReLU(),nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), nn.ReLU(), nn.ConvTranspose2d(16, c, 3, stride=2, padding=1, output_padding=1),nn.Sigmoid())

#     def forward(self, x):
#         encoded = self.encoder(x)
#         return self.decoder(encoded)
 





