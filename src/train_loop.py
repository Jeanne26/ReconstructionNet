from model import ReconstructionNet
import torch
import os
from torch.utils.data import DataLoader, TensorDataset, Subset
import torch.optim as optim
import torch.nn as nn
import yaml
import argparse

#exemple pour lancer le code: python3 train_loop.py  --config config_diabetes.yaml

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config_mnist.yaml', help='Path to the config file')
args = parser.parse_args()

config_name = args.config
print("config : ", config_name)
CONFIG_PATH = '../configs/'

def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)
    return config

config = load_config(config_name)


path = os.path.join(config["data_folder"], config["data_name"]) ## Testing first with MINST but it "should" work with the rest TODO test the other datasets
data = torch.load(path)
X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train']
y_test = data['y_test']


is_image = config["is_image"] #!! change depending on data 
num_classes = config["num_classes"]
if is_image:
    input_dim = X_train.shape[1:] #(c,h,w)
else: 
    input_dim = X_train.shape[1]
 

def get_best_device():
    """Returns the best device on this computer"""

    if torch.cuda.is_available():
        device = torch.device("cuda")
        total_memory = torch.cuda.get_device_properties(device).total_memory
        print(f"GPU Memory: {total_memory / 1e9:.1f} GB")
        print(f"GPU Name: {torch.cuda.get_device_name(device)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Found device: {device}")
    return device

device = get_best_device()
batch_size = config["batch_size"]

train_dataset = TensorDataset(X_train, y_train)
class_loaders = []
for i in range(num_classes):
    indices = (y_train == i).nonzero().squeeze() # Matching i with the classes
    subset = Subset(train_dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
    class_loaders.append(loader)

# Now that I have loader for each class I can train each one individually first 
rec_Net = ReconstructionNet(input_dim,num_classes,is_image).to(device)# # Make sur of is_image at the top of this file !!


loss_fn = nn.MSELoss()
# Same as the paper
lr = config["lr"]
weight_decay = config["weight_decay"]

num_epochs = config["num_epochs"]
for i in range(num_classes):
    model_i = rec_Net.autoencoders[i]
    model_i.train() 
    optimizer = optim.Adam(model_i.parameters(), lr=lr, weight_decay=weight_decay)
    print(f"Training AutoEncoder class {i}")

    for ep in range(num_epochs):
        train_loss =0

        for batch_id, (batch, labels) in enumerate(class_loaders[i]):
            batch = batch.to(device)
            optimizer.zero_grad()
            reconstructed = model_i(batch)
            loss = loss_fn(reconstructed, batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_loss = train_loss / len(class_loaders[i])
        print(f"  Epoch {ep+1}/{num_epochs} \t Loss: {avg_loss:.6f}")

# Now for the big model

loss_recNet = nn.CrossEntropyLoss()

dataset = TensorDataset(X_train, y_train)
loader_recNet = DataLoader(dataset, batch_size=batch_size, shuffle=True)
num_epochs_recNet = config["num_epochs_recNet"]
lr_recNet = config["lr_recNet"] #lr = 0.01 wasnt learning well got a loss of 0.3

optimizer_recNet = optim.Adam(rec_Net.parameters(), lr=lr_recNet, weight_decay=weight_decay)
print("RecNet:")
for ep in range(num_epochs_recNet):
    train_loss=0
    total_correct =0
    total_samples =0
    rec_Net.train()
    for batch_id, (batch, labels) in enumerate(loader_recNet):
        batch = batch.to(device)
        labels = labels.to(device)
        optimizer_recNet.zero_grad()
        _,_,wre = rec_Net(batch)
        logits=-wre
        loss=loss_recNet(logits,labels)
        loss.backward()
        preds = torch.argmax(logits, dim=1)
        correct = (preds == labels).sum().item()
        total_correct += correct
        total_samples += labels.size(0)
        optimizer_recNet.step()
        train_loss += loss.item()

    avg_loss = train_loss / len(loader_recNet)
    accuracy = 100 * total_correct / total_samples 
    print(f"  Epoch {ep+1} \t Loss: {avg_loss:.4f} \t Acc: {accuracy:.2f}%")



# Test

dataset_test = TensorDataset(X_test, y_test)
loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

total_correct = 0
total_samples = 0
rec_Net.eval() 
with torch.no_grad():
    for batch, labels in loader_test:
        batch = batch.to(device)
        labels = labels.to(device)
        _, _, wre = rec_Net(batch)
        logits = -wre  
        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

accuracy = 100 * total_correct / total_samples
print(f"Test Accuracy: {accuracy:.2f}%")

"""
Resultat Obtenue pour MNIST:

Training AutoEncoder class 0
  Epoch 1/5      Loss: 0.049036
  Epoch 2/5      Loss: 0.015562
  Epoch 3/5      Loss: 0.010501
  Epoch 4/5      Loss: 0.008525
  Epoch 5/5      Loss: 0.007491
Training AutoEncoder class 1
  Epoch 1/5      Loss: 0.067508
  Epoch 2/5      Loss: 0.065443
  Epoch 3/5      Loss: 0.065404
  Epoch 4/5      Loss: 0.063389
  Epoch 5/5      Loss: 0.045867
Training AutoEncoder class 2
  Epoch 1/5      Loss: 0.063902
  Epoch 2/5      Loss: 0.025484
  Epoch 3/5      Loss: 0.015102
  Epoch 4/5      Loss: 0.011201
  Epoch 5/5      Loss: 0.009239
Training AutoEncoder class 3
  Epoch 1/5      Loss: 0.056840
  Epoch 2/5      Loss: 0.017600
  Epoch 3/5      Loss: 0.010665
  Epoch 4/5      Loss: 0.008353
  Epoch 5/5      Loss: 0.007203
Training AutoEncoder class 4
  Epoch 1/5      Loss: 0.056900
  Epoch 2/5      Loss: 0.021565
  Epoch 3/5      Loss: 0.012002
  Epoch 4/5      Loss: 0.009014
  Epoch 5/5      Loss: 0.007555
Training AutoEncoder class 5
  Epoch 1/5      Loss: 0.057272
  Epoch 2/5      Loss: 0.022942
  Epoch 3/5      Loss: 0.013637
  Epoch 4/5      Loss: 0.009734
  Epoch 5/5      Loss: 0.007745
Training AutoEncoder class 6
  Epoch 1/5      Loss: 0.051610
  Epoch 2/5      Loss: 0.017728
  Epoch 3/5      Loss: 0.010519
  Epoch 4/5      Loss: 0.008187
  Epoch 5/5      Loss: 0.007066
Training AutoEncoder class 7
  Epoch 1/5      Loss: 0.038238
  Epoch 2/5      Loss: 0.010946
  Epoch 3/5      Loss: 0.007304
  Epoch 4/5      Loss: 0.005919
  Epoch 5/5      Loss: 0.005173
Training AutoEncoder class 8
  Epoch 1/5      Loss: 0.059120
  Epoch 2/5      Loss: 0.026709
  Epoch 3/5      Loss: 0.014871
  Epoch 4/5      Loss: 0.010990
  Epoch 5/5      Loss: 0.009225
Training AutoEncoder class 9
  Epoch 1/5      Loss: 0.068492
  Epoch 2/5      Loss: 0.035116
  Epoch 3/5      Loss: 0.016491
  Epoch 4/5      Loss: 0.010579
  Epoch 5/5      Loss: 0.008356
RecNet:
  Epoch 1        Loss: 0.1914    Acc: 96.30%
  Epoch 2        Loss: 0.0749    Acc: 98.17%
  Epoch 3        Loss: 0.0518    Acc: 98.62%
  Epoch 4        Loss: 0.0336    Acc: 99.03%
  Epoch 5        Loss: 0.0349    Acc: 98.98%
Test Accuracy: 97.70%

"""






