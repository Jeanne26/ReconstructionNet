from model import ReconstructionNet
import torch
import os
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
from data_isic import ISICDataset   # là où ta classe est définie
from torchvision import transforms
from pathlib import Path
import argparse
import yaml
from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config_isic.yaml', help='Path to the config file')
args = parser.parse_args()

config_name = args.config
print("config : ", config_name)
CONFIG_PATH = '../configs/'

def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)
    return config

config = load_config(config_name)



#log path de la forme ../logs/log_name_X
log_name = config["log_name"]
log_folder = config["log_folder"]
num_file = 0
while os.path.exists(os.path.join(log_folder, f"{log_name}_{num_file}")):
    num_file += 1
log_name = f"{log_name}_{num_file}"
writer = SummaryWriter(log_dir=os.path.join(log_folder, log_name))



path = os.path.join(config["data_folder"], config["data_name"])
data = torch.load(path, weights_only=False)

beta = config["beta"]
is_image = config["is_image"]
lr_recNet =config["lr_recNet"]
weight_decay = config["weight_decay"]
num_epochs_recNet = config["num_epochs_recNet"]
batch_size = config["batch_size"]
num_classes = config["num_classes"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

path_data = Path("../data/isic")

train_dataset = ISICDataset(path_data/"train", path_data/"train/groundtruth.csv", transform)
val_dataset   = ISICDataset(path_data/"val",   path_data/"val/groundtruth.csv",   transform)
test_dataset  = ISICDataset(path_data/"test",  path_data/"test/groundtruth.csv",  transform)


import torchvision.models as models
import torch.nn as nn

def get_resnet18():
    resnet = models.resnet18(weights="IMAGENET1K_V1")
    resnet.fc = nn.Identity()   # remove classifier
    for p in resnet.parameters():
        p.requires_grad = False
    return resnet


from torch.utils.data import DataLoader

model = get_resnet18().to(device)
def extract_features(dataset, batch_size=32):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()

    feats, labels = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            f = model(x)
            feats.append(f.cpu())
            labels.append(y)

    return torch.cat(feats), torch.cat(labels)

X_train, y_train = extract_features(train_dataset)
X_val, y_val     = extract_features(val_dataset)
X_test, y_test   = extract_features(test_dataset)

print(X_train.shape)

def class_mse_loss(x, model, y):
    total = 0.0
    N = x.size(0)

    for j in range(model.num_class):
        mask = (y == j)
        if mask.sum() == 0:
            continue

        x_j = x[mask]
        recon_j = model.autoencoders[j](x_j)

        mse_j = torch.mean((recon_j - x_j) ** 2)
        total += mask.sum() * mse_j

    return total / N


beta = 2.0  # diabetes (dapres l'article beta = 2.0 pour diabetes apres le grid search)

is_image = False #change depending on data
num_classes = len(torch.unique(y_train))

input_dim = X_train.shape[1] 
dataset = TensorDataset(X_train, y_train)

loader_recNet = DataLoader(dataset, batch_size=64, shuffle=True) #essayer avec batch size 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lr_recNet =0.001
weight_decay = 1e-5  #d'apres l'arcticle the lower the better
rec_Net = ReconstructionNet(input_dim,num_classes, is_image).to(device)# # Make sur of is_image at the top of this file !!
optimizer_recNet = optim.Adam(rec_Net.parameters(), lr=lr_recNet, weight_decay=weight_decay)
criterion = nn.CrossEntropyLoss()

num_epochs_recNet = 20

for ep in range(num_epochs_recNet):
    rec_Net.train()
    train_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch, labels in loader_recNet:
        batch = batch.to(device)
        labels = labels.to(device)

        optimizer_recNet.zero_grad()

        _, _, wre = rec_Net(batch)
        logits = (-wre)

        ce_loss = criterion(logits, labels)
        mse_loss = class_mse_loss(batch, rec_Net, labels)

        loss = ce_loss + beta * mse_loss
        loss.backward()
        optimizer_recNet.step()

        train_loss += loss.item()

        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    avg_loss = train_loss / len(loader_recNet)
    accuracy = 100 * total_correct / total_samples

    print(f"Epoch {ep+1:02d} | Loss: {avg_loss:.4f} | Acc: {accuracy:.2f}%")


#test 

def evaluate_recnet(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            logits, _, _ = model(x)
            preds = torch.argmax(logits, dim=1)

            all_preds.append(preds)
            all_labels.append(y)

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    acc = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    return acc, report, cm


test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64)

acc, report, cm = evaluate_recnet(rec_Net, test_loader, device)
print("Test Acc:", acc)

#Test Accuracy: 0.8939

#save the model 
#model_path = os.path.join('../models', 'recnet_diabetes.pt')
#torch.save(rec_Net.state_dict(), model_path)

