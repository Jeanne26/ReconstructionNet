from model import ReconstructionNet
import torch
import os
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import yaml
import argparse
from torch.utils.tensorboard import SummaryWriter
from figure_1b import make_figure
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import random


#exemple pour lancer le code: python3 train_test_diabetes.py  --config config_diabetes.yaml

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config_mnist.yaml', help='Path to the config file')
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()
seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
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
data = torch.load(path)
X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train']
y_test = data['y_test']

def class_mse_loss(x, r_list, y, num_class):
    total = 0.0
    N = x.size(0)

    for j in range(num_class):
        mask = (y == j)
        if mask.sum() == 0:
            continue

        x_j = x[mask]
        recon_j = r_list[j][mask]
        x_j = x[mask]
        

        mse_j = torch.mean((recon_j - x_j) ** 2)
        total += mask.sum() * mse_j

    return total / N

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

beta = config["beta"]
is_image = config["is_image"]
lr_recNet =config["lr_recNet"]
weight_decay = config["weight_decay"]
num_epochs_recNet = config["num_epochs_recNet"]
batch_size = config["batch_size"]
num_classes = config["num_classes"]

if is_image:
    input_dim = X_train.shape[1:] #(c,h,w)
else: 
    input_dim = X_train.shape[1]
dataset = TensorDataset(X_train, y_train)
loader_recNet = DataLoader(dataset, batch_size=batch_size, shuffle=True) #essayer avec batch size 32

rec_Net = ReconstructionNet(input_dim,num_classes, is_image).to(device)# # Make sur of is_image at the top of this file !!
optimizer_recNet = optim.Adam(rec_Net.parameters(), lr=lr_recNet, weight_decay=weight_decay)
criterion = nn.CrossEntropyLoss()

test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



#test 
def get_images(test_loader):
    #recupere pour chaque classe une instace dans le data loader pour qu'on puisse tracker l'evolution
    images_per_class = {}
    with torch.no_grad():
        for x, y in test_loader:
            for i in range(x.size(0)):
                label = y[i].item()
                if label not in images_per_class:
                    images_per_class[label] = x[i]
                if len(images_per_class) == num_classes:
                    break

            if len(images_per_class) == num_classes:
                break
    return images_per_class

images_per_class = get_images(test_loader=test_loader)

def evaluate_recnet(model, dataloader, device, step,plot_grid = False):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            logits, _, _ = model(x)
            preds = torch.argmax(logits, dim=1)

            all_preds.append(preds)
            all_labels.append(y)
            all_probs.append(logits.softmax(dim=1))

    all_preds = torch.cat(all_preds).cpu().numpy()
    all_labels = torch.cat(all_labels).cpu().numpy()
    all_probs = torch.cat(all_probs).cpu().numpy()

    acc = accuracy_score(all_labels, all_preds)
    if num_classes == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
    else:
        auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
    report = classification_report(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    
    if plot_grid and is_image:
        recons_per_class = {}
        for y in images_per_class.keys():
            x = images_per_class[y]
            x = x.to(device)
            probs,r_list,wre = model(x)
            #dans model modif r_list contient plus les erreurs de reconstruction mais directement les image reconstruites
            recons_per_class[y] = [r_list[c].cpu().detach() for c in range(num_classes)]
        make_figure(images_per_class,recons_per_class,num_classes)
        fig = plt.gcf()
        writer.add_figure('RecNet/reconstructions_grid', fig, global_step=step)
        plt.close(fig)

    return acc,auc, report, cm




for ep in range(num_epochs_recNet):
    rec_Net.train()
    train_loss = 0.0
    total_correct = 0
    total_samples = 0
    for batch, labels in loader_recNet:
        batch = batch.to(device)
        labels = labels.to(device)


        optimizer_recNet.zero_grad()

        logits, r_list, wre = rec_Net(batch)

        ce_loss = criterion(logits, labels)
        mse_loss = class_mse_loss(batch, r_list, labels, num_classes)

        loss = ce_loss + beta * mse_loss
        loss.backward()

        optimizer_recNet.step()

        train_loss += loss.item()

        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    avg_loss = train_loss / len(loader_recNet)
    accuracy = 100 * total_correct / total_samples
    writer.add_scalar('RecNet/train_loss', avg_loss, ep)
    writer.add_scalar('RecNet/train_accuracy', accuracy, ep)
    
    print(f"Epoch {ep+1:02d} | Loss: {avg_loss:.4f} | Acc: {accuracy:.2f}%")
    
    if (ep+1) % 5 == 0:
        test_acc, test_auc,test_report, test_cm = evaluate_recnet(rec_Net, test_loader, device,step =ep, plot_grid=True)
        writer.add_scalar('RecNet/test_accuracy_epoch', test_acc, ep)
        writer.add_scalar('RecNet/test_auc_epoch', test_auc, ep)
    
    
    




print("Test Accuracy:", test_acc)
print("\nClassification Report:\n", test_report)
print("\nConfusion Matrix:\n", test_cm)
writer.add_scalar('RecNet/test_accuracy', test_acc)
writer.add_text('RecNet/classification_report', test_report)
writer.add_text('RecNet/confusion_matrix', str(test_cm))



#Test Accuracy: 0.8939

#sauvegarde du modele
model_save_path = os.path.join(config["model_save_folder"], f'{config["model_save_name"]}_{num_file}.pth')
torch.save(rec_Net.state_dict(), model_save_path)
