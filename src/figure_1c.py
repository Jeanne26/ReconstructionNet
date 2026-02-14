from model import ReconstructionNet
import torch
import os
import matplotlib.pyplot as plt
import yaml
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

config_name = "config_mnist.yaml"
CONFIG_PATH = '../configs/'

def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)
    return config

config = load_config(config_name)

path = os.path.join(config["data_folder"], config["data_name"]) 
data = torch.load(path)
X_test = data['X_test']
y_test = data['y_test']

is_image = config["is_image"] #!! change depending on data 
num_classes = config["num_classes"]
batch_size = config["batch_size"]

input_dim = X_test.shape[1:] #(c,h,w)

model_weights_path = "../models/model_minst_18.pth"
model = ReconstructionNet(input_dim,num_classes,is_image)
model.load_state_dict(torch.load(model_weights_path))
model.eval()

test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

#get 3 misclassified images
num_misclassified = 3
images = []

# recons = {}
with torch.no_grad():
    for x, y in test_loader:
        
        probs, r_list, wre = model(x)
        
        preds = torch.argmax(probs, dim=1)

            
        label = y[0].item()
        pred_label = preds[0].item()

        if label != pred_label:
            weighted_rec_err = wre[0][pred_label] * r_list[pred_label]
            dico = {"label": label, 
                    "pred_label": pred_label, 
                    "recons_label": r_list[label], 
                    "recons_pred": r_list[pred_label],
                    "weighted_rec_err":weighted_rec_err,
                    "image": x[0] }
            images.append(dico)
            if len(images) == num_misclassified:
                break

        if len(images) == num_misclassified:
            break
   

 
fig, axes = plt.subplots(len(images),4,figsize=(12, 5*len(images)))
for i in range(len(images)):
    label= images[i]["label"]
    pred_label= images[i]["pred_label"]
    recons_label = images[i]["recons_label"]
    recons_pred = images[i]["recons_pred"]
    wre = images[i]["weighted_rec_err"]
    image = images[i]["image"]

    axes[i,0].imshow(image.permute(1,2,0).numpy(), cmap='gray')
    axes[i,0].set_title(f"true: {label}, pred:{pred_label}", )
    axes[i,0].axis('off')

    minr = torch.min(wre)
    maxr = torch.max(wre)
    wre = (wre - minr)/(maxr-minr)
    mask_recons = torch.where(wre>0.7,0,1)

    axes[i,1].imshow(image.reshape(28,28), cmap='gray')
    axes[i,1].imshow(mask_recons.reshape(28,28), cmap = "Set1",alpha=0.5)
    axes[i,1].set_title("erreur recons")
    axes[i,1].axis('off')

    axes[i,2].imshow(recons_label.reshape(28,28), cmap='gray')
    axes[i,2].set_title(f"recons par ae {label}")
    axes[i,2].axis('off')
    
    axes[i,3].imshow(recons_pred.reshape(28,28), cmap='gray')
    axes[i,3].set_title(f"reconspar ae {pred_label}")
    axes[i,3].axis('off')
    
plt.tight_layout()
plt.savefig("../figures/tmp.png")
