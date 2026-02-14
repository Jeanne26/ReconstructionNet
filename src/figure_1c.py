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

model_weights_path = "../models/model_minst_0.pth"
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

        print(label, pred_label)
        if label != pred_label:
            dico = {"label": label, "pred_label": pred_label, "recons": r_list[label], "image": x[0], "recons_by_pred": r_list[pred_label]}
            images.append(dico)
            if len(images) == num_misclassified:
                break

        if len(images) == num_misclassified:
            break
   

 
fig, axes = plt.subplots(len(images),4,figsize=(10, 10*len(images)))
for i in range(len(images)):
    image = images[i]["image"]
    recons = images[i]["recons"]
    pred_label   = images[i]["pred_label"]
    true_label   = images[i]["label"] 
    recons_by_pred = images[i]["recons_by_pred"]
    axes[i,0].imshow(image.permute(1,2,0).numpy(), cmap='gray')
    axes[i,0].set_title(f"true: {label}, pred:{pred_label}", fontsize = 30)
    axes[i,0].axis('off')

    mask_recons = torch.where(recons>0.5,0,1)

    axes[i,1].imshow(image.reshape(28,28), cmap='gray')
    axes[i,1].imshow(mask_recons.reshape(28,28), cmap='Set1', alpha=0.5)
    axes[i,1].set_title("erreur recons")
    axes[i,1].axis('off')

    axes[i,2].imshow(recons.reshape(28,28), cmap='gray')
    axes[i,2].set_title(f"recons par ae {label}")
    axes[i,2].axis('off')
    
    axes[i,3].imshow(recons_by_pred.reshape(28,28), cmap='gray')
    axes[i,3].set_title(f"reconspar ae {pred_label}")
    axes[i,3].axis('off')
    
plt.tight_layout()
plt.savefig("../figures/tmp.png")
