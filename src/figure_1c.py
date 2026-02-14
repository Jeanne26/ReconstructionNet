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

model_weights_path = "../models/model_minst_4.pth"
model = ReconstructionNet(input_dim,num_classes,is_image)
model.load_state_dict(torch.load(model_weights_path))
model.eval()

test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#get 3 misclassified images
num_misclassified = 3
images = {}
recons = {}
with torch.no_grad():
    for x, y in test_loader:
        probs, r_list, wre = model(x)
        print(x.shape,probs.shape, r_list[0].shape, wre.shape)
        preds = torch.argmax(probs, dim=1)
        for i in range(x.size(0)):
            label = y[i].item()
            pred_label = preds[i].item()
            if label != pred_label and label not in images:
                images[label] = x[i]
                recons[label] = r_list[y[i]]
            if len(images) == num_misclassified:
                break

        if len(images) == num_classes:
            break
        
fig, axes = plt.subplots(1,2)
axes[0].imshow(images[0].permute(1,2,0).numpy(), cmap='gray')
axes[0].set_title("Original Image")
axes[0].axis('off')

masked_recon = recons[0].cpu().numpy() * (recons[0].cpu().numpy() > 0.15) # thresholding the reconstruction error to visualize only the most important parts
axes[1].imshow(masked_recon.reshape(28,28), cmap='hot')
axes[1].set_title("Reconstruction Error (thresholded)")
axes[1].axis('off')

plt.tight_layout()
plt.savefig("../figures/tmp.png")