from model import ReconstructionNet
import torch
import os
import matplotlib.pyplot as plt
import yaml
from torch.utils.data import DataLoader, TensorDataset

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

# model_weights_path = "../models/model_minst_4.pth"
# model = ReconstructionNet(input_dim,num_classes,is_image)
# model.load_state_dict(torch.load(model_weights_path))
# model.eval()

test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# #pour une image aleatoire
# with torch.no_grad():
#     for x, y in test_loader:
#         probs, r_list, wre = model(x)
#         fig, axes = plt.subplots(1, num_classes + 1, figsize=(15, 3))
#         axes[0].imshow(x[0].permute(1,2,0).numpy())
#         axes[0].set_title("Image")
#         axes[0].axis('off')
#         for i in range(num_classes):
#             axes[i+1].imshow(r_list[i][0].permute(1,2,0).numpy())
#             axes[i+1].set_title(f"Class {i}")
#             axes[i+1].axis('off')
#         plt.tight_layout()
#         plt.savefig("../figures/reconstructions/all_reconstructions.png")
#         break


# images_per_class = {}
# recons_per_class = {}
# #recupuration d'une image par classe
# with torch.no_grad():
#     for x, y in test_loader:
#         probs, r_list, wre = model(x)
#         for i in range(x.size(0)):
#             label = y[i].item()
#             if label not in images_per_class:
#                 images_per_class[label] = x[i]
#                 recons_per_class[label] = [torch.sqrt(r_list[c][i])+x[i] for c in range(num_classes)]
#             if len(images_per_class) == num_classes:
#                 break

#         if len(images_per_class) == num_classes:
#             break





def make_figure(images_per_class,recons_per_class,num_classes):
    fig, axes = plt.subplots(num_classes, num_classes+1,figsize=(2*(num_classes+1), 2*num_classes))
    for row in range(num_classes):
        image = images_per_class[row]
        axes[row,0].imshow(image.permute(1,2,0).cpu().numpy())
        axes[row,0].set_title(f"Label {row}")
        axes[row,0].axis('off')

        for col in range(num_classes):
            recons = recons_per_class[row][col]
            #parce que pour les images pas dans des batch le forward rajoute une dim donc on doit la retirer 
            recons = recons.squeeze(0) if recons.dim() == 4 else recons
            axes[row, col+1].imshow(
                recons.permute(1,2,0).cpu().numpy())
            axes[row,col+1].axis('off')

    plt.tight_layout()
    plt.savefig("../figures/all_reconstructions_grid.png")
    plt.show()


if __name__ == "__main__":
    make_figure(images_per_class,recons_per_class)