"""
Qualitative results with uncertainty explanation (ζ) for ALL 3 types:

  (a) High Aleatoric  — correct but ambiguous (Def. 3, Eq. 13)
      → ζ shows WHICH PIXELS make the image ambiguous between classes

  (b) High Distributional — all probabilities low (Def. 4, Eq. 14)
      → ζ shows WHICH PIXELS are poorly reconstructed (OOD features)

  (c) Misclassified — wrong prediction
      → ζ shows WHICH PIXELS caused the wrong prediction

  In all cases:  ζ(x) = ω_ŷ ⊙ e_ŷ  (Definition 5)
  The explanation is a GENERAL tool that works for any image, not just misclassified ones.
"""

from model import ReconstructionNet
import torch
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import yaml
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

config_name = "config_mnist.yaml"
CONFIG_PATH = '../configs/'
model_weights_path = "../models/model_minst_9.pth"
TOP_K = 30              
ERROR_THRESHOLD = 0.3

def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        return yaml.safe_load(file)

config = load_config(config_name)

path = os.path.join(config["data_folder"], config["data_name"])
data = torch.load(path, weights_only=False)
X_test = data['X_test']
y_test = data['y_test']

is_image = config["is_image"]
num_classes = config["num_classes"]
input_dim = X_test.shape[1:]  # (c, h, w)
img_size = X_test.shape[-1]   # 28

model = ReconstructionNet(input_dim, num_classes, is_image)
model.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu'), weights_only=False))
model.eval()

dataset_name = config["data_name"].replace("_processed.pt", "").upper()

loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64, shuffle=False)

all_images = []
all_labels = []
all_preds = []
all_probs = []
all_wre = []
all_r_list = []
all_entropy = []
all_dist_unc = []

with torch.no_grad():
    for x, y in loader:
        logits, r_list, wre = model(x)
        probs = F.softmax(-wre, dim=1)
        preds = torch.argmax(logits, dim=1)

        log_probs = torch.log(probs + 1e-10)
        entropy = -(probs * log_probs).sum(dim=1)

        dist_unc = wre[torch.arange(x.size(0)), preds]

        for i in range(x.size(0)):
            all_images.append(x[i])
            all_labels.append(y[i].item())
            all_preds.append(preds[i].item())
            all_probs.append(probs[i].numpy())
            all_wre.append(wre[i].numpy())
            all_entropy.append(entropy[i].item())
            all_dist_unc.append(dist_unc[i].item())
            recons = {c: r_list[c][i] for c in range(num_classes)}
            all_r_list.append(recons)

all_entropy = np.array(all_entropy)
all_dist_unc = np.array(all_dist_unc)
all_labels = np.array(all_labels)
all_preds = np.array(all_preds)
all_probs = np.array(all_probs)

correct_mask = all_preds == all_labels
print(f"Dataset: {dataset_name} — {len(all_images)} test samples")
print(f"Accuracy: {correct_mask.mean()*100:.2f}%")


correct_idx = np.where(correct_mask)[0]
sorted_aleatoric = correct_idx[np.argsort(-all_entropy[correct_idx])]
top_aleatoric = sorted_aleatoric[:TOP_K]

print(f"\n─── High Aleatoric (correct but ambiguous) ───")
for idx in top_aleatoric:
    t2 = np.argsort(-all_probs[idx])[:2]
    print(f"  idx={idx:4d}  label={all_labels[idx]}  pred={all_preds[idx]}  "
          f"entropy={all_entropy[idx]:.4f}  "
          f"p({t2[0]})={all_probs[idx][t2[0]]:.3f}  p({t2[1]})={all_probs[idx][t2[1]]:.3f}")

fig_a, axes_a = plt.subplots(TOP_K, 5, figsize=(20, 3.5 * TOP_K))
if TOP_K == 1:
    axes_a = axes_a[np.newaxis, :]
fig_a.suptitle("High Aleatoric Uncertainty — ζ explains which pixels cause ambiguity",
               fontsize=14, fontweight='bold', y=1.02)

for i, idx in enumerate(top_aleatoric):
    img = all_images[idx]
    label = all_labels[idx]
    pred = all_preds[idx]
    probs = all_probs[idx]

    axes_a[i, 0].imshow(img.reshape(img_size, img_size), cmap='gray')
    axes_a[i, 0].set_title(f"True: {label}, Pred: {pred}")
    axes_a[i, 0].axis('off')

    colors = ['#d32f2f' if c == pred else '#1976d2' if c == label else '#bdbdbd'
              for c in range(num_classes)]
    axes_a[i, 1].barh(range(num_classes), probs, color=colors)
    axes_a[i, 1].set_yticks(range(num_classes))
    axes_a[i, 1].set_xlabel('Probability')
    axes_a[i, 1].set_title(f"σ_aleatoric = {all_entropy[idx]:.3f}")
    axes_a[i, 1].set_xlim(0, 1)

    with torch.no_grad():
        reconstruction = all_r_list[idx][pred]
        squared_error = (reconstruction - img) ** 2
        error_weights = model.weights[pred].view_as(squared_error)
        wre_map = error_weights * squared_error
    wre_map_sq = wre_map.squeeze()
    minr, maxr = wre_map_sq.min(), wre_map_sq.max()
    wre_norm = (wre_map_sq - minr) / (maxr - minr) if maxr - minr > 1e-10 else torch.zeros_like(wre_map_sq)
    mask = (wre_norm > ERROR_THRESHOLD).float()

    axes_a[i, 2].imshow(img.reshape(img_size, img_size), cmap='gray')
    axes_a[i, 2].imshow(mask.reshape(img_size, img_size), cmap='Reds', alpha=0.6, vmin=0, vmax=1)
    axes_a[i, 2].set_title(f"Explanation")
    axes_a[i, 2].axis('off')

    axes_a[i, 3].imshow(all_r_list[idx][label].reshape(img_size, img_size), cmap='gray')
    axes_a[i, 3].set_title(f"Recons (ae {label})")
    axes_a[i, 3].axis('off')

    axes_a[i, 4].imshow(all_r_list[idx][pred].reshape(img_size, img_size), cmap='gray')
    axes_a[i, 4].set_title(f"Recons (ae {pred})")
    axes_a[i, 4].axis('off')

fig_a.tight_layout()
fig_a.savefig(f"../figures/qualitative_aleatoric_{dataset_name.lower()}.png", dpi=150, bbox_inches='tight')
print(f" ../figures/qualitative_aleatoric_{dataset_name.lower()}.png")


# Find samples where max probability is lowest (all classes have low probs)
max_probs = np.max(all_probs, axis=1)
sorted_dist = np.argsort(max_probs)  # ascending - lowest max prob first
top_distributional = sorted_dist[:TOP_K]

print(f"\n─── High Distributional (all probs low) ───")
for idx in top_distributional:
    print(f"  idx={idx:4d}  label={all_labels[idx]}  pred={all_preds[idx]}  "
          f"σ_dist={all_dist_unc[idx]:.4f}  max_prob={all_probs[idx].max():.4f}")

fig_b, axes_b = plt.subplots(TOP_K, 4, figsize=(14, 3.5 * TOP_K))
if TOP_K == 1:
    axes_b = axes_b[np.newaxis, :]
fig_b.suptitle("High Distributional Uncertainty — ζ explains which pixels are OOD",
               fontsize=14, fontweight='bold', y=1.02)

for i, idx in enumerate(top_distributional):
    img = all_images[idx]
    label = all_labels[idx]
    pred = all_preds[idx]
    probs = all_probs[idx]
    
    p_true = probs[label]
    p_pred = probs[pred]

    axes_b[i, 0].imshow(img.reshape(img_size, img_size), cmap='gray')
    axes_b[i, 0].set_title(f"True: {label} / Pred: {pred}")
    axes_b[i, 0].axis('off')

    with torch.no_grad():
        reconstruction = all_r_list[idx][pred]
        squared_error = (reconstruction - img) ** 2
        error_weights = model.weights[pred].view_as(squared_error)
        wre_map = error_weights * squared_error
    wre_map_sq = wre_map.squeeze()
    minr, maxr = wre_map_sq.min(), wre_map_sq.max()
    wre_norm = (wre_map_sq - minr) / (maxr - minr) if maxr - minr > 1e-10 else torch.zeros_like(wre_map_sq)
    mask = (wre_norm > ERROR_THRESHOLD).float()

    axes_b[i, 1].imshow(img.reshape(img_size, img_size), cmap='gray')
    axes_b[i, 1].imshow(mask.reshape(img_size, img_size), cmap='Reds', alpha=0.6, vmin=0, vmax=1)
    axes_b[i, 1].set_title(f"Explanation")
    axes_b[i, 1].axis('off')

    axes_b[i, 2].imshow(all_r_list[idx][label].reshape(img_size, img_size), cmap='gray')
    axes_b[i, 2].set_title(f"Recons (ae {label})\np={p_true:.3f}")
    axes_b[i, 2].axis('off')

    axes_b[i, 3].imshow(all_r_list[idx][pred].reshape(img_size, img_size), cmap='gray')
    axes_b[i, 3].set_title(f"Recons (ae {pred})\np={p_pred:.3f}")
    axes_b[i, 3].axis('off')

fig_b.tight_layout()
fig_b.savefig(f"../figures/qualitative_distributional_{dataset_name.lower()}.png", dpi=150, bbox_inches='tight')
print(f"/figures/qualitative_distributional_{dataset_name.lower()}.png")


misclassified_idx = np.where(~correct_mask)[0]
sorted_misclass = misclassified_idx[np.argsort(-all_entropy[misclassified_idx])]
top_explanation = sorted_misclass[:TOP_K]

print(f"\n─── Misclassified (uncertainty explanation) ───")
for idx in top_explanation:
    print(f"  idx={idx:4d}  label={all_labels[idx]}  pred={all_preds[idx]}  "
          f"entropy={all_entropy[idx]:.4f}  σ_dist={all_dist_unc[idx]:.4f}")

fig_c, axes_c = plt.subplots(TOP_K, 4, figsize=(14, 3.5 * TOP_K))
if TOP_K == 1:
    axes_c = axes_c[np.newaxis, :]
fig_c.suptitle("Misclassified — ζ explains which pixels caused wrong prediction",
               fontsize=14, fontweight='bold', y=1.02)

for i, idx in enumerate(top_explanation):
    img = all_images[idx]
    label = all_labels[idx]
    pred = all_preds[idx]
    probs = all_probs[idx]
    
    p_true = probs[label]
    p_pred = probs[pred]

    axes_c[i, 0].imshow(img.reshape(img_size, img_size), cmap='gray')
    axes_c[i, 0].set_title(f"True: {label} / Pred: {pred}")
    axes_c[i, 0].axis('off')

    with torch.no_grad():
        reconstruction = all_r_list[idx][pred]
        squared_error = (reconstruction - img) ** 2
        error_weights = model.weights[pred].view_as(squared_error)
        wre_map = error_weights * squared_error

    wre_map = wre_map.squeeze()
    minr, maxr = wre_map.min(), wre_map.max()
    if maxr - minr > 1e-10:
        wre_norm = (wre_map - minr) / (maxr - minr)
    else:
        wre_norm = torch.zeros_like(wre_map)

    mask = (wre_norm > ERROR_THRESHOLD).float()

    axes_c[i, 1].imshow(img.reshape(img_size, img_size), cmap='gray')
    axes_c[i, 1].imshow(mask.reshape(img_size, img_size), cmap='Reds', alpha=0.6, vmin=0, vmax=1)
    axes_c[i, 1].set_title(f"Explanation")
    axes_c[i, 1].axis('off')

    axes_c[i, 2].imshow(all_r_list[idx][label].reshape(img_size, img_size), cmap='gray')
    axes_c[i, 2].set_title(f"Recons (ae {label})\np={p_true:.3f}")
    axes_c[i, 2].axis('off')

    axes_c[i, 3].imshow(all_r_list[idx][pred].reshape(img_size, img_size), cmap='gray')
    axes_c[i, 3].set_title(f"Recons (ae {pred})\np={p_pred:.3f}")
    axes_c[i, 3].axis('off')

fig_c.tight_layout()
fig_c.savefig(f"../figures/qualitative_explanation_{dataset_name.lower()}.png", dpi=150, bbox_inches='tight')
print(f" ../figures/qualitative_explanation_{dataset_name.lower()}.png")

