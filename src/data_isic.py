
import pandas as pd
from pathlib import Path 
from torch.utils.data import Dataset, Subset
from PIL import Image
from torchvision import transforms
from collections import Counter
import os
import torch
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import DataLoader

CLASSES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()   #[0,255] → [0,1]
])


class ISICDataset(Dataset):
    def __init__(self, images_dir, groundtruth_csv, transform=None):
        self.images_dir = Path(images_dir)
        self.df = pd.read_csv(groundtruth_csv)
        self.transform = transform

        self.image_ids = self.df["image"].values
        self.labels = self.df[CLASSES].values.argmax(axis=1)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        label = self.labels[idx]
        img_path = self.images_dir / f"{image_id}.jpg"
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label


path_data = Path("../data/isic")

train_dataset = ISICDataset(
    images_dir=path_data / "train",  
    groundtruth_csv=path_data / "train" / "groundtruth.csv",
    transform=transform
)

val_dataset = ISICDataset(
    images_dir=path_data / "val",
    groundtruth_csv=path_data / "val" / "groundtruth.csv",
    transform=transform
)

test_dataset = ISICDataset(
    images_dir=path_data / "test", 
    groundtruth_csv=path_data / "test" / "groundtruth.csv",
    transform=transform
)

print("Datasets loaded")
print("Train:", len(train_dataset))
print("Val  :", len(val_dataset))
print("Test :", len(test_dataset))



print("Train distribution:", Counter(train_dataset.labels))
print("Val distribution  :", Counter(val_dataset.labels))
print("Test distribution :", Counter(test_dataset.labels))


def split_dataset_by_class(dataset):
    class_subsets = {}
    for c in range(len(CLASSES)):
        indices = [i for i, y in enumerate(dataset.labels) if y == c]
        class_subsets[c] = Subset(dataset, indices)
    return class_subsets

train_sets_by_class = split_dataset_by_class(train_dataset)
for c in train_sets_by_class:
    print(f"Class {CLASSES[c]}: {len(train_sets_by_class[c])} images")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_resnet18():
    resnet = models.resnet18(weights="IMAGENET1K_V1")
    resnet.fc = nn.Identity()   # remove classifier
    for p in resnet.parameters():
        p.requires_grad = False
    return resnet



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

fold = "../data/processed_data"
os.makedirs(fold, exist_ok=True)

data = {
    "X_train": X_train,
    "X_val": X_val,
    "X_test": X_test,
    "y_train": y_train,
    "y_val": y_val,
    "y_test": y_test
}

file_path = os.path.join(fold, "isic_processed.pt")
torch.save(data, file_path)