import pandas as pd
from pathlib import Path 
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from PIL import Image
from collections import defaultdict

ISIC_LABEL_MAP = {
    "Melanoma, NOS": "MEL",
    "Nevus": "NV",
    "Basal cell carcinoma": "BCC",
    "Solar or actinic keratosis": "AKIEC",
    "Seborrheic keratosis": "BKL",
    "Dermatofibroma": "DF",
    "Vascular lesion": "VASC"
}

def load_isic(images_dir):
    images_dir = Path(images_dir)
    metadata_path = images_dir/ "metadata.csv"
    df = pd.read_csv(metadata_path)

    vascular_mask = (
    df["diagnosis_2"].str.contains("Vascular", na=False) &
    df["diagnosis_3"].isna())
    
    if vascular_mask.any():
        #print("Found 'Vascular lesion' in diagnosis_2, mapping to 'VASC'")
        df.loc[vascular_mask, "diagnosis_3"] = "Vascular lesion"

    df = df[df["diagnosis_3"].isin(ISIC_LABEL_MAP.keys())]
    df["label"] = df["diagnosis_3"].map(ISIC_LABEL_MAP)

    image_paths = df["isic_id"].apply(lambda x: images_dir / f"{x}.jpg")

    return image_paths.tolist(), df["label"].tolist()


def encodage_labels(labels):
    le = LabelEncoder()
    encoded_labels = le.fit_transform(labels)
    return encoded_labels, le


class ISICDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels 
        self.transform  = transform 

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        img = Image.open(self.image_paths[index]).convert("RGB")
        label = self.labels[index]
        if self.transform:
            img = self.transform(img)
        return img, label



def split_dataset_by_class(image_paths, labels, transform):
    class_datasets = defaultdict(list)

    for img, lbl in zip(image_paths, labels):
        class_datasets[lbl].append((img, lbl))

    final_datasets = {}
    for c, data in class_datasets.items():
        imgs = [x[0] for x in data]
        lbls = [x[1] for x in data]
        final_datasets[c] = ISICDataset(imgs, lbls, transform)

    return final_datasets


if __name__ == "__main__":
    images_dir = "../data/isic_images"
    image_paths, labels = load_isic(images_dir)
    encoded_labels, le = encodage_labels(labels)

    print(f"Loaded {len(image_paths)} images.")
    print(f"labels: {set(labels)}")
    print(f"Encoded labels: {set(encoded_labels)}")