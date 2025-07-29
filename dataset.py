from datasets import load_dataset
from torch.utils.data import random_split, ConcatDataset, Dataset, DataLoader
from typing import Callable
from PIL.Image import Image
import random
import torch
from torchvision.transforms import v2


flickr30k = load_dataset("nlphuji/flickr30k")['test'] # only has test split for whatever reason...
flickr30k_train, flickr30k_test = random_split(flickr30k, [0.8, 0.2])
places365 = load_dataset("Andron00e/Places365-custom")['train']   
places365_train, places365_test = random_split(places365, [0.8, 0.2])

rotation_to_label = {
    0: 0,
    90: 1,
    180: 2,
    270: 3
}
rotation_label_to_text = {
    0: "UPRIGHT",
    1: "LEFT",
    2: "UPSIDE_DOWN",
    3: "RIGHT",
}
                        
class RandomRotationDataset(Dataset):
    def __init__(self, 
                 dataset: Dataset, 
                 xform: v2.Transform, 
                 rotation_sample_weights: dict[int, float] | None = None):
        super().__init__()
        self.dataset = dataset
        self.xform = xform
        if rotation_sample_weights is not None:
            self.rotations = list(rotation_sample_weights.keys())
            self.rotation_weights = list(rotation_sample_weights.values())
        else:
            self.rotations = [0, 90, 180, 270]
            self.rotation_weights = [0.25, 0.25, 0.25, 0.25]

    def __getitem__(self, idx: int):
        image = self.dataset[idx]['image']
        rotation = random.choices(
            population=self.rotations,
            weights=self.rotation_weights,
            k=1
        )[0]
        image = image.rotate(rotation)
        image = self.xform(image)
        return {
            "image": image,
            "rotation": rotation,
            "rotation_label": rotation_to_label[rotation]
        }
    
    def __len__(self):
        return len(self.dataset)
    

xform = v2.Compose([
    v2.ToImage(),
    v2.RandomResizedCrop((224, 224), antialias=True, ratio=(1,1)),
    v2.ToDtype(dtype=torch.float32, scale=True),
    # Apparently these are the distribution stats for ImageNet, on which MobileNet is trained, 
    # so let's bring our images into this distribution.
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_train_dataset(oversample: int = 1, rotation_sample_weights: dict[int, int] | None = None) -> Dataset:
    datasets = []
    for i in range(oversample):
        datasets.append(RandomRotationDataset(flickr30k_train, xform=xform, rotation_sample_weights=rotation_sample_weights))
        datasets.append(RandomRotationDataset(places365_train, xform=xform, rotation_sample_weights=rotation_sample_weights))
    return ConcatDataset(datasets)

def get_test_dataset(oversample: int = 1, rotation_sample_weights: dict[int, int] | None = None) -> Dataset:
    datasets = []
    for i in range(oversample):
        datasets.append(RandomRotationDataset(flickr30k_test, xform=xform, rotation_sample_weights=rotation_sample_weights))
        datasets.append(RandomRotationDataset(places365_test, xform=xform, rotation_sample_weights=rotation_sample_weights))
    return ConcatDataset(datasets)

# Visualization of the first 25 images in ds in a 5x5 grid
import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    # img: torch.Tensor, shape [C, H, W], normalized
    img = img.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    return img

if __name__ == "__main__":
    ds = get_test_dataset(oversample=1)
    fig, axes = plt.subplots(5, 5, figsize=(12, 12))
    for i in range(25):
        sample = ds[i]
        img = sample["image"]
        label = rotation_label_to_text[sample["rotation_label"]]
        ax = axes[i // 5, i % 5]
        ax.imshow(imshow(img))
        ax.set_title(f"Label: {label}")
        ax.axis('off')
    plt.tight_layout()
    plt.show()












