import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from dataset import get_train_dataset

class RotationClassfier(nn.Module):
    def __init__(self):
        super().__init__()
        self.efficientnet_v2 = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
        self.efficientnet_v2.classifier = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(in_features=1280, out_features=2048, bias=True),
            nn.ReLU6(),
            nn.Linear(in_features=2048, out_features=4, bias=True)
        )

    def forward(self, x: torch.Tensor):
        efficientnet_out = self.efficientnet_v2(x)
        return self.classifier(efficientnet_out)

if __name__ == "__main__":
    ds = get_train_dataset()
    rc = RotationClassfier()
    img = ds[0]['image']
    img = img[None, ...]
    out = rc(img)
    print(out.shape)

