import torch
from torchvision import models, transforms
from PIL import Image

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
_model.eval().to(_device)

_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])

FOOD_KEYWORDS = [
    "food","dish","plate","meal","rice","soup",
    "cake","fruit","vegetable","pizza","burger"
]


@torch.no_grad()
def is_food_image(pil_img, threshold=0.20):

    x = _transform(pil_img).unsqueeze(0).to(_device)
    out = _model(x)
    prob = torch.softmax(out, dim=1)[0]

    topk = torch.topk(prob, 5)

    from torchvision.models import ResNet50_Weights
    labels = ResNet50_Weights.DEFAULT.meta["categories"]

    hits = 0

    for idx in topk.indices:
        label = labels[idx]
        if any(k in label.lower() for k in FOOD_KEYWORDS):
            hits += 1

    return hits >= 1
