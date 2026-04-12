import torch
import torchvision.transforms as T
from torchvision import models
from PIL import Image

# =========================
# DEVICE
# =========================

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# RESNET50
# =========================

_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
_model.eval()
_model.to(_device)

# simpan target conv layer untuk CAM
_target_layer = _model.layer4[-1]

# =========================
# FEATURE EXTRACTOR VERSION
# =========================

_feat_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
_feat_model.fc = torch.nn.Identity()
_feat_model.eval()
_feat_model.to(_device)

# =========================
# TRANSFORM
# =========================

_transform = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# =========================
# FEATURE EXTRACTION
# =========================

@torch.no_grad()
def extract_image_feature(pil_img: Image.Image):

    x = _transform(pil_img).unsqueeze(0).to(_device)
    feat = _feat_model(x)
    feat = feat.squeeze(0)

    return feat.cpu()


# =========================
# CAM ACCESSORS
# =========================

def get_encoder_model():
    return _model


def get_target_layer():
    return _target_layer


def get_device():
    return _device


def get_cam_transform():
    return _transform
