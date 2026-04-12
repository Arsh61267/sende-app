import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ======================
# CONFIG
# ======================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ganti sesuai jumlah kelas menu Anda
NUM_CLASSES = 12   # ← UBAH jika perlu

CLASS_NAMES = [
    "Bubur Tim Ikan Lele",
    "Bubur Hati Ayam",
    "Bubur Ayam Wortel",
    "Puree Pisang Pepaya",
    "Bubur Kentang Ayam",
    "Bubur Soto Ayam",
    "Bubur Kacang Ayam",
    "Sup Daging Kacang",
    "Mie Kukus Telur",
    "Misoa Hati Ayam",
    "Nasi Tim Ayam",
    "Nasi Tim Ikan"
]

# ======================
# MODEL
# ======================

_model = models.resnet18(weights=None)
_model.fc = nn.Linear(_model.fc.in_features, NUM_CLASSES)

# load weight Anda
_model.load_state_dict(
    torch.load("models/menu_resnet.pt", map_location=DEVICE)
)

_model.to(DEVICE)
_model.eval()

# ======================
# TRANSFORM
# ======================

_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])

# ======================
# PREDICT
# ======================

@torch.no_grad()
def classify_menu(pil_img):

    x = _tf(pil_img).unsqueeze(0).to(DEVICE)

    logits = _model(x)
    prob = torch.softmax(logits, dim=1)

    conf, idx = torch.max(prob, dim=1)

    return CLASS_NAMES[idx.item()], float(conf.item())


# ======================
# FOOD DETECTOR (threshold)
# ======================

def is_food_image(conf, threshold=0.55):
    return conf >= threshold
