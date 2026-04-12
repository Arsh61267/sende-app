import torch
import numpy as np
from pipelines.image_encoder import extract_image_feature


# ==========================
# DEVICE
# ==========================

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================
# MODEL — IDENTIK TRAINING
# ==========================

_model = torch.nn.Sequential(

    torch.nn.Linear(2816, 1024),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.3),

    torch.nn.Linear(1024, 256),
    torch.nn.ReLU(),

    torch.nn.Linear(256, 10)

).to(_device)


# ==========================
# LOAD WEIGHT TRAINING
# ==========================

state = torch.load(
    "models/best_fusion_mlp_logvita.pt",
    map_location=_device
)

_model.load_state_dict(state)
_model.eval()


# ==========================
# TEXT FEATURE — runtime dummy
# ==========================

def dummy_text_feature():
    return torch.zeros(768)


# ==========================
# PREDICT
# ==========================

@torch.no_grad()
def predict_fusion(pil_image, text_feat=None):

    img_feat = extract_image_feature(pil_image)

    if text_feat is None:
        text_feat = dummy_text_feature()

    if isinstance(text_feat, np.ndarray):
        text_feat = torch.tensor(text_feat, dtype=torch.float32)

    fusion = torch.cat([img_feat, text_feat], dim=0)
    fusion = fusion.unsqueeze(0).to(_device)

    pred = _model(fusion)

    return pred.squeeze(0).cpu().numpy()
