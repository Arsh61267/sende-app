import torch
import numpy as np
import joblib
from pipelines.image_encoder import extract_image_feature


# =========================
# DEVICE
# =========================

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# MODEL STRUCTURE — SAMA TRAINING
# =========================

_model = torch.nn.Sequential(
    torch.nn.Linear(2816, 1024),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.3),
    torch.nn.Linear(1024, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 10)
).to(_device)

state = torch.load(
    "models/best_fusion_mlp_logvita.pt",
    map_location=_device
)

_model.load_state_dict(state)
_model.eval()


# =========================
# LOAD TARGET SCALER (LOG VITA)
# =========================

y_scaler = joblib.load("models/y_scaler_log_vita.pkl")


# =========================
# LIGHT TEXT ENCODER (768 dim)
# =========================

def simple_text_encoder(text):

    if text is None or str(text).strip() == "":
        return torch.zeros(768)

    vec = np.zeros(768)
    words = str(text).lower().split()

    for w in words:
        h = abs(hash(w)) % 768
        vec[h] += 1

    vec = vec / (np.linalg.norm(vec) + 1e-6)

    return torch.tensor(vec, dtype=torch.float32)


# =========================
# PREDICT — TRAINING CONSISTENT
# =========================

@torch.no_grad()
def predict_fusion_eval(pil_image, recipe_text=None):

    # image feature
    img_feat = extract_image_feature(pil_image)

    # text feature
    text_feat = simple_text_encoder(recipe_text)

    # fusion
    fusion = torch.cat([img_feat, text_feat], dim=0)
    fusion = fusion.unsqueeze(0).to(_device)

    # model output (scaled space)
    pred_scaled = _model(fusion).cpu().numpy()

    # inverse scaler → original target space
    pred = y_scaler.inverse_transform(pred_scaled)

    # vitamin A index = 5 → inverse log1p
    pred[:, 5] = np.clip(pred[:, 5], -20, 20)
    pred[:, 5] = np.expm1(pred[:, 5])

    return pred.squeeze(0)
