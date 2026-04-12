import gdown
import os

def download_all_models():
    os.makedirs("models", exist_ok=True)

    files = {
        "best_fusion_mlp_logvita.pt": "1RQIKT3utg_rmUNNtn3al1RC6GQmSjs_f",
        "best_img_mlp.pt": "1b90jvPYJqFLXEq_mArofgwVsX1aSLfHs",
        "best_img_mlp_logvita.pt": "1SJ_XoJ-XI2zrhxZy_6NgfxUaovUObesN",
        "reproduce_best_fusion.pt": "1UMco7Al-Cs5YMnH1DkNrt3--HJ-si1cg"
    }

    for name, file_id in files.items():
        path = f"models/{name}"

        if not os.path.exists(path):
            print(f"Downloading {name}...")
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, path, quiet=False)
        else:
            print(f"{name} already exists.")
