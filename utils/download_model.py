import os
import gdown

def download_all_models():
    folder_id = "1StjMaAPcdpYs4m6wflUufvowBwpGNKHc"
    output = "models"

    if not os.path.exists(output):
        os.makedirs(output)

    # download entire folder
    gdown.download_folder(
        id=folder_id,
        output=output,
        quiet=False,
        use_cookies=False
    )