import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from PIL import Image
from pipelines.image_encoder import extract_image_feature

img = Image.open("demo_data/image/ComFoodID_001.jpg").convert("RGB")

feat = extract_image_feature(img)

print("Feature shape:", feat.shape)
