import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from PIL import Image
from pipelines.fusion_inference import predict_fusion


img = Image.open("demo_data/image/ComFoodID_001.jpg").convert("RGB")

out = predict_fusion(img)

print("Fusion output shape:", out.shape)
print(out)
