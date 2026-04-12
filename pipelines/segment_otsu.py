import cv2
import numpy as np
from PIL import Image


def otsu_segment(pil_img: Image.Image):
    """
    Return:
        masked_rgb (PIL)
        mask (PIL)
    """

    img = np.array(pil_img.convert("RGB"))
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    _, mask = cv2.threshold(
        blur, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # auto invert jika background putih dominan
    if np.sum(mask == 255) > np.sum(mask == 0):
        mask = cv2.bitwise_not(mask)

    mask3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    masked = cv2.bitwise_and(bgr, mask3)

    rgb_masked = cv2.cvtColor(masked, cv2.COLOR_BGR2RGB)

    return Image.fromarray(rgb_masked), Image.fromarray(mask)
