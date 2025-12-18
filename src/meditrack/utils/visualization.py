from pathlib import Path
import numpy as np
import cv2

def save_overlay(image: np.ndarray, mask: np.ndarray, output_path: Path):
    img = (image * 255).astype("uint8") if image.max() <= 1.0 else image.astype("uint8")
    mask_uint8 = (mask > 0).astype("uint8") * 255
    color_mask = cv2.applyColorMap(mask_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.7, color_mask, 0.3, 0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
