from pathlib import Path
from typing import Tuple
import cv2
import numpy as np

DEFAULT_SIZE: Tuple[int, int] = (256, 256)

def load_and_preprocess_image(path: Path, size: Tuple[int, int] = DEFAULT_SIZE) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    img = img.astype("float32") / 255.0
    return img
