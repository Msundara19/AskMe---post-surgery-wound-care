import numpy as np

def segment_wound(image: np.ndarray) -> np.ndarray:
    # Dummy segmentation: center square
    h, w, _ = image.shape
    mask = np.zeros((h, w), dtype="uint8")
    h1, h2 = h // 4, 3 * h // 4
    w1, w2 = w // 4, 3 * w // 4
    mask[h1:h2, w1:w2] = 1
    return mask
