import numpy as np

def postprocess_mask(mask: np.ndarray, min_area: int = 10) -> np.ndarray:
    # Very simple: just ensure binary mask
    return (mask > 0).astype("uint8")
