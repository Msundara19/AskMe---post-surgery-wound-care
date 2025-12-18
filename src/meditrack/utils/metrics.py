import numpy as np

def compute_wound_area(mask: np.ndarray, pixel_spacing_mm: float) -> float:
    pixel_area = pixel_spacing_mm ** 2
    return (mask > 0).sum() * pixel_area
