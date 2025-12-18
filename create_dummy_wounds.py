from pathlib import Path
from PIL import Image, ImageDraw

base = Path("data") / "sample_wounds"
base.mkdir(parents=True, exist_ok=True)

def make_image(filename, color, text):
    img = Image.new("RGB", (256, 256), color)
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), text, fill=(255, 255, 255))
    img.save(base / filename)

make_image("wound_01.png", (200, 50, 50), "wound 1")
make_image("wound_02.png", (50, 200, 50), "wound 2")
make_image("wound_03.png", (50, 50, 200), "wound 3")

print("Dummy wound images created in data/sample_wounds")
