import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

fig_dir = Path("../vis_sep16/del")
files = sorted(fig_dir.glob("*.png"))

data = []
for f in files:
    parts = f.stem.split("_")
    if len(parts) >= 2:
        dataset = parts[0]
        model = parts[1]
        data.append((dataset, model, f))

datasets = ["UMLS", "KINSHIP", "NELL-995-h100"]
models = ["DistMult", "ComplEx", "TransE", "TransH", "MuRE", "RotatE", "Keci", "DeCaL"]

target_w, target_h = 600, 500

label_space_left = 150
label_space_top = 100

canvas_w = label_space_left + target_w * len(models)
canvas_h = label_space_top + target_h * len(datasets)

big_img = Image.new("RGB", (canvas_w, canvas_h), "white")
draw = ImageDraw.Draw(big_img)

try:
    font = ImageFont.truetype("DejaVuSans-Bold.ttf", 32)
except:
    font = ImageFont.load_default()

# Paste images
for d, dataset in enumerate(datasets):
    for m, model in enumerate(models):
        match = [f for ds, mo, f in data if ds == dataset and mo == model]
        if match:
            img = Image.open(match[0]).convert("RGB")
            img = img.resize((target_w, target_h), Image.LANCZOS)
            x0 = label_space_left + m * target_w
            y0 = label_space_top + d * target_h
            big_img.paste(img, (x0, y0))

# Model names (top, horizontal)
for m, model in enumerate(models):
    x = label_space_left + m * target_w + target_w // 2
    y = label_space_top // 2
    draw.text((x, y), model, font=font, fill="black", anchor="mm")

# Dataset names (left, vertical)
for d, dataset in enumerate(datasets):
    y_center = label_space_top + d * target_h + target_h // 2
    # Make a generous canvas for the text
    txt_img = Image.new("RGBA", (target_h, 200), (255, 255, 255, 0))
    txt_draw = ImageDraw.Draw(txt_img)
    txt_draw.text((target_h // 2, 100), dataset, font=font, fill="black", anchor="mm")
    rotated = txt_img.rotate(90, expand=1, fillcolor="white")
    # Paste next to row
    big_img.paste(rotated, (40, y_center - rotated.size[1] // 2), rotated)

big_img.save("del_without_defense.png", dpi=(400, 400))
