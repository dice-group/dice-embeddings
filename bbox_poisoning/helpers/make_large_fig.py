import os
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

input_folder = Path("../vis_new")

images_by_type = {"add": [], "del": []}

for file in input_folder.glob("*.png"):
    parts = file.stem.split("@")
    if len(parts) != 3:
        continue
    db, model, edit_type = parts
    if edit_type in images_by_type:
        images_by_type[edit_type].append((db, model, file))

for etype in images_by_type:
    images_by_type[etype].sort(key=lambda x: (x[0], x[1]))

def save_image_grid(image_infos, out_file, cols=8):
    if not image_infos:
        return
    imgs = [Image.open(f) for _, _, f in image_infos]
    n = len(imgs)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
    axes = axes.flatten()

    for ax, (db, model, f), img in zip(axes, image_infos, imgs):
        ax.imshow(img)
        ax.set_title(f"{db}\n{model}", fontsize=9)
        ax.axis("off")

    for ax in axes[len(imgs):]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

save_image_grid(images_by_type["add"], "all_add_new.png")
save_image_grid(images_by_type["del"], "all_del_new.png")
