import json
import os
import numpy as np
from PIL import Image, ImageDraw


json_path = "./Synapse/train_annotations.json"
mask_dir = "./Synapse/train_masks"         
os.makedirs(mask_dir, exist_ok=True)


class_to_id = { 
    "group_of_trees": 1,
    "individual_tree": 2
}

with open(json_path, "r") as f:
    data = json.load(f)["images"]

for entry in data:
    fname = entry["file_name"]
    width, height = entry["width"], entry["height"]
    anns = entry["annotations"]

    # empty mask (background = 0)
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    for ann in anns:
        cls_name = ann["class"]
        cls_id = class_to_id[cls_name]

        poly = ann["segmentation"]
        poly_xy = [(poly[i], poly[i+1]) for i in range(0, len(poly), 2)]

        draw.polygon(poly_xy, fill=cls_id, outline=cls_id)

    base = os.path.splitext(fname)[0]
    out_path = os.path.join(mask_dir, base + ".png")
    mask.save(out_path)

    print("Saved:", out_path)

print("\nAll masks created in:", mask_dir)



# # test masks
# mask_path = "./Synapse/train_masks/80cm_train_147.png"
# mask = np.array(Image.open(mask_path))

# # --- Compute mask size ---
# height, width = mask.shape
# print("Mask shape (H, W):", (height, width))
# print("Total pixels:", height * width)

# # --- Pixel count for each class ---
# unique_vals, counts = np.unique(mask, return_counts=True)
# print("\nClass pixel counts:")
# for u, c in zip(unique_vals, counts):
#     print(f"  Class {u}: {c} pixels")

# # --- Create color preview ---
# color = np.zeros((height, width, 3), dtype=np.uint8)

# color[mask == 1] = [0, 255, 0]    # class 1 = green
# color[mask == 0] = [0, 0, 0]      # background = black
# color[mask == 2] = [255, 0, 0]    # class 2 = red

# out_preview = "preview_color.png"
# Image.fromarray(color).save(out_preview)

# print("\nSaved", out_preview)

# mask = np.array(Image.open(mask_path))

# print("Unique pixel values in this mask:", np.unique(mask))
