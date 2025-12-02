import os
import json
import numpy as np
import cv2
import torch
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from networks.vision_transformer import SwinUnet as ViT_seg
from config import get_config
import argparse


# ============================
# CONFIG
# ============================
IMAGE_DIR = "./datasets/Synapse/evaluation_images/"
MODEL_PATH = "./model_out/best_model.pth"
OUTPUT_JSON = "submission_swinunet.json"
SAMPLE_JSON = "./datasets/Synapse/sample_answer.json"

IMG_SIZE = 224
CONF_THRESHOLD = 0.5
MIN_CONTOUR_AREA = 5

CLASS_MAP = {
    1: "group_of_trees",
    2: "individual_tree"
}
# background = 0


# ============================
# Dataset
# ============================
class EvalDataset(Dataset):
    def __init__(self, image_dir):
        self.files = sorted([
            f for f in os.listdir(image_dir)
            if os.path.isfile(os.path.join(image_dir, f))
        ])
        self.image_dir = image_dir

        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file_name = self.files[index]
        path = os.path.join(self.image_dir, file_name)

        img = Image.open(path).convert("RGB")
        orig_w, orig_h = img.size
        tensor_img = self.transform(img)

        return tensor_img, file_name, (orig_w, orig_h)


# ============================
# Load Swin-Unet
# ============================
def load_swinunet(model_path):
    parser = argparse.ArgumentParser()

    parser.add_argument('--root_path', type=str, default='./datasets')
    parser.add_argument('--dataset', type=str, default='Synapse')
    parser.add_argument('--list_dir', type=str, default='./datasets/Synapse/lists')
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--max_iterations', type=int, default=30000)
    parser.add_argument('--max_epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--n_gpu', type=int, default=1)
    parser.add_argument('--deterministic', type=int, default=1)
    parser.add_argument('--base_lr', type=float, default=0.01)
    parser.add_argument('--img_size', type=int, default=IMG_SIZE)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--cfg', type=str, default="configs/swin_tiny_patch4_window7_224_lite.yaml")
    parser.add_argument('--opts', default=None, nargs='+')
    parser.add_argument('--zip', action='store_true')
    parser.add_argument('--cache-mode', type=str, default='part')
    parser.add_argument('--resume', default=None)
    parser.add_argument('--accumulation-steps', default=None)
    parser.add_argument('--use-checkpoint', action='store_true')
    parser.add_argument('--amp-opt-level', type=str, default='O1')
    parser.add_argument('--tag', default=None)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--throughput', action='store_true')
    parser.add_argument('--n_class', type=int, default=3)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--eval_interval', type=int, default=1)

    args = parser.parse_args(args=[])

    config = get_config(args)
    model = ViT_seg(config, img_size=args.img_size, num_classes=args.num_classes).cuda()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


# ============================
# Mask → Polygons in ORIGINAL RESOLUTION
# ============================
def mask_to_polygons(pred_mask, class_id, orig_size):
    orig_w = int(orig_size[0])
    orig_h = int(orig_size[1])

    binary_mask = (pred_mask == class_id).astype(np.uint8)
    small_mask = cv2.resize(binary_mask, (128, 128), interpolation=cv2.INTER_NEAREST)

    contours, _ = cv2.findContours(small_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []
    scale_w = orig_w / 128.0
    scale_h = orig_h / 128.0

    for cnt in contours:
        if cv2.contourArea(cnt) < MIN_CONTOUR_AREA:
            continue

        poly = cnt.reshape(-1, 2).astype(float)
        poly[:, 0] *= scale_w
        poly[:, 1] *= scale_h

        polygons.append(poly.flatten().tolist())

    return polygons


# ============================
# Confidence from probability map
# ============================
def compute_confidence(prob_map, binary_mask_224):
    vals = prob_map[binary_mask_224 == 1]
    if len(vals) == 0:
        return 0.0
    return float(vals.mean())


# ============================
# Main inference
# ============================
def run_inference():
    print("Loading model...")
    model = load_swinunet(MODEL_PATH)

    dataset = EvalDataset(IMAGE_DIR)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    print("Loading sample JSON...")
    with open(SAMPLE_JSON, "r") as f:
        submission_data = json.load(f)

    predictions = defaultdict(list)

    print("Running inference...")

    with torch.no_grad():
        for img_tensor, file_name, orig_size in loader:
            img_tensor = img_tensor.cuda()
            file_name = file_name[0]

            logits = model(img_tensor)
            probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()  # (3,224,224)
            mask = np.argmax(probs, axis=0)

            orig_w = int(orig_size[0])
            orig_h = int(orig_size[1])

            scale_x = IMG_SIZE / orig_w
            scale_y = IMG_SIZE / orig_h

            for cid, cname in CLASS_MAP.items():

                polygons = mask_to_polygons(mask, cid, orig_size)

                for poly in polygons:
                    poly_np = np.array(poly).reshape(-1, 2)

                    # Scale ORIGINAL polygon → 224×224
                    poly_scaled = poly_np.copy().astype(float)
                    poly_scaled[:, 0] *= scale_x
                    poly_scaled[:, 1] *= scale_y

                    # Fill confidence mask
                    poly_mask = np.zeros((IMG_SIZE, IMG_SIZE), np.uint8)
                    cv2.fillPoly(poly_mask, [poly_scaled.astype(np.int32)], 1)

                    score = compute_confidence(probs[cid], poly_mask)
                    if score < CONF_THRESHOLD:
                        continue

                    predictions[file_name].append({
                        "class": cname,
                        "confidence_score": score,
                        "segmentation": poly  # original resolution polygon
                    })

    # attach predictions
    for img in submission_data["images"]:
        fname = img["file_name"]
        img["annotations"] = predictions[fname]

    with open(OUTPUT_JSON, "w") as f:
        json.dump(submission_data, f, indent=2)

    print("Submission saved as:", OUTPUT_JSON)


# RUN
if __name__ == "__main__":
    run_inference()
