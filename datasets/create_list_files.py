import os
import random

base = "./Synapse/train_images"
list_dir = "./Synapse/lists"
os.makedirs(list_dir, exist_ok=True)

items = [os.path.splitext(f)[0] for f in os.listdir(base) if f.endswith(".tif")]

random.seed(42)
random.shuffle(items)

n_total = len(items)
n_val = int(n_total * 0.10)

val_list = items[:n_val]
train_list = items[n_val:]

print("Train =", len(train_list))
print("Val =", len(val_list))

with open(os.path.join(list_dir, "train.txt"), "w") as f:
    for x in train_list:
        f.write(x + "\n")

with open(os.path.join(list_dir, "val.txt"), "w") as f:
    for x in val_list:
        f.write(x + "\n")
