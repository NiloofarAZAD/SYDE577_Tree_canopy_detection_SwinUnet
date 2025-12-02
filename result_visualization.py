import re
import matplotlib.pyplot as plt

log_file = "./model_out/log.txt"

epochs = []
train_loss = []
val_loss = []
train_acc = []
val_acc = []

with open(log_file, "r") as f:
    for line in f:

        m = re.search(r"Train epoch:\s*(\d+)\s*:\s*loss\s*:\s*([0-9.]+)", line)
        if m:
            epochs.append(int(m.group(1)))
            train_loss.append(float(m.group(2)))

        m = re.search(r"Val epoch:\s*\d+\s*:\s*loss\s*:\s*([0-9.]+)", line)
        if m:
            val_loss.append(float(m.group(1)))

        m = re.search(r"train_acc:\s*([0-9.]+)", line)
        if m:
            train_acc.append(float(m.group(1)))

        m = re.search(r"Pixel Accuracy =\s*([0-9.]+)", line)
        if m:
            val_acc.append(float(m.group(1)))


# loss
plt.figure(figsize=(10,4))
plt.plot(epochs, train_loss, label="Train loss")
plt.plot(epochs, val_loss, label="Validation loss")
plt.xlabel("Epoch", fontsize=18)
plt.ylabel("Loss", fontsize=18)
plt.title("Train vs Validation Loss", fontsize=16)
plt.grid(True, linestyle='--', alpha=0.4)
plt.legend(fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
plt.show()

# accuracy
plt.figure(figsize=(10,4))
plt.plot(epochs, train_acc, label="Train pixel accuracy")
plt.plot(epochs, val_acc, label="Validation pixel accuracy")
plt.xlabel("Epoch", fontsize=18)
plt.ylabel("Pixel accuracy", fontsize=18)
plt.title("Train vs Validation Pixel Accuracy", fontsize=16)
plt.grid(True, linestyle='--', alpha=0.4)
plt.legend(fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
plt.show()
