"""
Author: Zhengyao Li, Joy Zhuo
Date: Oct 27 2025
"""
# Behavior Cloning demo for Malmo
# This script can either:
# (1): Generate fake arrow images (so we can test training)
# (2): Train a small CNN to imitate player actions from images
# (3): Load any test image from the dataset for prediction.
# Later we can swap the fake data for real Malmo frames + actions.

import os, csv, argparse, random
from typing import List
import numpy as np
from PIL import Image, ImageDraw

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

DATA_ROOT   = "../BC_demo/dataset_malmo"                          # dataset folder
FRAMES_DIR  = os.path.join(DATA_ROOT, "frames")        # images stores here
CSV_PATH    = os.path.join(DATA_ROOT, "actions.csv")   # mapping image with action
IMG_SIZE    = 84                                       # resize images to 84x84
BATCH_SIZE  = 64
EPOCHS      = 10
LR          = 3e-4
WEIGHT_DECAY= 1e-5
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
# SAVE_PATH   = "malmo_bc_cnn.pth"
THIS_DIR = os.path.dirname(__file__)
SAVE_PATH = os.path.join(THIS_DIR, "malmo_bc_cnn.pth")
SEED        = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# Action label mapping
ACTION2ID = {"LEFT":0, "RIGHT":1, "GO":2}
ID2ACTION = {v:k for k,v in ACTION2ID.items()}
NUM_ACTIONS = len(ACTION2ID)

# Simple dataset wrapper that loads images + action labels
class MalmoImageDataset(Dataset):

    def __init__(self, root=DATA_ROOT, img_size=IMG_SIZE):
        self.root = root
        self.img_size = img_size
        frames_dir = os.path.join(root, "frames")
        assert os.path.isdir(frames_dir), f"Missing {frames_dir}"
        assert os.path.exists(CSV_PATH), f"Missing {CSV_PATH}"
        self.samples = []

        # Read .CSV line by line
        with open(CSV_PATH, "r", newline="") as f:
            r = csv.reader(f)
            header = next(r, None)
            if header and header[0].lower() != "filename":
                fname, act = header
                self.samples.append((os.path.join(frames_dir, fname), ACTION2ID[act]))
            for row in r:
                fname, act = row
                self.samples.append((os.path.join(frames_dir, fname), ACTION2ID[act]))
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, action = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = img.resize((self.img_size, self.img_size), resample=Image.BILINEAR)
        x = torch.from_numpy(np.array(img)).permute(2,0,1).float() / 255.0
        y = torch.tensor(action, dtype=torch.long)
        return x, y

# Define the Behavior Cloning Convolutional Neural Network
# This model takes an RGB images (later from the Malmo environment)
# and oredict which action the plauer would take (LEFT, RIGHT, GO, BACK) 
# using 3 layer CNN encoder followed by two fully connected layers that 
# maps those feature to actions.
class BC_CNN(nn.Module):
    def __init__(self, num_actions=NUM_ACTIONS):
        super().__init__()

        # This part extracts visual features from raw RGB image
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 8, 4), nn.ReLU(inplace=True),
            nn.Conv2d(32,64, 4, 2), nn.ReLU(inplace=True),
            nn.Conv2d(64,64, 3, 1), nn.ReLU(inplace=True),
            nn.Flatten()
        )
        with torch.no_grad():
            dummy = torch.zeros(1,3,IMG_SIZE,IMG_SIZE)
            flat = self.enc(dummy).shape[-1]
        self.head = nn.Sequential(
            nn.Linear(flat, 512), nn.ReLU(inplace=True),
            nn.Linear(512, num_actions)
        )
    def forward(self, x):
        z = self.enc(x)
        return self.head(z)

# Training loop for the Behavior Cloning CNN model
#
# process:
# 1. Loads the image,action dataset and splits it into training/validation sets.
# 2. Feeds mini batches of images through the BC_CNN model.
# 3. Calculates how far off the model’s predictions are (cross-entropy loss).
# 4. Updates model weights using Adam optimizer to improve accuracy.
# 5. Evaluates validation performance each epoch.
# 6. Saves the best-performing model checkpoint to disk.
#
# By the end, we get a trained CNN that can predict the most likely
# human action (LEFT, RIGHT, GO, BACK) from a given frame.
def train():
    print(f"Device: {DEVICE}")
    ds = MalmoImageDataset(DATA_ROOT, IMG_SIZE)
    n_total = len(ds)
    n_val = max(1, int(0.15*n_total)) # Reserve 15% for later validation
    n_train = n_total - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(SEED))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = BC_CNN().to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()

    best_val = 0.0
    for epoch in range(1, EPOCHS+1):
        # Train
        model.train()
        tr_loss = 0.0; tr_hits = 0.0; seen = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            b = xb.size(0)
            tr_loss += loss.item() * b
            tr_hits += (logits.argmax(1) == yb).float().sum().item()
            seen += b
        tr_loss /= seen
        tr_acc  = tr_hits / seen

        # Validate
        model.eval()
        v_loss = 0.0; v_hits = 0.0; v_seen = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)
                loss = loss_fn(logits, yb)
                b = xb.size(0)
                v_loss += loss.item() * b
                v_hits += (logits.argmax(1) == yb).float().sum().item()
                v_seen += b
        v_loss /= v_seen
        v_acc  = v_hits / v_seen

        print(f"Epoch {epoch:02d} | train {tr_loss:.3f}/{tr_acc*100:.1f}% | validation {v_loss:.3f}/{v_acc*100:.1f}%")

        # Save best checkpoint
        if v_acc > best_val:
            best_val = v_acc
            # Save on CPU so it's easy to load anywhere
            torch.save({
                "model": model.cpu().state_dict(),
                "img_size": IMG_SIZE,
                "action2id": ACTION2ID
            }, SAVE_PATH)
            if DEVICE != "cpu":
                model.to(DEVICE)
    print(f"[OK] Saved {SAVE_PATH} (best val acc {best_val*100:.1f}%)")

# Minimal inference
def load_policy_and_predict(frame_rgb_numpy: np.ndarray) -> str:
    # Load checkpoint
    ckpt = torch.load(SAVE_PATH, map_location="cpu")
    img_size = ckpt.get("img_size", IMG_SIZE)
    action2id = ckpt["action2id"]
    id2action = {v:k for k,v in action2id.items()}
    # Recreate model and load weights
    model = BC_CNN(num_actions=len(action2id))
    model.load_state_dict(ckpt["model"])
    model.eval()
    # Convert raw RGB frame to normalized tensor
    img = Image.fromarray(frame_rgb_numpy).convert("RGB").resize((img_size, img_size), Image.BILINEAR)
    x = torch.from_numpy(np.array(img)).permute(2,0,1).float()[None] / 255.0
    with torch.no_grad():
        logits = model(x)
        a_id = logits.argmax(1).item()
    return id2action[a_id]

def main():
    parser = argparse.ArgumentParser(description="Behavior Cloning Demo", allow_abbrev=False)
    parser.add_argument("--train", action="store_true", help="Train CNN on dataset_malmo/")
    parser.add_argument("--predict", action="store_true", help="Run model inference on a sample frame")

    args = parser.parse_args()

    if args.train:
        train()
    if args.predict:
        # Load any test image from the dataset for prediction.
        # You can change the file path below to test a different frame.
        img = Image.open("dataset_malmo/frames/000001.png").convert("RGB")
        frame = np.array(img)
        print(load_policy_and_predict(frame))

    if not args.train and not args.predict:
        print("Nothing to do. Try:  python bc.py --make-fake 3000   and/or   python bc.py --train")

if __name__ == "__main__":
    main()