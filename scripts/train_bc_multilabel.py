#!/usr/bin/env python3
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF
from PIL import Image

# Auto-pick newest rollout
def newest_rollout_dir(base="data/teleop") -> Path:
    base = Path(base)
    runs = sorted([p for p in base.glob("*") if p.is_dir()])
    if not runs:
        raise FileNotFoundError(f"No rollouts found under {base}")
    return runs[-1]


class TeleopDataset(Dataset):
    def __init__(self, rollout_dir: Path):
        meta = json.loads((rollout_dir / "rollout.json").read_text(encoding="utf-8"))
        self.rollout_dir = rollout_dir
        self.steps = meta["steps"]
        self.action_dim = len(meta["keymap"])

    def __len__(self):
        return len(self.steps)

    def __getitem__(self, idx):
        s = self.steps[idx]
        img_path = self.rollout_dir / s["frame"]
        img = Image.open(img_path).convert("RGB")  # 320x180
        x = TF.to_tensor(img)  # [3,H,W], float in [0,1]
        y = torch.tensor(s["action"], dtype=torch.float32)  # multi-hot
        return x, y


class SmallCNN(nn.Module):
    def __init__(self, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Linear(128, action_dim)

    def forward(self, x):
        z = self.net(x).flatten(1)
        return self.head(z)  # logits


def main():
    rollout_dir = newest_rollout_dir()
    ds = TeleopDataset(rollout_dir)
    dl = DataLoader(ds, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Training on: {rollout_dir}  (n={len(ds)})")

    model = SmallCNN(ds.action_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(10):
        total = 0.0
        n = 0
        for x, y in dl:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = loss_fn(logits, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item() * x.size(0)
            n += x.size(0)

        print(f"epoch {epoch+1:02d}  loss={total/n:.4f}")

    out_dir = Path("checkpoints")
    out_dir.mkdir(exist_ok=True)
    ckpt = out_dir / "bc_multilabel.pt"
    torch.save({"model": model.state_dict(), "rollout": str(rollout_dir)}, ckpt)
    print(f"Saved: {ckpt}")


if __name__ == "__main__":
    main()