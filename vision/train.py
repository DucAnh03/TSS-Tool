#!/usr/bin/env python3
"""
Vision pipeline – SimpleCNN on CIFAR-10.

Trains (or loads) the model, evaluates on the test split, and writes
results into the shared metrics.json contract.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


METRICS_PATH = Path("artifacts/latest/metrics.json")
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


# ── Model ─────────────────────────────────────────────────────────────────────

class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


# ── Data ──────────────────────────────────────────────────────────────────────

def get_loaders(batch_size: int, quick: bool):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    train_ds = torchvision.datasets.CIFAR10(
        root=".cifar10_cache", train=True, download=True, transform=tf
    )
    test_ds = torchvision.datasets.CIFAR10(
        root=".cifar10_cache", train=False, download=True, transform=tf
    )
    if quick:
        from torch.utils.data import Subset
        train_ds = Subset(train_ds, range(2000))
        test_ds = Subset(test_ds, range(500))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader


# ── Training ──────────────────────────────────────────────────────────────────

def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0.0, 0
    all_preds, all_labels = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        total_loss += criterion(outputs, labels).item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += preds.eq(labels).sum().item()
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    n = len(loader.dataset)
    return total_loss / n, correct / n, all_labels, all_preds


# ── Confusion matrix plot ─────────────────────────────────────────────────────

def save_confusion_matrix(labels, preds, path: Path):
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import numpy as np

    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xticklabels(CIFAR10_CLASSES, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(CIFAR10_CLASSES, fontsize=8)
    for i in range(10):
        for j in range(10):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=6)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("CIFAR-10 Confusion Matrix")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[vision] Confusion matrix → {path}")


# ── Metrics contract ──────────────────────────────────────────────────────────

def update_metrics(run_id: str, accuracy: float, loss: float, confusion_png: str):
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        data = json.loads(METRICS_PATH.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}

    import subprocess
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short=7", "HEAD"],
            stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        sha = "nogit"

    data.update({
        "run_id": run_id,
        "git_sha": sha,
        "created_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "vision": {
            "dataset": "CIFAR-10",
            "model": "SimpleCNN",
            "accuracy": round(accuracy, 4),
            "loss": round(loss, 4),
            "confusion_png": confusion_png,
        },
    })
    METRICS_PATH.write_text(json.dumps(data, indent=2))
    print(f"[vision] Metrics written → {METRICS_PATH}")


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train SimpleCNN on CIFAR-10")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--quick", action="store_true", help="Use 2k/500 subset")
    p.add_argument("--run-id", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[vision] device={device}  quick={args.quick}")

    train_loader, test_loader = get_loaders(args.batch_size, quick=args.quick)
    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        print(f"  epoch {epoch}/{args.epochs}  train_loss={train_loss:.4f}")

    test_loss, accuracy, labels, preds = evaluate(model, test_loader, criterion, device)
    print(f"[vision] test accuracy={accuracy:.4f}  test_loss={test_loss:.4f}")

    confusion_png = "artifacts/latest/vision_confusion.png"
    save_confusion_matrix(labels, preds, Path(confusion_png))
    update_metrics(args.run_id, accuracy, test_loss, confusion_png)


if __name__ == "__main__":
    main()
