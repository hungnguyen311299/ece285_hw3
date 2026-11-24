import csv
import json
import os
import random
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image

from metrics import InceptionMetrics
from model import build_diffusion

DATA = Path("data/Histopathology")
OUTPUT = Path("results")
IMAGE_SIZE = 64
EPOCHS = 200
BATCH_SIZE = 16
LEARNING_RATE = 8e-5
EVAL_EVERY = 25
SAMPLE_EVERY = 10
SEED = 1265


class Histopathology(Dataset):
    def __init__(self, paths, augment=False):
        steps = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))]
        if augment:
            steps.append(transforms.RandomHorizontalFlip())
        steps.append(transforms.ToTensor())  # denoising-diffusion-pytorch expects [0, 1]
        self.transform = transforms.Compose(steps)
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB"))


def group_id(path):
    """Keep `image_12.png` and `image_12 (2).png` in the same data split."""
    match = re.fullmatch(r"image_(\d+)(?: \(2\))?\.png", path.name)
    if match is None:
        raise ValueError(f"Unexpected image filename: {path.name}")
    return int(match.group(1))


def split_paths():
    paths = sorted(DATA.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No PNG images found in {DATA.resolve()}")
    groups = sorted({group_id(path) for path in paths})
    random.Random(SEED).shuffle(groups)
    train_end = int(0.8 * len(groups))
    validation_end = int(0.9 * len(groups))
    train_groups = set(groups[:train_end])
    validation_groups = set(groups[train_end:validation_end])
    train = [path for path in paths if group_id(path) in train_groups]
    validation = [path for path in paths if group_id(path) in validation_groups]
    test = [path for path in paths if group_id(path) not in train_groups | validation_groups]
    return train, validation, test


def write_csv(path, header, rows):
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


def plot_curves(history, scores):
    figure, axes = plt.subplots(1, 3, figsize=(12, 3))
    axes[0].plot([row[0] for row in history], [row[1] for row in history], color="C0")
    axes[0].set(title="Diffusion loss", xlabel="epoch", ylabel="MSE")
    for axis, column, title, color in ((axes[1], 1, "FID", "C1"),
                                       (axes[2], 2, "Inception Score", "C2")):
        if scores:
            axis.plot([row[0] for row in scores], [row[column] for row in scores], "o-",
                      color=color)
        axis.set(title=title, xlabel="epoch")
    figure.tight_layout()
    figure.savefig(OUTPUT / "training_curves.png", dpi=180)
    plt.close(figure)


@torch.inference_mode()
def save_samples(diffusion, path, count=32):
    diffusion.eval()
    save_image(diffusion.sample(batch_size=count).cpu(), path, nrow=8)
    diffusion.train()


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUTPUT.mkdir(exist_ok=True)

    train_paths, validation_paths, test_paths = split_paths()
    workers = min(4, os.cpu_count() or 1)
    loader_args = dict(batch_size=BATCH_SIZE, num_workers=workers, pin_memory=device.type == "cuda",
                       persistent_workers=workers > 0)
    train_loader = DataLoader(Histopathology(train_paths, augment=True), shuffle=True,
                              drop_last=True, **loader_args)
    validation_loader = DataLoader(Histopathology(validation_paths), shuffle=False,
                                   drop_last=False, **loader_args)
    test_loader = DataLoader(Histopathology(test_paths), shuffle=False, drop_last=False, **loader_args)
    if len(train_loader) == 0:
        raise RuntimeError("The batch size is larger than the training split.")

    config = {"image_size": IMAGE_SIZE, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
              "learning_rate": LEARNING_RATE, "eval_every": EVAL_EVERY,
              "sample_every": SAMPLE_EVERY, "seed": SEED,
              "train_images": len(train_paths), "validation_images": len(validation_paths),
              "test_images": len(test_paths),
              "device": str(device)}
    (OUTPUT / "settings.json").write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps(config, indent=2), flush=True)

    diffusion = build_diffusion(IMAGE_SIZE).to(device)
    optimizer = torch.optim.Adam(diffusion.parameters(), lr=LEARNING_RATE)
    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    validation_metrics = InceptionMetrics(device, validation_loader)
    history, scores, best_fid = [], [], float("inf")

    for epoch in range(1, EPOCHS + 1):
        diffusion.train()
        loss_total = 0.0
        for images in train_loader:
            images = images.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                loss = diffusion(images)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_total += loss.item()

        history.append((epoch, loss_total / len(train_loader)))
        write_csv(OUTPUT / "history.csv", ["epoch", "diffusion_loss"], history)
        print(f"epoch {epoch}: loss={history[-1][1]:.5f}", flush=True)

        if epoch % SAMPLE_EVERY == 0:
            save_samples(diffusion, OUTPUT / f"samples_{epoch:03d}.png")

        if epoch % EVAL_EVERY == 0:
            fid, inception_score = validation_metrics.score(diffusion)
            scores.append((epoch, fid, inception_score))
            write_csv(OUTPUT / "metrics.csv", ["epoch", "fid", "inception_score"], scores)
            print(f"epoch {epoch}: validation FID={fid:.2f} IS={inception_score:.3f}", flush=True)
            if fid < best_fid:
                best_fid = fid
                torch.save({"epoch": epoch, "diffusion": diffusion.state_dict(),
                            "optimizer": optimizer.state_dict(), "config": config}, OUTPUT / "best.pt")
        plot_curves(history, scores)

    if not scores:
        raise RuntimeError("No metric was computed; set EVAL_EVERY no larger than EPOCHS.")
    best = torch.load(OUTPUT / "best.pt", map_location=device, weights_only=False)
    diffusion.load_state_dict(best["diffusion"])
    save_samples(diffusion, OUTPUT / "generated.png")
    test_fid, test_inception_score = InceptionMetrics(device, test_loader).score(diffusion)
    write_csv(OUTPUT / "test_metrics.csv", ["epoch", "fid", "inception_score"],
              [(best["epoch"], test_fid, test_inception_score)])
    print(f"best epoch {best['epoch']}: validation FID={best_fid:.2f}; "
          f"test FID={test_fid:.2f} IS={test_inception_score:.3f}", flush=True)


if __name__ == "__main__":
    main()
