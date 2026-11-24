import csv
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from main import (BATCH_SIZE, IMAGE_SIZE, LEARNING_RATE, OUTPUT, SEED, Histopathology,
                  split_paths)
from model import build_diffusion


def train_epoch_one(device, train_paths):
    random.seed(SEED)
    torch.manual_seed(SEED)
    loader = DataLoader(Histopathology(train_paths, augment=True), batch_size=BATCH_SIZE,
                        shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    diffusion = build_diffusion(IMAGE_SIZE).to(device)
    optimizer = torch.optim.Adam(diffusion.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler("cuda")
    diffusion.train()
    for images in loader:
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda"):
            loss = diffusion(images.to(device, non_blocking=True))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    torch.save({"epoch": 1, "diffusion": diffusion.state_dict()}, OUTPUT / "epoch001.pt")
    return diffusion


def load(path, device):
    diffusion = build_diffusion(IMAGE_SIZE).to(device)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    diffusion.load_state_dict(checkpoint["diffusion"])
    return diffusion, checkpoint["epoch"]


@torch.inference_mode()
def sample(diffusion, count, seed):
    diffusion.eval()
    torch.manual_seed(seed)
    return diffusion.sample(batch_size=count).clamp(0, 1).cpu()


def show(axis, image):
    axis.imshow(image.permute(1, 2, 0).numpy())
    axis.axis("off")


def real_vs_generated(real, generated, path):
    """(a) real test images | (b) generated samples, each 2x4."""
    figure = plt.figure(figsize=(12.6, 3.4))
    grid = figure.add_gridspec(2, 9, width_ratios=[1] * 4 + [0.3] + [1] * 4,
                               wspace=0.04, hspace=0.04, bottom=0.12)
    for offset, images, label in ((0, real, "(a)"), (5, generated, "(b)")):
        for index, image in enumerate(images):
            axis = figure.add_subplot(grid[index // 4, offset + index % 4])
            show(axis, image)
            if index == 5:
                axis.text(1.02, -0.06, label, transform=axis.transAxes, ha="center",
                          va="top", fontsize=13)
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"saved {path}")


def sample_grid(images, path):
    figure, axes = plt.subplots(2, 8, figsize=(12, 3), gridspec_kw=dict(wspace=0.04, hspace=0.04))
    for axis, image in zip(axes.flat, images):
        show(axis, image)
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"saved {path}")


def main():
    device = torch.device("cuda")
    train_paths, _, test_paths = split_paths()
    test = Histopathology(random.Random(SEED).sample(test_paths, 8))
    real = torch.stack([test[i] for i in range(8)])

    with (OUTPUT / "metrics.csv").open() as file:
        best_epoch = int(min(csv.DictReader(file), key=lambda row: float(row["fid"]))["epoch"])

    epoch_one = OUTPUT / "epoch001.pt"
    diffusion = load(epoch_one, device)[0] if epoch_one.exists() else train_epoch_one(device, train_paths)
    real_vs_generated(real, sample(diffusion, 8, seed=0), OUTPUT / "qualitative_epoch001.png")

    diffusion, epoch = load(OUTPUT / "best.pt", device)
    assert epoch == best_epoch, (epoch, best_epoch)
    real_vs_generated(real, sample(diffusion, 8, seed=0), OUTPUT / f"qualitative_epoch{epoch:03d}_best.png")
    # A different seed, so none of these repeat the samples in the figure above.
    sample_grid(sample(diffusion, 16, seed=1), OUTPUT / f"samples_2x8_epoch{epoch:03d}_best.png")


if __name__ == "__main__":
    main()
