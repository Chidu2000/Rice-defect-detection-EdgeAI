import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_DATASET_DIR = (
    r"C:\Users\User\.cache\kagglehub\datasets\anshulm257\rice-disease-dataset"
    r"\versions\1\Rice_Leaf_AUG"
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resize_image(image: Image.Image, size: int) -> Image.Image:
    return image.resize((size, size), Image.Resampling.BILINEAR)


def center_crop_image(image: Image.Image, size: int) -> Image.Image:
    width, height = image.size
    left = max((width - size) // 2, 0)
    top = max((height - size) // 2, 0)
    return image.crop((left, top, left + size, top + size))


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32)
    tensor = torch.from_numpy(array)
    tensor = tensor.permute(2, 0, 1).float().div(255.0)
    return tensor


def normalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=tensor.dtype).view(3, 1, 1)
    return (tensor - mean) / std


class RiceLeafDataset(Dataset):
    def __init__(self, samples, image_size: int, train: bool) -> None:
        self.samples = samples
        self.image_size = image_size
        self.train = train

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            if self.train and random.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            image = resize_image(image, self.image_size + 16)
            image = center_crop_image(image, self.image_size)
            tensor = normalize_tensor(image_to_tensor(image))
        return tensor, label


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EdgeRiceNet(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            DepthwiseSeparableBlock(16, 32, stride=1),
            DepthwiseSeparableBlock(32, 64, stride=2),
            DepthwiseSeparableBlock(64, 96, stride=2),
            DepthwiseSeparableBlock(96, 128, stride=2),
            DepthwiseSeparableBlock(128, 160, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(160, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


def build_samples(dataset_dir: Path):
    classes = sorted([entry.name for entry in dataset_dir.iterdir() if entry.is_dir()])
    class_to_idx = {class_name: index for index, class_name in enumerate(classes)}
    samples = []
    for class_name in classes:
        class_dir = dataset_dir / class_name
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.suffix.lower() in IMAGE_EXTENSIONS and image_path.is_file():
                samples.append((image_path, class_to_idx[class_name]))
    if not samples:
        raise ValueError(f"No image files found under {dataset_dir}")
    return samples, classes, class_to_idx


def split_samples(samples, validation_ratio: float, seed: int):
    labels = [label for _, label in samples]
    train_samples, val_samples = train_test_split(
        samples,
        test_size=validation_ratio,
        stratify=labels,
        random_state=seed,
    )
    return train_samples, val_samples


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    predictions = logits.argmax(dim=1)
    return (predictions == targets).float().mean().item()


def run_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_accuracy = 0.0
    total_samples = 0

    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_samples += batch_size
        total_loss += loss.item() * batch_size
        total_accuracy += accuracy_from_logits(logits.detach(), targets) * batch_size

    return total_loss / total_samples, total_accuracy / total_samples


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    total_samples = 0

    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)
        logits = model(images)
        loss = criterion(logits, targets)

        batch_size = images.size(0)
        total_samples += batch_size
        total_loss += loss.item() * batch_size
        total_accuracy += accuracy_from_logits(logits, targets) * batch_size

    return total_loss / total_samples, total_accuracy / total_samples


def save_outputs(output_dir: Path, model, classes, history, args) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "rice_leaf_edge_model.pt"
    metadata_path = output_dir / "metadata.json"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "classes": classes,
            "image_size": args.image_size,
        },
        checkpoint_path,
    )

    metadata = {
        "classes": classes,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "scheduler": args.scheduler,
        "image_size": args.image_size,
        "history": history,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a rice leaf disease classifier with PyTorch.")
    parser.add_argument("--dataset-dir", type=Path, default=Path(DEFAULT_DATASET_DIR))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--scheduler",
        choices=["none", "plateau", "cosine"],
        default="plateau",
    )
    parser.add_argument("--lr-patience", type=int, default=2)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    dataset_dir = args.dataset_dir
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    samples, classes, _ = build_samples(dataset_dir)
    train_samples, val_samples = split_samples(samples, args.validation_ratio, args.seed)

    train_dataset = RiceLeafDataset(train_samples, args.image_size, train=True)
    val_dataset = RiceLeafDataset(val_samples, args.image_size, train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EdgeRiceNet(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = None
    if args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.lr_factor,
            patience=args.lr_patience,
            min_lr=args.min_lr,
        )
    elif args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.min_lr,
        )

    train_counts = Counter(label for _, label in train_samples)
    val_counts = Counter(label for _, label in val_samples)

    print(f"Dataset: {dataset_dir}")
    print(f"Device: {device}")
    print(f"Classes: {classes}")
    print(f"Train samples: {len(train_samples)}")
    print(f"Validation samples: {len(val_samples)}")
    print(f"Scheduler: {args.scheduler}")
    print("Train distribution:", {classes[idx]: count for idx, count in sorted(train_counts.items())})
    print("Validation distribution:", {classes[idx]: count for idx, count in sorted(val_counts.items())})

    history = []
    best_val_accuracy = 0.0
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)

        epoch_result = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_accuracy": round(train_accuracy, 4),
            "val_loss": round(val_loss, 4),
            "val_accuracy": round(val_accuracy, 4),
            "learning_rate": round(optimizer.param_groups[0]["lr"], 8),
        }
        history.append(epoch_result)

        print(
            f"Epoch {epoch:02d}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_accuracy:.4f} "
            f"lr={optimizer.param_groups[0]['lr']:.6f}"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if val_loss < best_val_loss:
            best_val_loss = val_loss

        if scheduler is not None:
            if args.scheduler == "plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()

    if best_state is not None:
        model.load_state_dict(best_state)

    save_outputs(args.output_dir, model, classes, history, args)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"Saved model and metadata to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
