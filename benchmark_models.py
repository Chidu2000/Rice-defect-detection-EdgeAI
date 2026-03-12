import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from predict import load_checkpoint, preprocess_image
from train import build_samples, evaluate


def benchmark_model(model, input_tensor: torch.Tensor, iterations: int, warmup: int) -> float:
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)

        timings = []
        for _ in range(iterations):
            start = time.perf_counter()
            _ = model(input_tensor)
            end = time.perf_counter()
            timings.append((end - start) * 1000.0)

    return statistics.mean(timings)


class EdgeWrapper(torch.nn.Module):
    def __init__(self, scripted_model):
        super().__init__()
        self.scripted_model = scripted_model

    def forward(self, x):
        return self.scripted_model(x)


def evaluate_scripted_model(scripted_model, dataset_dir: Path, image_size: int, batch_size: int):
    from train import RiceLeafDataset, split_samples
    from torch.utils.data import DataLoader
    from torch import nn

    samples, _, _ = build_samples(dataset_dir)
    _, val_samples = split_samples(samples, validation_ratio=0.2, seed=42)
    val_dataset = RiceLeafDataset(val_samples, image_size=image_size, train=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    loss, accuracy = evaluate(EdgeWrapper(scripted_model), val_loader, nn.CrossEntropyLoss(), torch.device("cpu"))
    return loss, accuracy


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark baseline and edge models.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models") / "baseline" / "rice_leaf_edge_model.pt",
    )
    parser.add_argument(
        "--edge-model",
        type=Path,
        default=Path("models") / "edge" / "rice_leaf_edge_mobile_int8.ptl",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(
            r"C:\Users\User\.cache\kagglehub\datasets\anshulm257\rice-disease-dataset"
            r"\versions\1\Rice_Leaf_AUG"
        ),
    )
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def main():
    args = parse_args()

    baseline_model, classes, image_size = load_checkpoint(args.checkpoint)
    edge_scripted_model = torch.jit.load(str(args.edge_model), map_location="cpu")
    edge_scripted_model.eval()

    example_image = next(args.dataset_dir.rglob("*.jpg"))
    input_tensor = preprocess_image(example_image, image_size)

    baseline_latency_ms = benchmark_model(baseline_model, input_tensor, args.iterations, args.warmup)
    edge_latency_ms = benchmark_model(edge_scripted_model, input_tensor, args.iterations, args.warmup)

    from train import RiceLeafDataset, split_samples
    from torch.utils.data import DataLoader
    from torch import nn

    samples, _, _ = build_samples(args.dataset_dir)
    _, val_samples = split_samples(samples, validation_ratio=0.2, seed=42)
    val_dataset = RiceLeafDataset(val_samples, image_size=image_size, train=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    baseline_loss, baseline_accuracy = evaluate(
        baseline_model,
        val_loader,
        nn.CrossEntropyLoss(),
        torch.device("cpu"),
    )
    edge_loss, edge_accuracy = evaluate_scripted_model(
        edge_scripted_model,
        args.dataset_dir,
        image_size=image_size,
        batch_size=args.batch_size,
    )

    results = {
        "classes": classes,
        "image_size": image_size,
        "example_image": str(example_image),
        "baseline": {
            "file_size_bytes": args.checkpoint.stat().st_size,
            "val_loss": round(baseline_loss, 4),
            "val_accuracy": round(baseline_accuracy * 100.0, 2),
            "latency_ms": round(baseline_latency_ms, 2),
        },
        "edge": {
            "file_size_bytes": args.edge_model.stat().st_size,
            "val_loss": round(edge_loss, 4),
            "val_accuracy": round(edge_accuracy * 100.0, 2),
            "latency_ms": round(edge_latency_ms, 2),
        },
    }

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
