import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from train import EdgeRiceNet, center_crop_image, normalize_tensor, resize_image


def load_checkpoint(checkpoint_path: Path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    classes = checkpoint["classes"]
    image_size = checkpoint["image_size"]
    model = EdgeRiceNet(num_classes=len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, classes, image_size


def preprocess_image(image_path: Path, image_size: int) -> torch.Tensor:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image = resize_image(image, image_size + 16)
        image = center_crop_image(image, image_size)
        array = np.asarray(image, dtype=np.float32)
        tensor = torch.from_numpy(array).permute(2, 0, 1).float().div(255.0)
    return normalize_tensor(tensor).unsqueeze(0)


def predict(model: torch.nn.Module, input_tensor: torch.Tensor, classes: list[str], top_k: int):
    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        top_probs, top_indices = torch.topk(probabilities, k=min(top_k, len(classes)))

    results = []
    for probability, index in zip(top_probs.tolist(), top_indices.tolist()):
        results.append(
            {
                "label": classes[index],
                "confidence": round(probability, 4),
            }
        )
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on a rice leaf image.")
    parser.add_argument("image", type=Path, help="Path to an input image")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts_cosine") / "rice_leaf_edge_model.pt",
        help="Path to a checkpoint produced by train.py",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Number of predictions to print")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print predictions as JSON instead of plain text",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.image.exists():
        raise FileNotFoundError(f"Input image not found: {args.image}")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    model, classes, image_size = load_checkpoint(args.checkpoint)
    input_tensor = preprocess_image(args.image, image_size)
    results = predict(model, input_tensor, classes, args.top_k)

    if args.json:
        print(json.dumps(results, indent=2))
        return

    print(f"Image: {args.image}")
    print(f"Checkpoint: {args.checkpoint}")
    for rank, result in enumerate(results, start=1):
        print(f"{rank}. {result['label']} ({result['confidence'] * 100:.2f}%)")


if __name__ == "__main__":
    main()
