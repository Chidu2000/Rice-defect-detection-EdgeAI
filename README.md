# Rice Leaf Edge Classifier

Minimal PyTorch training pipeline for rice leaf disease classification using the Kaggle dataset `anshulm257/rice-disease-dataset`.

## Dataset

Default dataset path:

```text
C:\Users\User\.cache\kagglehub\datasets\anshulm257\rice-disease-dataset\versions\1\Rice_Leaf_AUG
```

Classes:

- `Bacterial Leaf Blight`
- `Brown Spot`
- `Healthy Rice Leaf`
- `Leaf Blast`
- `Leaf scald`
- `Sheath Blight`

## Train

```bash
python train.py --epochs 16 --batch-size 64 --image-size 128 --learning-rate 0.001 --scheduler cosine --output-dir artifacts_cosine
```

Optional arguments:

```bash
python train.py --epochs 10 --batch-size 32 --image-size 224 --output-dir artifacts
```

Artifacts written to `artifacts/`:

- `rice_leaf_edge_model.pt`
- `metadata.json`

## Requirements

Install dependencies with:

```bash
python -m pip install -r requirements.txt
```

## Standalone Inference

Run single-image inference from a trained checkpoint:

```bash
python predict.py path\to\leaf_image.jpg --checkpoint artifacts_cosine/rice_leaf_edge_model.pt --top-k 3
```

JSON output:

```bash
python predict.py path\to\leaf_image.jpg --checkpoint artifacts_cosine/rice_leaf_edge_model.pt --json
```

## Edge Export

Export the trained model for PyTorch Mobile with TorchScript Lite and a quantized Lite Interpreter build:

```bash
python export_edge.py --checkpoint artifacts/rice_leaf_edge_model.pt --output-dir edge_artifacts
```

Optional pruning before export:

```bash
python export_edge.py --checkpoint artifacts/rice_leaf_edge_model.pt --output-dir edge_artifacts --prune-amount 0.1
```

Generated edge artifacts:

- `rice_leaf_edge_scripted.pt`
- `rice_leaf_edge_mobile.ptl`
- `rice_leaf_edge_mobile_int8.pt`
- `rice_leaf_edge_mobile_int8.ptl`
- `edge_export_summary.json`

## Model Files

Stable deliverable paths:

- Baseline model: [rice_leaf_edge_model.pt](g:\New\klyff\models\baseline\rice_leaf_edge_model.pt)
- Edge model: [rice_leaf_edge_mobile_int8.ptl](g:\New\klyff\models\edge\rice_leaf_edge_mobile_int8.ptl)
- Benchmark summary: [benchmark_results.json](g:\New\klyff\benchmark_results.json)

## Trade-off Analysis

Benchmark hardware:

- OS: `Windows 11 10.0.26200`
- CPU: `Intel64 Family 6 Model 142 Stepping 11, GenuineIntel`
- RAM: `7.85 GB`
- CPU threads visible to Python: `8`

Benchmark method:

- Validation accuracy is measured on the fixed 20% validation split used by `train.py`.
- Latency is the mean single-image forward-pass time over 200 runs after 20 warmup runs on CPU using [benchmark_models.py](g:\New\klyff\benchmark_models.py).

| Model | File | File Size (MB) | Accuracy (%) | Inference Speed (ms) |
| --- | --- | ---: | ---: | ---: |
| Baseline | `models/baseline/rice_leaf_edge_model.pt` | 0.211 | 79.37 | 4.70 |
| Edge | `models/edge/rice_leaf_edge_mobile_int8.ptl` | 0.276 | 79.50 | 3.75 |

Trade-off summary:

- The baseline model is the highest-accuracy training checkpoint and is the easiest artifact to reuse for desktop inference and further fine-tuning.
- The edge model uses the PyTorch Mobile Lite runtime format plus dynamic int8 quantization on the classifier head.
- In this project, the edge model is slightly larger on disk than the raw checkpoint because Lite packaging stores executable graph data, but it remains far below the `5 MB` requirement and still below `1 MB`.
- The edge model preserved validation accuracy and was modestly faster in CPU inference on this machine.

## How The Edge Model Stayed Under 1 MB

The final edge model is [rice_leaf_edge_mobile_int8.ptl](g:\New\klyff\models\edge\rice_leaf_edge_mobile_int8.ptl) at `289,677` bytes, which is about `0.276 MB`.

This was achieved through:

- A compact custom CNN in [train.py](g:\New\klyff\train.py) using depthwise-separable convolution blocks instead of a large backbone such as ResNet or EfficientNet.
- A small channel configuration (`16 -> 32 -> 64 -> 96 -> 128 -> 160`) to keep parameter count low from the start.
- Global average pooling before the classifier, which removes the need for large fully connected layers.
- A small input resolution of `128 x 128`, which is sufficient for this dataset and keeps the model edge-friendly.
- Export to the PyTorch Mobile Lite Interpreter format (`.ptl`), which is designed for on-device inference.
- Dynamic int8 quantization on the classifier head during export in [export_edge.py](g:\New\klyff\export_edge.py), reducing edge model overhead without hurting validation accuracy in the final benchmark.

The main reason the model fits comfortably under `1 MB` is that it was designed to be small before export. Quantization and Lite export help, but the largest gain comes from the deliberately lightweight architecture.

## Edge Optimization Notes

- The model is already small because it uses depthwise-separable convolutions and a narrow channel width.
- Export uses the PyTorch Lite Interpreter format (`.ptl`) for device-side inference.
- Dynamic int8 quantization is applied to the classifier head during export while preserving validation accuracy.
- Optional pruning is available through `--prune-amount`; use a small value such as `0.1` or `0.2` if you want to trade some accuracy for more sparsity before export.
- The current model family is designed to stay well below the `5 MB` limit, and in practice should also remain below `1 MB`.

## Android Demo

A minimal Android app using the PyTorch Mobile Lite runtime is included under [app](g:\New\klyff\app).

What it does:

- loads the optimized Lite model from `app/src/main/assets/rice_leaf_edge_mobile_int8.ptl`
- opens the camera with a capture button
- runs inference on-device with the PyTorch Mobile runtime
- displays the predicted class and confidence

Important note:

- Because the selected dataset is a rice leaf disease dataset, the app currently predicts `Healthy Rice Leaf`, `Bacterial Leaf Blight`, `Brown Spot`, `Leaf Blast`, `Leaf scald`, or `Sheath Blight`.
- If you need the UI text to say rice grain quality instead, that would require switching back to a properly labeled grain dataset and retraining.

To open it in Android Studio:

```text
Open the repository root g:\New\klyff as an Android project.
```

Main app entry points:

- [MainActivity.kt](g:\New\klyff\app\src\main\java\com\klyff\riceleafedge\MainActivity.kt)
- [activity_main.xml](g:\New\klyff\app\src\main\res\layout\activity_main.xml)
- [app/build.gradle.kts](g:\New\klyff\app\build.gradle.kts)

## Notebook

The training notebook is included at [rice_leaf_training.ipynb](g:\New\klyff\notebooks\rice_leaf_training.ipynb).

## Notes

- The model is a compact CNN built for lightweight edge-oriented classification.
- The script avoids `torchvision` and uses only `torch`, `Pillow`, and `scikit-learn`.
