# Rice Leaf Edge Classifier

PyTorch rice leaf disease classification project with a baseline model, a PyTorch Mobile edge model, and a minimal Android demo for on-device inference.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Dataset

Dataset used: `anshulm257/rice-disease-dataset` from Kaggle.

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

## Inference

```bash
python predict.py path\to\leaf_image.jpg --checkpoint models\baseline\rice_leaf_edge_model.pt --top-k 3
```

## Edge Export

```bash
python export_edge.py --checkpoint models\baseline\rice_leaf_edge_model.pt --output-dir edge_artifacts
```

Final edge model:
- [rice_leaf_edge_mobile_int8.ptl](g:\New\klyff\models\edge\rice_leaf_edge_mobile_int8.ptl)

## Android Demo

Open the repo root in Android Studio and run the app in [app](g:\New\klyff\app). It loads the PyTorch Mobile `.ptl` model and runs inference locally after capturing a leaf image.

## Model Files

- Baseline: [rice_leaf_edge_model.pt](g:\New\klyff\models\baseline\rice_leaf_edge_model.pt)
- Edge: [rice_leaf_edge_mobile_int8.ptl](g:\New\klyff\models\edge\rice_leaf_edge_mobile_int8.ptl)
- Notebook: [rice_leaf_training.ipynb](g:\New\klyff\notebooks\rice_leaf_training.ipynb)

## Trade-off Analysis

Hardware used for benchmark:
- `Windows 11 10.0.26200`
- `Intel64 Family 6 Model 142 Stepping 11, GenuineIntel`
- `7.85 GB RAM`
- `8` CPU threads visible to Python

| Model | File Size (MB) | Accuracy (%) | Inference Speed (ms) |
| --- | ---: | ---: | ---: |
| Baseline | 0.211 | 79.37 | 4.70 |
| Edge | 0.276 | 79.50 | 3.75 |

Edge model under `1 MB` was achieved by using a compact depthwise-separable CNN, `128x128` inputs, global average pooling, PyTorch Lite export, and dynamic int8 quantization on the classifier head.
