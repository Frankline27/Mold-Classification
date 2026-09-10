# Mold Classification — Comparative CNN and Hybrid CNN-ML Study

This repository contains the full source code for the study:

> **"I-MOLD: Artificial Intelligence and Internet of Things (AI/IoDT)-based Framework for the Detection of Mold from Food Samples"**

The study evaluates 26 model configurations across an 11-class fruit/mould
classification task, comparing standalone CNN architectures, classical ML
baselines, and hybrid CNN + classical ML (KNN, SVM) pipelines.

Trained model weights for selected architectures are hosted on Hugging Face:
🤗 **[https://huggingface.co/NdahTah/MoldTwoPhaseClassification](https://huggingface.co/NdahTah/MoldTwoPhaseClassification)**

The real-time screening application is deployed via Streamlit.

---

## Repository Structure

```
Mold-Classification/
│
├── Standalone CNN / ML Baselines
│   ├── CustomCNN6.ipynb            # Custom 6-layered CNN
│   ├── CustomCNN6L.ipynb           # Custom 6-layered CNN (variant)
│   ├── CustomCNN9.ipynb            # Custom 9-layered CNN
│   ├── SVM.ipynb                   # SVM (flat pixel features)
│   ├── KNN.ipynb                   # KNN (flat pixel features)
│   ├── VGG19.ipynb                 # VGG19
│   ├── Vgg19.ipynb                 # VGG19 (variant run)
│   ├── Mobilenet.ipynb             # MobileNetV2
│   ├── MobileNetV2.ipynb           # MobileNetV2 (variant run)
│   ├── EfficientNetB0.ipynb        # EfficientNetB0
│   ├── DenseNet121.ipynb           # DenseNet121
│   ├── ResNet50.ipynb              # ResNet50
│   └── ResNet101.ipynb             # ResNet101
│
├── Hybrid CNN + KNN / SVM Pipelines
│   ├── CNN6SVMKNN.ipynb            # Custom 6-layered CNN + KNN / SVM
│   ├── CNN9SVMKNN.ipynb            # Custom 9-layered CNN + KNN / SVM
│   ├── CustomCNN9SVMKNN.ipynb      # Custom 9-layered CNN + KNN / SVM 
│   ├── VGG19SVMKNN.ipynb           # VGG19 + KNN / SVM
│   ├── MobileNetV2SVMKNN.ipynb     # MobileNetV2 + KNN / SVM
│   ├── EfficientNetB0SVMKNN.ipynb  # EfficientNetB0 + KNN / SVM
│   ├── EfficientNetBoSVMKNN.ipynb  # EfficientNetB0 + KNN / SVM 
│   ├── DenseNet121SVMKNN.ipynb     # DenseNet121 + KNN / SVM
│   ├── DenseNetSVMKNN.ipynb        # DenseNet121 + KNN / SVM 
│   ├── ResNet50SVMKNN.ipynb        # ResNet50 + KNN / SVM
│   └── ResNet101SVMKNN.ipynb       # ResNet101 + KNN / SVM
│
├── Deployment
│   ├── app.py                      # Main Streamlit deployment app
│   ├── app2.py                     # Binary Deployment app 
│   └── app3.py                     # Multiclass Deployment app 
│
├── Data Splitting
│   ├── split.ipynb                 # Dataset splitting utility Binary
│   └── split_data.ipynb            # Dataset splitting utility Multiclass 
│
├── Model Weights (in-repo)
│   ├── efficientnetb0_binary_run5_best.keras
│   └── efficientnetb0_run5_best.keras
│
└── requirements.txt                # Pinned dependency versions
```

---

## Models Compared

Each hybrid notebook (`*SVMKNN.ipynb`) covers both the KNN and SVM variants
for that architecture in a single file.

| Model | Type | Notebook |
|---|---|---|
| Custom 6-layered CNN | Standalone CNN | `CustomCNN6.ipynb` |
| Custom 9-layered CNN | Standalone CNN | `CustomCNN9.ipynb` |
| SVM | Classical ML | `SVM.ipynb` |
| KNN | Classical ML | `KNN.ipynb` |
| VGG19 | Standalone CNN | `VGG19.ipynb` |
| MobileNetV2 | Standalone CNN | `MobileNetV2.ipynb` |
| EfficientNetB0 | Standalone CNN | `EfficientNetB0.ipynb` |
| DenseNet121 | Standalone CNN | `DenseNet121.ipynb` |
| ResNet50 | Standalone CNN | `ResNet50.ipynb` |
| ResNet101 | Standalone CNN | `ResNet101.ipynb` |
| Custom 6-layered CNN + KNN | Hybrid CNN-ML | `CNN6SVMKNN.ipynb` |
| Custom 9-layered CNN + KNN | Hybrid CNN-ML | `CNN9SVMKNN.ipynb` |
| VGG19 + KNN | Hybrid CNN-ML | `VGG19SVMKNN.ipynb` |
| MobileNetV2 + KNN | Hybrid CNN-ML | `MobileNetV2SVMKNN.ipynb` |
| EfficientNetB0 + KNN | Hybrid CNN-ML | `EfficientNetB0SVMKNN.ipynb` |
| DenseNet121 + KNN | Hybrid CNN-ML | `DenseNet121SVMKNN.ipynb` |
| ResNet50 + KNN | Hybrid CNN-ML | `ResNet50SVMKNN.ipynb` |
| ResNet101 + KNN | Hybrid CNN-ML | `ResNet101SVMKNN.ipynb` |
| Custom 6-layered CNN + SVM | Hybrid CNN-ML | `CNN6SVMKNN.ipynb` |
| Custom 9-layered CNN + SVM | Hybrid CNN-ML | `CNN9SVMKNN.ipynb` |
| VGG19 + SVM | Hybrid CNN-ML | `VGG19SVMKNN.ipynb` |
| MobileNetV2 + SVM | Hybrid CNN-ML | `MobileNetV2SVMKNN.ipynb` |
| EfficientNetB0 + SVM | Hybrid CNN-ML | `EfficientNetB0SVMKNN.ipynb` |
| DenseNet121 + SVM | Hybrid CNN-ML | `DenseNet121SVMKNN.ipynb` |
| ResNet50 + SVM | Hybrid CNN-ML | `ResNet50SVMKNN.ipynb` |
| ResNet101 + SVM | Hybrid CNN-ML | `ResNet101SVMKNN.ipynb` |

---

## Trained Model Weights

Selected trained model weights are publicly available on Hugging Face:
🤗 **[https://huggingface.co/NdahTah/MoldTwoPhaseClassification](https://huggingface.co/NdahTah/MoldTwoPhaseClassification)**

| File | Architecture | Task |
|---|---|---|
| `efficientnetb0_binary_run5_best_final.h5` | EfficientNetB0 | Binary (mould/no-mould) |
| `efficientnetb0_run5_best_final.h5` | EfficientNetB0 | Multiclass (11-class) |
| `mobilenetv2_binary_run5_best_final.h5` | MobileNetV2 | Binary |
| `mobilenetv2_run5_best_final.h5` | MobileNetV2 | Multiclass |
| `densenet121_binary_final.h5` | DenseNet121 | Binary |
| `densenet121_multiclass_final.h5` | DenseNet121 | Multiclass |

---

## Data Integrity and Leakage Prevention

The following measures were applied throughout the study to ensure
experimental integrity and prevent data leakage:

- **Duplicate removal**: DupeGuru was used to scan the full image pool
  for near-duplicate and exact-duplicate images using perceptual hashing
  before any split was performed, eliminating cross-split contamination
  from images appearing under different filenames across public datasets.

- **Strict three-way split**: All data was partitioned into
  non-overlapping `train/`, `val/`, and `test/` folders using the
  splitting scripts (`split.ipynb` / `split_data.ipynb`) with a fixed
  random seed for reproducibility.

- **Test set withheld until after training**: `data/test` was not
  loaded, referenced, or evaluated at any point during training or
  hyperparameter selection. It was introduced only in the evaluation
  phase, after all training decisions had been finalised based on
  `data/val` performance alone.

- **Augmentation applied to train only**: Data augmentation was applied
  exclusively to `data/train`. Validation and test generators received
  preprocessing only — no augmentation.

---

## Reproducibility for (requirement.txt)
streamlit,

tensorflow-cpu==2.15.0,

numpy,

pillow,

requests,

h5py==3.10.0

### Environment

All experiments were conducted in a Miniconda `tf-cpu` environment on
Windows. The exact dependency versions are pinned in `requirements.txt`.

### Installation

```bash
pip install -r requirements.txt
```

The app downloads model weights from Hugging Face automatically on
first run.

### Running a training notebook

1. Open the relevant `.ipynb` file in Jupyter Notebook or JupyterLab
2. Update the `DATA_DIR` path variable at the top of the notebook to
   point to your local dataset copy
3. Run all cells in order

---

## Dataset

The dataset covers 11 classes of fruits/fungi for mould detection:
a binary phase (mould present / not present) followed by an 11-class
multiclass phase identifying the specific fruit or mould type.

Dataset splits were performed using `split.ipynb` .

---
