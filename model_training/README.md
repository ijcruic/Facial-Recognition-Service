# Facial Recognition Embedding Training & NNI Hyperparameter Search

## Overview

This repository contains everything you need to train a supervised‐contrastive (SupCon) image embedding model (based on InceptionResnetV1) and to run automated hyperparameter searches using Microsoft NNI. The three main files are:

- **`train_nni.py`**  
  The main Python script that (1) reads dataset paths and default hyperparameters from `config.yaml`, (2) builds/train/tests a SupConContrastive embedding model, (3) integrates NNI’s `get_next_parameter` and `report_*_result` calls so you can either run with defaults or launch an NNI experiment.

- **`config.yaml`**  
  General configuration for dataset directories, default hyperparameters, scheduler/logging settings, and fine-tuning parameters. If you run `train_nni.py` without NNI, it uses these values.

- **`nni_config.yaml`**  
  Consolidated NNI experiment configuration. This single file defines the search space (ranges or distributions for “temperature,” “learning_rate,” “batch_size,” etc.), plus tuner settings, concurrency, trial command, and so on. When you do `nnictl create --config nni_config.yaml`, NNI will automatically override the values in `config.yaml` for each trial.

Below are step-by-step instructions on how to train the model with standard settings and how to launch an NNI experiment.

---

## File Summaries

### 1. `train_nni.py`

- **Purpose:**  
  - Loads `config.yaml` (dataset paths + default hyperparameters).  
  - Optionally reads NNI parameters (`nni.get_next_parameter()`) and overrides defaults.  
  - Builds Albumentations‐based data pipelines (ContrastiveImageDataset for training, AlbumentationsImageDataset for testing).  
  - Instantiates a pre-trained Facenet (`InceptionResnetV1`) backbone, wraps it in `torch.nn.DataParallel` if >1 GPU is available, and moves it to the appropriate device.  
  - Defines a two-stage fine-tuning loop:  
    1. First `first_stage_epochs`, freeze all backbone parameters except the projection & batch norm layers.  
    2. After that, unfreeze everything.  
  - Uses SupConLoss (with temperature drawn from either `config.yaml` or NNI) plus Adam (learning_rate, weight_decay) and a CosineAnnealingLR scheduler.  
  - Logs metrics to TensorBoard.  
  - Reports intermediate/final Precision@1 to NNI.  
  - Saves the best checkpoint whenever Precision@1 improves.

- **Usage Patterns:**  
  1. **Standard training (no NNI):** reads all hyperparameters from `config.yaml`.  
  2. **NNI‐driven training:** if you launch this script via NNI, each trial’s hyperparameters come from `nni_config.yml`, override `config.yaml`, and results are reported back to NNI automatically.

---

### 2. `config.yaml`

- **Purpose:** Provides single-source defaults for:
  1. **`datasets:`**  
     - `train_dirs:` list of 1–n folders (each is passed to `ContrastiveImageDataset`).  
     - `test_dir:` a single folder (passed to `AlbumentationsImageDataset`).
  2. **`pretrained:`**  
     - `weights_path:` path to the pre-trained Facenet checkpoint.
  3. **`dataloader:`**  
     - `num_workers:` number of worker threads for PyTorch DataLoader.
  4. **`scheduler:`**  
     - `eta_min:` minimum LR for CosineAnnealingLR.
  5. **`logging:`**  
     - `log_dir_prefix:` prefix for TensorBoard log directories (e.g. `logs/`).
  6. **`finetune:`**  
     - `first_stage_layers:` list of substrings (e.g. `["last_linear","last_bn"]`) – these are the only parameters unfrozen for the first stage.
  7. **`hyperparameters:`** (defaults, overridden by NNI if launched under NNI)  
     - `batch_size:` default batch size for DataLoader  
     - `learning_rate:` default Adam learning rate  
     - `num_epochs:` total number of training epochs  
     - `first_stage_epochs:` number of “freeze backbone except final layers” epochs  
     - `weight_decay:` default Adam weight decay  
     - `temperature:` SupConLoss temperature  

---

### 3. `nni_config.yml`

- **Purpose:** Contains everything NNI needs for an experiment:
  1. **Metadata:**  
     - `authorName:` your name (for NNI UI).  
     - `experimentName:` a memorable name for this hyperparameter search.
  2. **Trial settings:**  
     - `trialCommand: python train_nni.py` (NNI will run this command for each trial).  
     - `trialCodeDirectory: .` (current folder).
  3. **Concurrency & Trial limits:**  
     - `trialConcurrency:` how many trials to run in parallel (e.g. `2`).  
     - `maxTrialNumber:` how many total trials before stopping (e.g. `50`).
  4. **`searchSpace:`** (all six hyperparameters):  
     - `temperature:` `_type: uniform`, `_value: [0.05, 0.5]`  
     - `learning_rate:` `_type: loguniform`, `_value: [0.0001, 0.1]`  
     - `batch_size:` `_type: choice`, `_value: [128, 256, 512, 1024]`  
     - `num_epochs:` `_type: choice`, `_value: [100, 150, 200]`  
     - `first_stage_epochs:` `_type: choice`, `_value: [20, 50, 80]`  
     - `weight_decay:` `_type: loguniform`, `_value: [0.000001, 0.001]`
5. **Tuner:**  
     - Using `TPE` with `optimize_mode: maximize` (so NNI tries to maximize Precision@1).

---

## Prerequisites

1. **Python ≥ 3.8**  
2. **PyTorch ≥ 1.9** (tested on 1.12+) with CUDA support if you have GPUs.  
3. **CUDA drivers** installed (if you have NVIDIA GPUs).  
4. **Required packages:**  
   ```bash
   pip install \
     torch torchvision tensorboard \
     albumentations numpy tqdm pyyaml \
     pytorch-metric-learning \
     nni
