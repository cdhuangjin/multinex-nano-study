# Multinex-Nano training-intervention study

Minimal research reproduction for **An Empirical Evaluation of Training-time Interventions for Extreme-Lightweight Low-Light Image Enhancement** (Jin Huang and Qisen Gao).

## Scope
This release provides the baseline/Rep-Nano model, branch ablations A/B, matched training runner, fixed data split with image hashes, five pairs of 3,000-step endpoint weights, CPU evaluation, equivalent folding, ONNX export, alternating timing, and selected original numerical records. It is not an official Multinex release. The distillation screening is not implemented by this minimal release; the full historical research workspace is not included. No dataset images or VGG weights are redistributed.

## Setup
Python 3.12 was used; the validation environment used PyTorch 2.13.0+cu130. Install a PyTorch/torchvision pair appropriate for your hardware. Training requires CUDA; evaluation and folding run on CPU.

```sh
python -m pip install -r requirements.txt
git clone https://github.com/albrateanu/multinex.git code
git -C code checkout e15790a963935569fff1ba6a8b0d33a791765749
```

The upstream architecture, metrics, HybridLoss and scheduler are loaded unchanged from this pinned checkout. Upstream code is governed by the PolyForm Noncommercial License, reproduced in `UPSTREAM_LICENSE.txt`; observe upstream attribution and terms. This repository does not grant a replacement license for upstream components.

## Existing data and cache
Obtain LOL/LOL-v2 from their authors: https://github.com/flyywh/CVPR-2020-Semi-Low-Light . Use an existing intact dataset instead of downloading it again. The examples use the study's external storage convention `E:/Datasets`; change this to your external data directory on other systems.

```sh
python prepare.py --data-root E:/Datasets/LOL_v1_original_author --torch-home E:/Datasets/pretrained_model_cache/torch
```

The directory must contain `our485/low`, `our485/high`, `eval15/low`, and `eval15/high`. Preparation verifies all 485 training/development pairs against the published SHA256 hashes and writes ignored machine-local paths. The fixed split is 435 training / 50 development. The official 15-image test split had been inspected during earlier reproduction and is not an untouched holdout.

Training also requires the official torchvision VGG19 IMAGENET1K_V1 file `vgg19-dcbb9e9d.pth` at `<torch-home>/hub/checkpoints/`. Reuse an existing verified cache; the runner checks its hash prefix and fails if it is absent. No implicit dataset download is performed.

## Evaluate supplied endpoint weights (no retraining)
```sh
python evaluate.py --arm rep --checkpoint checkpoints/rep-s110.pth --low E:/Datasets/LOL_v1_original_author/eval15/low --high E:/Datasets/LOL_v1_original_author/eval15/high --output test15.csv
python evaluate.py --arm baseline --checkpoint checkpoints/baseline-s110.pth --low E:/Datasets/LOL_v1_original_author/our485/low --high E:/Datasets/LOL_v1_original_author/our485/high --names dev.txt --output dev.csv
```
The first command was verified on CPU: PSNR 18.1433825157 dB, SSIM 0.7357818539. Use matching low/high image filenames. Checkpoints contain only model tensors and step number; `checkpoints/manifest.json` records their hashes and hashes of original checkpoints. No optimizer states or private paths are included.

## Paired training
```sh
python train.py --arm baseline --seed 110
python train.py --arm rep --seed 110
```
Repeat for 101, 102, 103, 104. For exploratory branch controls use `--arm A --seed 110` (3x3 + 1x1 with BN) and `--arm B --seed 110` (3x3 + identity with BN). Output directories must not already exist. Runs use 3,000 updates, batch 2, 256 crops, fixed step-indexed sampling/mixup, original loss and scheduler, and validation every 500 updates. Use `latest.pth` for the fixed endpoint, not the development-selected `best.pth`. Portability edits affect imports/paths/CLI only; no new full training was executed for this release. Numerical reproducibility can depend on the hardware/software stack.

## Folding and timing
```sh
python rep_convert.py
python rep_convert.py --rep-checkpoint checkpoints/rep-s110.pth --baseline-checkpoint checkpoints/baseline-s110.pth --export
python benchmark.py
```
The first command tests fresh model equivalence and asserts 696 baseline / 744 training / 696 folded parameters. The second uses trained weights and exports fixed 256x256 ONNX graphs. The last performs 20 warmups and 10 alternating blocks of 100 calls per model, one CPU thread, and writes raw timings and a figure. Timing is machine-specific, not an end-to-end camera measurement.

## Interpretation
The five-seed development mean paired PSNR change is -0.9656 dB (sample SD 1.2430 dB). These results describe this architecture, data split and budget; they do not establish a general sub-1K capacity threshold or a causal BN mechanism. Single-seed branch ablations do not isolate variance mechanisms. Historical timing records show no demonstrated intrinsic CPU acceleration. Original records are in `results/`; new outputs should be kept separately.
