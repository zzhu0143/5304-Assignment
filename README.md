# 5304-Assignment
Project 1. Denoising Network In this project, you're going to implement a neural network to denoise images, there are several parts you need to implement to make the whole pipeline complete.  1.Dataset2. Metric,3. Networks 4.Training5. Additional Question
# Project 1 — Denoising Network

## Overview

This project implements an image denoising pipeline using a CNN-based neural network trained on Gaussian noise. The implementation covers dataset preprocessing, PSNR metrics, network design, training, and an additional evaluation on speckle noise.

---

## Implementation Notes

### Dataset

`TrainingSet` and `TestingSet` both inherit from `torch.utils.data.Dataset`. Images are loaded using `PIL.Image.open` and converted to grayscale (`convert('L')`), then transformed to tensors of shape `[1, 180, 180]` via `torchvision.transforms.ToTensor`.

The 400 images are sorted alphabetically before splitting, so the train/test partition (350/50) is deterministic across environments. Noise is added per the provided formula using `torch.randn` scaled by `sigma/255.0`, with pixel values clamped to `[0, 1]`.

The download cell was rewritten using `requests` instead of `wget` because the original Dropbox link was no longer valid, and the updated URL extracts to a folder named `ImageSet-1` rather than `ImageSet`. The script handles the rename automatically and checks file size to catch failed downloads early.

---

### Metrics

PSNR is computed as:

```
PSNR = 10 × log10(1 / MSE)
```

since pixel values are normalised to `[0, 1]` (MAX = 1). A guard is included for the case where MSE = 0 (identical images), returning infinity. The function returns a PyTorch tensor to stay consistent with the rest of the pipeline.

The result on the provided test case (`[0.1,0.2,0.3,0.4]` vs `[0.5,0.6,0.7,0.8]`) is approximately **7.96**, matching the expected output.

---

### Network

The network follows the DnCNN architecture. It takes a noisy image as input, passes it through several convolutional layers to produce a predicted clean image, then returns the residual noise as `input - predicted_clean`. This residual formulation makes the learning target simpler since noise tends to be a higher-frequency signal than the underlying image content.

**Architecture (7 layers, 64 channels):**

| Layer      | Components                  |
| ---------- | --------------------------- |
| First      | Conv2d → ReLU               |
| Middle × 5 | Conv2d → BatchNorm2d → ReLU |
| Last       | Conv2d                      |

All convolutional layers use kernel size 3 with padding 1 to preserve spatial dimensions. BatchNorm is included in the middle layers to stabilise training. The final layer has no activation so the output can represent negative noise values.

Weights are initialised with `torch.nn.init.orthogonal_` and biases with `torch.nn.init.constant_(..., 0)` as specified.

A `device` property is added to the network class so `mean_psnr` can automatically move tensors to the correct device without external device tracking.

The template's `.cuda()` call is replaced with `.to(device)` where `device` is detected at runtime, allowing the notebook to run on CPU if no GPU is available.

---

### Training

| Setting     | Value                                           |
| ----------- | ----------------------------------------------- |
| Batch size  | 32                                              |
| Epochs      | 30                                              |
| Optimiser   | Adam, lr = 0.001                                |
| Loss        | MSELoss                                         |
| LR schedule | MultiStepLR, milestones = [10, 20], gamma = 0.1 |

The learning rate is reduced at epoch 10 and epoch 20 to allow finer convergence in later stages. The training target is `true_noise = noisy - original` (computed after clamping), which reflects the actual pixel-level difference rather than the raw sampled noise.

PSNR on the test set is logged at the end of every epoch to monitor training progress.

---

### Additional Question — Speckle Noise

Speckle noise is multiplicative: `noisy = img × N(1, sigma)`, which means brighter regions receive stronger noise compared to additive Gaussian noise of the same sigma.

**Part 1 — Choosing sigma values**

Six candidate sigma values were tested and the mean PSNR of the corrupted images was recorded. Three values were selected whose PSNR is approximately equal to that of the Gaussian-corrupted training images (~28 dB):

```
chosen_sigmas = [0.08, 0.10, 0.12]
```

**Part 2 — PSNR comparison table**

The trained model was evaluated on both Gaussian and speckle noise at the chosen sigma values. Results are printed as a table in the notebook and visualised in `speckle_vs_gaussian.png`.

**Part 3 — Conclusion (see PDF report)**

The model trained on Gaussian noise shows partial generalisation to speckle noise. At matched noise levels, the restored PSNR is lower than the Gaussian case, which is expected since the model has not seen multiplicative noise during training. The degree of recovery decreases as the speckle sigma increases.

---

## Files

| File                         | Description                                       |
| ---------------------------- | ------------------------------------------------- |
| `project_1_5304_fixed.ipynb` | Main notebook with all implementations            |
| `speckle_vs_gaussian.png`    | Visualisation output from the additional question |
| `[report].pdf`               | PDF report with analysis and results              |
