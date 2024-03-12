# CoMM

This is the official code of the work **Enhanced Online Test-time Adaptation with Feature-Weight Cosine Alignment (2024)**

# How to use:
## Environment setup
* Python 3.8
* PyTorch 1.9.0

## :package: Dataset
Download datasets from [here](https://github.com/hendrycks/robustness) or using the links below:
* [CIFAR-10-Corrupted](https://zenodo.org/record/2535967)
* [CIFAR-100-Corrupted](https://zenodo.org/record/3555552)
* [ImageNet-Corrupted](https://zenodo.org/record/2235448)

## :clock4: Training
Run `bash script/<dataset>.sh` for training.

:warning: Make sure the path to the dataset's directory and pre-trained weights are correct in the script.
