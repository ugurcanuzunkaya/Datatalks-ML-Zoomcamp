# Homework 8: Deep Learning

This folder contains the solution for Homework 8.

## Task Description

The goal is to build a CNN model for classifying hair types (Straight vs Curly) using PyTorch.
Key steps:
1.  Data preparation (download, unzip, `torchvision.transforms`).
2.  Reproducibility setup (seeds).
3.  Model definition (CNN architecture with Conv2d, MaxPool, Linear).
4.  Training loop (optimizer, loss function, manual loop).
5.  Evaluation (median accuracy, standard deviation of loss).
6.  Data Augmentation (training with augmented data).

For more details, see [hw8_task.md](hw8_task.md).

## Files

- `Datatalks - ML Zoomcamp - HW8.ipynb`: Jupyter Notebook with the solution.
- `hw8_task.md`: Homework instructions.

## Usage

1.  Open `Datatalks - ML Zoomcamp - HW8.ipynb`.
2.  Run cells (requires PyTorch).

## Setup

Download dataset:
```bash
wget https://github.com/SVizor42/ML_Zoomcamp/releases/download/straight-curly-data/data.zip
unzip data.zip
```
