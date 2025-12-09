# Homework 3: Machine Learning for Classification

This folder contains the solution for Homework 3.

## Task Description

The goal is to predict if a client will sign up (`converted`) using the Lead Scoring dataset.
Key steps:
1.  Data preparation (handle missing values, fill `NA`/`0`).
2.  EDA (mode, correlation matrix).
3.  Split data.
4.  Calculate Mutual Information scores.
5.  Train Logistic Regression (One-Hot Encoding, accuracy check).
6.  Feature Elimination.
7.  Hyperparameter Tuning (regularization `C`).

For more details, see [hw3_task.md](hw3_task.md).

## Files

- `DataTalks - ML Zoomcamp - HW3.ipynb`: Jupyter Notebook with the solution.
- `hw3_task.md`: Homework instructions.

## Usage

1.  Open `DataTalks - ML Zoomcamp - HW3.ipynb`.
2.  Run cells to perform classification tasks.

## Setup

Download dataset:
```bash
wget https://raw.githubusercontent.com/alexeygrigorev/datasets/master/course_lead_scoring.csv
```
