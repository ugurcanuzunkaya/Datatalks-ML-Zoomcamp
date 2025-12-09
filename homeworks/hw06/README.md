# Homework 6: Decision Trees and Ensemble Learning

This folder contains the solution for Homework 6.

## Task Description

The goal is to predict car fuel efficiency using Tree-based models (Decision Tree, Random Forest, XGBoost).
Key steps:
1.  Data preparation (handle missing values, `DictVectorizer`).
2.  Decision Tree Regressor (find splitting feature).
3.  Random Forest Regressor (RMSE calculation, tuning `n_estimators`, `max_depth`).
4.  Feature Importance extraction.
5.  XGBoost Regressor (tuning `eta`).

For more details, see [hw6_task.md](hw6_task.md).

## Files

- `Datatalks_ML_Zoomcamp_HW6.ipynb`: Jupyter Notebook with the solution.
- `hw6_task.md`: Homework instructions.

## Usage

1.  Open `Datatalks_ML_Zoomcamp_HW6.ipynb`.
2.  Run cells to perform training and evaluation.

## Setup

Download dataset:
```bash
wget https://raw.githubusercontent.com/alexeygrigorev/datasets/master/car_fuel_efficiency.csv
```
