# Homework 2: Machine Learning for Regression

This folder contains the solution for Homework 2.

## Task Description

The goal is to create a regression model for predicting car fuel efficiency (`fuel_efficiency_mpg`) using the Car Fuel Efficiency dataset.
Key steps:
1.  Prepare dataset (filter columns, handle missing values).
2.  EDA (check distribution).
3.  Split data (60/20/20).
4.  Train Linear Regression models (baseline, mean imputation, 0 imputation).
5.  Train Regularized Linear Regression (tune `r`).
6.  Analyze seed influence.

For more details, see [hw2_task.md](hw2_task.md).

## Files

- `Datatalks - ML Zoomcamp - HW2.ipynb`: Jupyter Notebook with the solution.
- `hw2_task.md`: Homework instructions.

## Usage

1.  Open `Datatalks - ML Zoomcamp - HW2.ipynb` in Jupyter.
2.  Run cells to perform EDA, training, and evaluation.

## Setup

Download dataset:
```bash
wget https://raw.githubusercontent.com/alexeygrigorev/datasets/master/car_fuel_efficiency.csv
```
