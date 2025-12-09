# ML Zoomcamp Homeworks

This repository contains my solutions for the DataTalks.Club Machine Learning Zoomcamp.

## Structure

The repository is organized by homework number:

- **[Homework 1](homeworks/hw01/):** Intro to Machine Learning
- **[Homework 2](homeworks/hw02/):** Machine Learning for Regression
- **[Homework 3](homeworks/hw03/):** Machine Learning for Classification
- **[Homework 4](homeworks/hw04/):** Evaluation Metrics for Classification
- **[Homework 5](homeworks/hw05/):** Deploying Machine Learning Models
- **[Homework 6](homeworks/hw06/):** Decision Trees and Ensemble Learning
- **[Homework 8](homeworks/hw08/):** Deep Learning
- **[Homework 9](homeworks/hw09/):** Serverless Deep Learning

## Setup

Each homework directory is self-contained. For homeworks requiring Python scripts (like HW9), I use `uv` for dependency management.

```bash
# Example for HW9
cd homeworks/hw09
uv sync
uv run src/inspection.py
```
