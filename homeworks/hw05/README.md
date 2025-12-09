# Homework 5: Deploying Machine Learning Models

This folder contains the solution for Homework 5.

## Task Description

The goal is to deploy a Lead Scoring model using Flask/FastAPI and Docker.
Key steps:
1.  Environment setup (`uv`).
2.  Install dependencies (Scikit-Learn).
3.  Load pre-trained model (`pipeline_v1.bin`) and score a record.
4.  Serve model as a web service (FastAPI/Flask).
5.  Dockerize the service.
6.  Run Docker container and score a record.

For more details, see [hw5_task.md](hw5_task.md).

## Files

- `hw5_solve.py`: Script for training/verifying model logic (also adapted for MPS).
- `q6_app.py` / `main.py`: Likely the service implementation (check file list).
- `Dockerfile`: Configuration for Docker.
- `hw5_task.md`: Homework instructions.

## Usage

### Local Execution
1.  Install dependencies:
    ```bash
    uv sync
    ```
2.  Run the service (example):
    ```bash
    uv run q6_app.py
    ```

### Docker Execution
1.  Build image:
    ```bash
    docker build -t homework-hw5 .
    ```
2.  Run container:
    ```bash
    docker run -it --rm -p 9696:9696 homework-hw5
    ```
