# Homework 9: Serverless Deep Learning

This folder contains the solution for Homework 9.

## Task Description

The goal is to deploy the Hair Type classification model (Straight vs Curly) as a serverless function using Docker and AWS Lambda (simulated).
Key steps:
1.  Model conversion (already provided as ONNX).
2.  Model inspection (input/output names, shapes).
3.  Image preprocessing (resize, normalize).
4.  Inference (run on local machine).
5.  Docker deployment (base image analysis, creating Dockerfile).
6.  Lambda simulation (handling events).

For more details, see [hw9_task.md](hw9_task.md).

## Files

- `src/inspection.py`: Code for Q1-Q4 (Model inspection, Preprocessing, Inference).
- `src/lambda_function.py`: Code for Q6 (Lambda handler).
- `src/model_utils.py`: Shared utilities for image processing.
- `Dockerfile`: Deployment configuration.
- `hw9_task.md`: Homework instructions.

## Usage

### Local Inference (Q1-Q4)
1.  Install dependencies:
    ```bash
    uv sync
    ```
2.  Run inspection script:
    ```bash
    uv run src/inspection.py
    ```

### Docker Deployment (Q5-Q6)
1.  Build the image:
    ```bash
    docker build -t homework-hw9 .
    ```
2.  Run the container:
    ```bash
    docker run -it --rm -p 8080:8080 homework-hw9
    ```
3.  Test the function (in another terminal):
    ```bash
    curl -XPOST "http://localhost:8080/2015-03-31/functions/function/invocations" -d '{"url": "https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"}'
    ```
