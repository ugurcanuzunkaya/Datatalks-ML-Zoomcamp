# Student Stress Level Classification

## Problem Description

Stress among students is a growing concern that affects mental health, academic performance, and overall well-being. Identifying stress levels early can help in providing timely support and interventions.

This project aims to build a machine learning model to predict the **stress level** of students based on various factors:

* **Psychological**: Anxiety, self-esteem, depression, mental health history.
* **Physiological**: Headaches, blood pressure, sleep quality.
* **Environmental/Academic**: Noise level, living conditions, study load, social support.

The model classifies stress into three categories: **Low**, **Medium**, and **High**.

## Project Structure

The project is modularized following best practices:

```
capstone2/
├── src/
│   ├── app.py          # FastAPI application serving the model and UI
│   ├── models.py       # Pydantic data models
│   ├── predict.py      # Prediction logic and model loader
│   └── train.py        # Model training script with MLflow tracking
├── models/             # Directory for saved models
├── notebooks/          # Jupyter notebooks for EDA
├── templates/          # HTML templates for the UI
├── requirements.txt    # (Managed via uv)
├── Dockerfile          # Container definition
└── README.md
```

## Dataset

We utilize the **StressLevelDataset.csv** which contains numeric data for 1100 students.

* **Target**: `stress_level` (0: Low, 1: Medium, 2: High)
* **Features**: 20 distinct features covering the aspects mentioned above.

## Results and Conclusion

* **EDA Insights**: `StressLevelDataset.csv` is balanced and appropriate for this classification task.
* **Model Performance**:
  * **XGBoost Model**: Achieved an accuracy of **86.82%** on the test set.
* **Conclusion**: The XGBoost model was selected for deployment due to its robust performance.

## How to Run

### Prerequisities

* Python 3.13
* `uv` (Universal Package Manager)
* Docker

### Setup

1. Clone the repository:

    ```bash
    git clone <repo-url>
    cd capstone2
    ```

2. Install dependencies:

    ```bash
    uv sync
    ```

### Training

To train the model and log experiments to MLflow:

```bash
uv run -m src.train
```

### Running the Application (API + UI)

#### Locally

Start the server:

```bash
uv run uvicorn src.app:app --port 8000 --reload
```

* **UI**: Open `http://localhost:8000` in your browser to use the prediction form.
* **API Docs**: Open `http://localhost:8000/docs` for Swagger UI.

#### With Docker

1. Build the image:

    ```bash
    docker build -t stress-predictor .
    ```

2. Run the container:

    ```bash
    docker run -p 8000:8000 stress-predictor
    ```
