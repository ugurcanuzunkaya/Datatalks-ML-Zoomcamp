
# 🏥 Quit & Save - Insurance Predictor (Capstone 1)

This project is a web application that predicts individual medical insurance costs and calculates potential savings from lifestyle changes (e.g., quitting smoking, optimizing BMI).

## 🚀 Key Features

- **Cost Prediction**: Uses Machine Learning (Random Forest) to estimate insurance premiums.
- **Savings Analysis**: Automatically identifies and quantifies savings from lifestyle improvements.
- **Deep EDA**: Comprehensive analysis of factors driving insurance costs.
- **MLOps Integration**: Experiment tracking with **MLflow**.
- **Modern Stack**: Built with **FastAPI**, **Pydantic**, **uv**, and **scikit-learn**.

## 📈 Data Trends & Insights

Key takeaways from our exploratory data analysis:

- **Cost Distribution**: Insurance charges are heavily right-skewed (log-normal). We apply log-transformation during training to normalize the target variable.
- **The Smoker Effect**: Smoking is the single biggest driver of cost. Smokers have a distinct, much higher cost distribution compared to non-smokers.
- **The BMI "Double Whammy"**: Calculating BMI is crucial. For non-smokers, BMI has a moderate linear effect. For smokers, high BMI (>30) triggers a massive spike in costs, creating a distinct "high-risk" cluster.

## 🛠️ Tech Stack

- **Language**: Python 3.12+
- **Dependency Management**: `uv`
- **Web Framework**: FastAPI + Jinja2 Templates
- **Machine Learning**: Scikit-Learn, XGBoost
- **Experiment Tracking**: MLflow
- **Visualization**: Seaborn, Matplotlib

## 🏃‍♂️ How to Run

### Local Development

1. **Clone & Setup**

   ```bash
   cd capstone1
   uv sync
   ```

2. **Analysis (EDA)**
   Open `notebooks/eda_deep_dive.ipynb` in VS Code or Jupyter to view the deep dive analysis.

3. **Train Model**
   Run the training script (experiments are tracked in the `mlruns` folder, View with `uv run mlflow ui`):

   ```bash
   uv run python src/train.py
   ```

   This will save the best model to `models/best_model.pkl`.

4. **Run Web App**

   ```bash
   uv run uvicorn src.app:app --reload
   ```

   Open [http://localhost:8000](http://localhost:8000) in your browser.

### Docker Deployment

1. **Build Image**

   ```bash
   docker build -t quit-and-save .
   ```

2. **Run Container**

   ```bash
   docker run -p 8000:8000 quit-and-save
   ```

   Access at [http://localhost:8000](http://localhost:8000).

## 📊 Evaluation & Metrics

The model is evaluated using **5-Fold Cross-Validation**. Experiments compare Linear Regression, Random Forest, and XGBoost. The XGBoost model performed best with an RMSE around **$4564**.

## 📁 Project Structure

- `src/`: Source code for app, training, and data loading.
  - `models.py`: Pydantic schemas.
  - `predict.py`: Business logic for savings calculation.
- `notebooks/`: Jupyter notebooks for EDA.
- `templates/`: HTML templates for the UI.
- `models/`: Serialized model artifacts.
