import pickle
import numpy as np

np.random.seed(42)

# Load the pipeline
with open("pipeline_v1.bin", "rb") as f_in:
    pipeline = pickle.load(f_in)

# Client to score
client = {
    "lead_source": "paid_ads",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0,
}

# The model expects a list of dictionaries
X = [client]
# Get the probability of the positive class (1)
prob = pipeline.predict_proba(X)[0, 1]

print(f"The probability of conversion is: {prob:.3f}")
