from fastapi.testclient import TestClient
from src.app import app
import os

# Ensure model exists for the test
if not os.path.exists("models/model.bin"):
    print("Model not found, running training...")
    from src.train import train

    train()

client = TestClient(app)


def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}
    print("Health check passed.")


def test_predict():
    payload = {
        "anxiety_level": 10,
        "self_esteem": 20,
        "mental_health_history": 0,
        "depression": 10,
        "headache": 2,
        "blood_pressure": 1,
        "sleep_quality": 2,
        "breathing_problem": 1,
        "noise_level": 2,
        "living_conditions": 3,
        "safety": 3,
        "basic_needs": 2,
        "academic_performance": 3,
        "study_load": 2,
        "teacher_student_relationship": 3,
        "future_career_concerns": 3,
        "social_support": 2,
        "peer_pressure": 3,
        "extracurricular_activities": 3,
        "bullying": 2,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "stress_level" in data
    assert "stress_level_code" in data
    print(f"Prediction test passed. Result: {data}")


if __name__ == "__main__":
    test_read_main()
    test_predict()
