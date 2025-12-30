from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from src.models import StressInput, PredictionOutput
from src.predict import Predictor
import os

app = FastAPI(title="Student Stress Level Predictor", version="1.0.0")

predictor = None
MODEL_PATH = "models/model.bin"

# Setup templates
templates = Jinja2Templates(directory="templates")


@app.on_event("startup")
def load_model():
    """
    Load the model on application startup.
    Initializes the global `predictor` instance.
    """
    global predictor
    if os.path.exists(MODEL_PATH):
        predictor = Predictor(MODEL_PATH)
        print(f"Model loaded from {MODEL_PATH}")
    else:
        print(f"Warning: Model not found at {MODEL_PATH}. Training may be required.")


@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    """
    Serve the input form (UI).
    """
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
def health_check():
    """
    Health check endpoint to verify service status and model loading.
    """
    return {"status": "ok", "model_loaded": predictor is not None}


@app.post("/predict", response_model=PredictionOutput)
def predict_api(input_data: StressInput):
    """
    Predict stress level via JSON API.

    Args:
        input_data (StressInput): JSON payload containing stress factors.

    Returns:
        PredictionOutput: JSON response with prediction results.
    """
    if not predictor:
        return {"error": "Model not loaded"}

    return predictor.predict_one(input_data)


@app.post("/predict_ui", response_class=HTMLResponse)
async def predict_ui(
    request: Request,
    anxiety_level: int = Form(...),
    self_esteem: int = Form(...),
    mental_health_history: int = Form(...),
    depression: int = Form(...),
    headache: int = Form(...),
    blood_pressure: int = Form(...),
    sleep_quality: int = Form(...),
    breathing_problem: int = Form(...),
    noise_level: int = Form(...),
    living_conditions: int = Form(...),
    safety: int = Form(...),
    basic_needs: int = Form(...),
    academic_performance: int = Form(...),
    study_load: int = Form(...),
    teacher_student_relationship: int = Form(...),
    future_career_concerns: int = Form(...),
    social_support: int = Form(...),
    peer_pressure: int = Form(...),
    extracurricular_activities: int = Form(...),
    bullying: int = Form(...),
):
    """
    Handle form submission from the UI, predict stress level,
    and render the result template.
    """
    if not predictor:
        return templates.TemplateResponse(
            "result.html",
            {"request": request, "result": {"stress_level": "Error: Model not loaded"}},
        )

    input_data = StressInput(
        anxiety_level=anxiety_level,
        self_esteem=self_esteem,
        mental_health_history=mental_health_history,
        depression=depression,
        headache=headache,
        blood_pressure=blood_pressure,
        sleep_quality=sleep_quality,
        breathing_problem=breathing_problem,
        noise_level=noise_level,
        living_conditions=living_conditions,
        safety=safety,
        basic_needs=basic_needs,
        academic_performance=academic_performance,
        study_load=study_load,
        teacher_student_relationship=teacher_student_relationship,
        future_career_concerns=future_career_concerns,
        social_support=social_support,
        peer_pressure=peer_pressure,
        extracurricular_activities=extracurricular_activities,
        bullying=bullying,
    )

    result = predictor.predict_one(input_data)

    return templates.TemplateResponse(
        "result.html", {"request": request, "result": result}
    )
