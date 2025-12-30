from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
from src.models import InsuranceInput
from src.predict import Predictor

app = FastAPI(title="Quit & Save Insurance Predictor")

templates = Jinja2Templates(directory="templates")

# Initialize Predictor
model_path = "models/best_model.pkl"
predictor = None


@app.on_event("startup")
def load_model():
    global predictor
    if os.path.exists(model_path):
        predictor = Predictor(model_path)
    else:
        print(f"Warning: Model not found at {model_path}. Please train model first.")
        # Create a dummy predictor or handle error appropriately in production


@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    age: int = Form(...),
    sex: str = Form(...),
    bmi: float = Form(...),
    children: int = Form(...),
    smoker: str = Form(...),
    region: str = Form(...),
):
    if not predictor:
        return templates.TemplateResponse("error.html", {"request": request, "message": "Model not loaded."})

    input_data = InsuranceInput(age=age, sex=sex, bmi=bmi, children=children, smoker=smoker, region=region)

    result = predictor.make_prediction(input_data)

    return templates.TemplateResponse("results.html", {"request": request, "result": result, "input": input_data})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
