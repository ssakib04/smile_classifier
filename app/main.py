import os
import uuid
from typing import List
from datetime import datetime
from PIL import Image
import io

from fastapi import FastAPI, Request, Form, UploadFile, File, Depends, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session

from app.database import engine, Base, get_db
from app import models, ml_pipeline

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Smile Classifier")

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

MODELS_DIR = "models_store"
UPLOADS_DIR = "app/static/uploads"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

def get_available_models():
    if not os.path.exists(MODELS_DIR):
        return []
    return [f.replace(".pkl", "") for f in os.listdir(MODELS_DIR) if f.endswith(".pkl")]

@app.get("/")
def home_page(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

# 2. TRAIN PAGE (GET)
@app.get("/train")
def train_page(request: Request):
    return templates.TemplateResponse("train.html", {"request": request})

# 2. TRAIN PAGE (POST)
@app.post("/train")
def train_model(
    request: Request,
    model_name: str = Form(...),
    smile_images: List[UploadFile] = File(...),
    not_smile_images: List[UploadFile] = File(...),
    db: Session = Depends(get_db)
):
    try:
        smile_bytes = []
        for file in smile_images:
            if file.filename:
                file.file.seek(0)
                content = file.file.read()
                if content:
                    smile_bytes.append(content)

        not_smile_bytes = []
        for file in not_smile_images:
            if file.filename:
                file.file.seek(0)
                content = file.file.read()
                if content:
                    not_smile_bytes.append(content)

        if not smile_bytes or not not_smile_bytes:
            return templates.TemplateResponse("train.html", {
                "request": request,
                "error": "Please upload at least one image for both classes."
            })

        acc_score = ml_pipeline.train_new_model(smile_bytes, not_smile_bytes, model_name, MODELS_DIR)

        log = models.TrainingLog(
            model_name=f"{model_name}.pkl",
            smile_count=len(smile_bytes),
            not_smile_count=len(not_smile_bytes),
            accuracy_score=acc_score
        )
        db.add(log)
        db.commit()

        return templates.TemplateResponse("train.html", {
            "request": request,
            "message": f"Model '{model_name}' trained successfully with accuracy: {acc_score*100:.1f}%"
        })

    except Exception as e:
        print(f"TRAINING ERROR: {str(e)}")
        return templates.TemplateResponse("train.html", {
            "request": request,
            "error": f"An error occurred during training: {str(e)}"
        })

# 3. CLASSIFY PAGE (GET)
@app.get("/classify")
def classify_page(request: Request):
    available_models = get_available_models()
    return templates.TemplateResponse("classify.html", {
        "request": request, 
        "models": available_models
    })

# 3. CLASSIFY PAGE (POST)
@app.post("/classify")
async def process_classification(
    request: Request,
    model_choice: str = Form(...),
    image_file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    if not image_file.filename:
        raise HTTPException(status_code=400, detail="No file selected.")

    contents = await image_file.read()
    
    img = Image.open(io.BytesIO(contents)).convert('RGB')
    filename = f"{uuid.uuid4().hex}.jpg"
    file_path = os.path.join(UPLOADS_DIR, filename)
    img.save(file_path, "JPEG")

    model_path = os.path.join(MODELS_DIR, f"{model_choice}.pkl")
    predicted_class, confidence = ml_pipeline.predict_smile(contents, model_path)

    relative_image_path = f"/static/uploads/{filename}"

    inf_log = models.InferenceLog(
        image_path=relative_image_path,
        predicted_class=predicted_class,
        confidence=confidence,
        model_used=f"{model_choice}.pkl"
    )
    db.add(inf_log)
    db.commit()

    return templates.TemplateResponse("result.html", {
        "request": request,
        "image_path": relative_image_path,
        "predicted_class": predicted_class,
        "confidence": confidence,
        "model_used": model_choice
    })

@app.get("/history")
def history_page(request: Request, db: Session = Depends(get_db)):
    train_logs = db.query(models.TrainingLog).order_by(models.TrainingLog.created_at.desc()).all()
    inference_logs = db.query(models.InferenceLog).order_by(models.InferenceLog.created_at.desc()).all()
    
    return templates.TemplateResponse("history.html", {
        "request": request,
        "train_logs": train_logs,
        "inference_logs": inference_logs
    })