import os
import uuid
import io
import json
import asyncio
from typing import List
from datetime import datetime
from PIL import Image, ImageOps

from fastapi import FastAPI, Request, Form, UploadFile, File, Depends, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.database import engine, Base, get_db
from app import models, ml_pipeline

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Smile Classifier Pro")

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

MODELS_DIR = "models_store"
UPLOADS_DIR = "app/static/uploads"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

def sanitize_and_compress_image(file_bytes: bytes, max_dim: int = 1024) -> bytes:
    try:
        img = Image.open(io.BytesIO(file_bytes))
        img = ImageOps.exif_transpose(img)

        if img.mode in ("RGBA", "P", "LA", "1"):
            img = img.convert("RGB")

        img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)

        output = io.BytesIO()
        img.save(output, format="JPEG", quality=85, optimize=True)
        return output.getvalue()
    except Exception:
        return None

def get_available_models():
    if not os.path.exists(MODELS_DIR):
        return []
    return [f.replace(".pkl", "") for f in os.listdir(MODELS_DIR) if f.endswith(".pkl")]

@app.get("/")
def home_page(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.get("/train")
def train_page(request: Request):
    return templates.TemplateResponse("train.html", {"request": request})

@app.post("/train-stream")
async def train_model_stream(
    request: Request,
    db: Session = Depends(get_db)
):
    form = await request.form(max_files=10000, max_fields=10000)
    model_name = form.get("model_name")
    smile_images = form.getlist("smile_images")
    not_smile_images = form.getlist("not_smile_images")

    if not model_name:
        async def err_gen():
            yield f"data: {json.dumps({'error': 'Model name is required.'})}\n\n"
        return StreamingResponse(err_gen(), media_type="text/event-stream")

    smile_bytes = []
    for file in smile_images:
        if hasattr(file, "filename") and file.filename:
            raw_content = await file.read()
            if raw_content:
                clean_jpg = sanitize_and_compress_image(raw_content)
                if clean_jpg:
                    smile_bytes.append(clean_jpg)

    not_smile_bytes = []
    for file in not_smile_images:
        if hasattr(file, "filename") and file.filename:
            raw_content = await file.read()
            if raw_content:
                clean_jpg = sanitize_and_compress_image(raw_content)
                if clean_jpg:
                    not_smile_bytes.append(clean_jpg)

    if not smile_bytes or not not_smile_bytes:
        async def err_gen():
            yield f"data: {json.dumps({'error': 'Please upload at least one valid image for both classes.'})}\n\n"
        return StreamingResponse(err_gen(), media_type="text/event-stream")

    async def event_generator():
        yield f"data: {json.dumps({'progress': 10, 'status': 'Images extracted. Training pipeline started...'})}\n\n"
        await asyncio.sleep(0.1)

        latest = {"pct": 10, "msg": "Training Random Forest classifier..."}

        def cb(pct, msg):
            latest["pct"] = pct
            latest["msg"] = msg

        def run_training():
            return ml_pipeline.train_new_model_stream(
                smile_bytes, not_smile_bytes, model_name, MODELS_DIR, progress_callback=cb
            )

        task = asyncio.create_task(asyncio.to_thread(run_training))

        while not task.done():
            yield f"data: {json.dumps({'progress': latest['pct'], 'status': latest['msg']})}\n\n"
            await asyncio.sleep(0.1)

        acc_score = await task

        log = models.TrainingLog(
            model_name=f"{model_name}.pkl",
            smile_count=len(smile_bytes),
            not_smile_count=len(not_smile_bytes),
            accuracy_score=acc_score
        )
        db.add(log)
        db.commit()

        yield f"data: {json.dumps({'progress': 100, 'status': f'Success! Accuracy: {acc_score*100:.1f}%'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/classify")
def classify_page(request: Request):
    available_models = get_available_models()
    return templates.TemplateResponse("classify.html", {
        "request": request, 
        "models": available_models
    })

@app.post("/classify")
async def process_classification(
    request: Request,
    model_choice: str = Form(...),
    image_file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    try:
        if not image_file or not image_file.filename:
            return templates.TemplateResponse("classify.html", {
                "request": request,
                "models": get_available_models(),
                "error": "Please select a valid image file."
            })

        raw_contents = await image_file.read()
        clean_jpg = sanitize_and_compress_image(raw_contents)
        if not clean_jpg:
            return templates.TemplateResponse("classify.html", {
                "request": request,
                "models": get_available_models(),
                "error": "Uploaded file is not a valid or supported image."
            })

        filename = f"{uuid.uuid4().hex}.jpg"
        file_path = os.path.join(UPLOADS_DIR, filename)
        with open(file_path, "wb") as f:
            f.write(clean_jpg)

        clean_model_name = model_choice.replace(".pkl", "").strip()
        model_path = os.path.join(MODELS_DIR, f"{clean_model_name}.pkl")

        if not os.path.exists(model_path):
            return templates.TemplateResponse("classify.html", {
                "request": request,
                "models": get_available_models(),
                "error": f"Model '{clean_model_name}' was not found on server."
            })

        predicted_class, confidence = ml_pipeline.predict_smile(clean_jpg, model_path)

        relative_image_path = f"/static/uploads/{filename}"

        inf_log = models.InferenceLog(
            image_path=relative_image_path,
            predicted_class=predicted_class,
            confidence=confidence,
            model_used=f"{clean_model_name}.pkl"
        )
        db.add(inf_log)
        db.commit()

        return templates.TemplateResponse("result.html", {
            "request": request,
            "image_path": relative_image_path,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "model_used": clean_model_name
        })
    except Exception as e:
        return templates.TemplateResponse("classify.html", {
            "request": request,
            "models": get_available_models(),
            "error": f"Error running classification: {str(e)}"
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