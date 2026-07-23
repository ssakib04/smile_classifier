import os
import uuid
import io
import json
import asyncio
import threading
from typing import List, Optional, Callable

from PIL import Image, ImageOps

from fastapi import FastAPI, Request, Form, UploadFile, File, Depends, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import StreamingResponse, JSONResponse
from sqlalchemy.orm import Session

from app.database import engine, Base, get_db
from app import models, ml_pipeline

Base.metadata.create_all(bind=engine)

MODELS_DIR = "models_store"
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
UPLOADS_DIR = os.path.join(STATIC_DIR, "uploads")
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", 10 * 1024 * 1024))

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

app = FastAPI(title="Smile Classifier")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory="app/templates")

training_jobs: dict = {}
_jobs_lock = threading.Lock()


def _convert_to_rgb_with_white_bg(img: Image.Image) -> Image.Image:
    """Convert any image mode to RGB, compositing transparency onto white."""
    if img.mode in ("RGBA", "LA"):
        background = Image.new("RGB", img.size, (255, 255, 255))
        alpha = img.split()[-1]
        rgb = img.convert("RGB")
        background.paste(rgb, mask=alpha)
        return background
    if img.mode == "P":
        if "transparency" in img.info:
            img = img.convert("RGBA")
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img.convert("RGB"), mask=img.split()[-1])
            return background
        return img.convert("RGB")
    if img.mode != "RGB":
        return img.convert("RGB")
    return img


def sanitize_and_compress_image(file_bytes: bytes, max_dim: int = 1024) -> Optional[bytes]:
    """Normalize any supported image format to JPEG with white-background RGB."""
    try:
        img = Image.open(io.BytesIO(file_bytes))
        img = ImageOps.exif_transpose(img)
        img = _convert_to_rgb_with_white_bg(img)
        img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)

        output = io.BytesIO()
        img.save(output, format="JPEG", quality=85, optimize=True)
        return output.getvalue()
    except Exception:
        return None


def validate_upload_size(content: bytes, filename: str = "") -> None:
    if len(content) > MAX_UPLOAD_BYTES:
        detail = (
            f"File '{filename}' exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit."
            if filename
            else f"Upload exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit."
        )
        raise HTTPException(status_code=413, detail=detail)


def get_available_models():
    if not os.path.exists(MODELS_DIR):
        return []
    return [f.replace(".pkl", "") for f in os.listdir(MODELS_DIR) if f.endswith(".pkl")]


def _update_job(job_id: str, **fields):
    with _jobs_lock:
        if job_id in training_jobs:
            training_jobs[job_id].update(fields)


def _run_training_job(
    job_id: str,
    model_name: str,
    smile_bytes: List[bytes],
    not_smile_bytes: List[bytes],
    db_factory: Callable,
):
    def progress_callback(percent, message, epoch=None, total_epochs=None):
        _update_job(
            job_id,
            status="training",
            percent=percent,
            message=message,
            epoch=epoch,
            total_epochs=total_epochs,
        )

    try:
        _update_job(job_id, status="training", percent=5, message="Starting training…")
        acc_score = ml_pipeline.train_new_model(
            smile_bytes,
            not_smile_bytes,
            model_name,
            MODELS_DIR,
            progress_callback=progress_callback,
        )

        db = db_factory()
        try:
            log = models.TrainingLog(
                model_name=f"{model_name}.pkl",
                smile_count=len(smile_bytes),
                not_smile_count=len(not_smile_bytes),
                accuracy_score=acc_score,
            )
            db.add(log)
            db.commit()
        finally:
            db.close()

        _update_job(
            job_id,
            status="completed",
            percent=100,
            message=f"Model '{model_name}' trained successfully.",
            accuracy=acc_score,
            model_name=model_name,
        )
    except Exception as exc:
        _update_job(
            job_id,
            status="failed",
            message=str(exc),
            error=str(exc),
        )


@app.get("/")
def home_page(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})


@app.get("/train")
def train_page(request: Request):
    return templates.TemplateResponse("train.html", {"request": request})


@app.post("/train")
async def train_model_legacy(request: Request):
    """Legacy form POST — training now uses POST /api/train with SSE progress."""
    return templates.TemplateResponse(
        "train.html",
        {
            "request": request,
            "error": "Training must be started from this page using the Train button.",
        },
    )


@app.post("/api/train")
async def start_training(request: Request, db: Session = Depends(get_db)):
    form = await request.form(max_files=10000, max_fields=10000)

    model_name = (form.get("model_name") or "").strip()
    smile_images = form.getlist("smile_images")
    not_smile_images = form.getlist("not_smile_images")

    if not model_name:
        return JSONResponse(status_code=400, content={"detail": "Model name is required."})

    smile_bytes = []
    for file in smile_images:
        if hasattr(file, "filename") and file.filename:
            raw_content = await file.read()
            if not raw_content:
                continue
            try:
                validate_upload_size(raw_content, file.filename)
            except HTTPException as exc:
                return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
            clean_jpg = sanitize_and_compress_image(raw_content)
            if clean_jpg:
                smile_bytes.append(clean_jpg)

    not_smile_bytes = []
    for file in not_smile_images:
        if hasattr(file, "filename") and file.filename:
            raw_content = await file.read()
            if not raw_content:
                continue
            try:
                validate_upload_size(raw_content, file.filename)
            except HTTPException as exc:
                return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
            clean_jpg = sanitize_and_compress_image(raw_content)
            if clean_jpg:
                not_smile_bytes.append(clean_jpg)

    if not smile_bytes or not not_smile_bytes:
        return JSONResponse(
            status_code=400,
            content={"detail": "Please upload at least one valid image for both classes."},
        )

    job_id = uuid.uuid4().hex
    with _jobs_lock:
        training_jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "percent": 0,
            "message": "Queued for training…",
            "epoch": None,
            "total_epochs": ml_pipeline.TRAINING_EPOCHS,
        }

    from app.database import SessionLocal

    thread = threading.Thread(
        target=_run_training_job,
        args=(job_id, model_name, smile_bytes, not_smile_bytes, SessionLocal),
        daemon=True,
    )
    thread.start()

    return JSONResponse(content={"job_id": job_id})


@app.get("/api/train/progress/{job_id}")
async def training_progress_sse(job_id: str):
    async def event_generator():
        while True:
            with _jobs_lock:
                job = training_jobs.get(job_id)
                payload = dict(job) if job else {"status": "not_found", "message": "Job not found."}

            yield f"data: {json.dumps(payload)}\n\n"

            if payload.get("status") in ("completed", "failed", "not_found"):
                break
            await asyncio.sleep(0.4)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/classify")
def classify_page(request: Request):
    available_models = get_available_models()
    return templates.TemplateResponse(
        "classify.html",
        {"request": request, "models": available_models},
    )


@app.post("/classify")
async def process_classification(
    request: Request,
    model_choice: str = Form(...),
    image_file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    if not image_file.filename:
        raise HTTPException(status_code=400, detail="No file selected.")

    raw_contents = await image_file.read()
    validate_upload_size(raw_contents, image_file.filename)

    clean_jpg = sanitize_and_compress_image(raw_contents)
    if not clean_jpg:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")

    filename = f"{uuid.uuid4().hex}.jpg"
    file_path = os.path.join(UPLOADS_DIR, filename)
    with open(file_path, "wb") as f:
        f.write(clean_jpg)

    model_path = os.path.join(MODELS_DIR, f"{model_choice}.pkl")
    predicted_class, confidence = ml_pipeline.predict_smile(clean_jpg, model_path)

    relative_image_path = f"/static/uploads/{filename}"

    inf_log = models.InferenceLog(
        image_path=relative_image_path,
        predicted_class=predicted_class,
        confidence=confidence,
        model_used=f"{model_choice}.pkl",
    )
    db.add(inf_log)
    db.commit()

    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "image_path": relative_image_path,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "model_used": model_choice,
        },
    )


@app.get("/history")
def history_page(request: Request, db: Session = Depends(get_db)):
    train_logs = db.query(models.TrainingLog).order_by(models.TrainingLog.created_at.desc()).all()
    inference_logs = db.query(models.InferenceLog).order_by(models.InferenceLog.created_at.desc()).all()

    return templates.TemplateResponse(
        "history.html",
        {
            "request": request,
            "train_logs": train_logs,
            "inference_logs": inference_logs,
        },
    )
