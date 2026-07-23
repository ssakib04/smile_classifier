import os
import pickle
import io
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

IMAGE_SIZE = (64, 64)
TRAINING_EPOCHS = 10
TREES_PER_EPOCH = 10


def process_image(image_bytes: bytes):
    """Safely reads raw image bytes, converts RGBA/PNG to RGB, resizes, and flattens to 1D vector."""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img = img.convert("RGB")
        img = img.resize(IMAGE_SIZE)
        return np.array(img).flatten() / 255.0
    except Exception as e:
        print(f"Skipping corrupted image: {e}")
        return None


def _report(progress_callback, percent, message, epoch=None, total_epochs=None):
    if progress_callback:
        progress_callback(
            percent=min(100, max(0, percent)),
            message=message,
            epoch=epoch,
            total_epochs=total_epochs,
        )


def train_new_model(
    smile_files: list,
    not_smile_files: list,
    model_name: str,
    models_dir: str = "models_store",
    progress_callback=None,
):
    X, y = [], []
    total_images = len(smile_files) + len(not_smile_files)
    processed = 0

    for file_bytes in smile_files:
        vec = process_image(file_bytes)
        if vec is not None:
            X.append(vec)
            y.append(1)
        processed += 1
        pct = int((processed / max(total_images, 1)) * 25)
        _report(
            progress_callback,
            pct,
            f"Processing smile images ({processed}/{total_images})",
        )

    for file_bytes in not_smile_files:
        vec = process_image(file_bytes)
        if vec is not None:
            X.append(vec)
            y.append(0)
        processed += 1
        pct = 25 + int(((processed - len(smile_files)) / max(len(not_smile_files), 1)) * 15)
        _report(
            progress_callback,
            pct,
            f"Processing not-smile images ({processed - len(smile_files)}/{len(not_smile_files)})",
        )

    if len(X) == 0:
        raise ValueError("No valid image files could be processed.")

    X = np.array(X)
    y = np.array(y)

    _report(progress_callback, 40, "Initializing model training…")

    clf = RandomForestClassifier(
        n_estimators=TREES_PER_EPOCH,
        warm_start=True,
        random_state=42,
    )

    for epoch in range(1, TRAINING_EPOCHS + 1):
        clf.n_estimators = epoch * TREES_PER_EPOCH
        clf.fit(X, y)
        pct = 40 + int((epoch / TRAINING_EPOCHS) * 50)
        _report(
            progress_callback,
            pct,
            f"Epoch {epoch}/{TRAINING_EPOCHS} - {pct}%",
            epoch=epoch,
            total_epochs=TRAINING_EPOCHS,
        )

    _report(progress_callback, 92, "Evaluating model accuracy…")

    if len(np.unique(y)) > 1 and len(y) >= 4:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        clf_eval = RandomForestClassifier(n_estimators=100, random_state=42)
        clf_eval.fit(X_train, y_train)
        acc = float(accuracy_score(y_test, clf_eval.predict(X_test)))
    else:
        acc = 1.0

    _report(progress_callback, 96, "Saving model to disk…")

    os.makedirs(models_dir, exist_ok=True)
    model_file_path = os.path.join(models_dir, f"{model_name}.pkl")

    with open(model_file_path, "wb") as f:
        pickle.dump(clf, f)

    _report(progress_callback, 100, "Training complete")

    return acc


def predict_smile(image_bytes: bytes, model_path: str):
    """Loads specified .pkl model and performs prediction on an image."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found.")

    with open(model_path, "rb") as f:
        clf = pickle.load(f)

    features = process_image(image_bytes)
    if features is None:
        raise ValueError("Could not read uploaded image for inference.")

    features = features.reshape(1, -1)
    prediction = clf.predict(features)[0]
    probabilities = clf.predict_proba(features)[0]

    predicted_label = "Smiling" if prediction == 1 else "Not Smiling"
    confidence = float(max(probabilities)) * 100

    return predicted_label, round(confidence, 2)
