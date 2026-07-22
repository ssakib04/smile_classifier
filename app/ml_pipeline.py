import io
import pickle
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def extract_features(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert('L')
    img = img.resize((64, 64))
    return np.array(img).flatten()

def train_new_model_stream(smile_bytes_list, not_smile_bytes_list, model_name, save_dir, progress_callback=None):
    total_images = len(smile_bytes_list) + len(not_smile_bytes_list)
    processed = 0

    X, y = [], []

    for b in smile_bytes_list:
        feat = extract_features(b)
        X.append(feat)
        y.append(1)
        processed += 1
        if progress_callback and total_images > 0:
            pct = int((processed / total_images) * 40)
            progress_callback(pct, f"Extracting features from smiling images ({processed}/{total_images})...")

    for b in not_smile_bytes_list:
        feat = extract_features(b)
        X.append(feat)
        y.append(0)
        processed += 1
        if progress_callback and total_images > 0:
            pct = int((processed / total_images) * 40)
            progress_callback(pct, f"Extracting features from non-smiling images ({processed}/{total_images})...")

    X = np.array(X)
    y = np.array(y)

    if progress_callback:
        progress_callback(50, "Splitting dataset into train/test sets...")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    if progress_callback:
        progress_callback(70, "Fitting RandomForestClassifier model...")

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    if progress_callback:
        progress_callback(90, "Evaluating model accuracy...")

    acc = float(clf.score(X_test, y_test))

    model_path = f"{save_dir}/{model_name}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)

    if progress_callback:
        progress_callback(100, "Training complete!")

    return acc

def predict_smile(image_bytes: bytes, model_path: str):
    with open(model_path, "rb") as f:
        clf = pickle.load(f)

    feat = extract_features(image_bytes).reshape(1, -1)
    pred = clf.predict(feat)[0]
    probs = clf.predict_proba(feat)[0]

    label = "Smile" if pred == 1 else "Not Smile"
    confidence = round(float(probs[pred]) * 100, 2)

    return label, confidence