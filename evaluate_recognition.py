import cv2
import numpy as np
import os
from insightface.app import FaceAnalysis
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             accuracy_score, confusion_matrix,
                             classification_report)
import pandas as pd

# ── CONFIG ──────────────────────────────────────────────────────────────
DATASET_PATH = r"C:\Users\Ian\Desktop\RPI\Faces4Arclight"
DATABASE     = r"C:\Users\Ian\Desktop\RPI\face_database.npy"
THRESHOLD    = 0.5   # cosine similarity threshold
TEST_SPLIT   = 1.0   # use 30% of images for testing, 70% were used for enrollment
# ────────────────────────────────────────────────────────────────────────

print("Loading ArcFace...")
app = FaceAnalysis(allowed_modules=['detection', 'recognition'])
app.prepare(ctx_id=-1)

print("Loading face database...")
database = np.load(DATABASE, allow_pickle=True).item()
print(f"  {len(database)} people loaded: {list(database.keys())}\n")

def recognize(embedding):
    best_name, best_score = "Unknown", -1
    for name, known_emb in database.items():
        score = np.dot(embedding, known_emb) / (
            np.linalg.norm(embedding) * np.linalg.norm(known_emb))
        if score > best_score:
            best_score = score
            best_name  = name
    return (best_name if best_score >= THRESHOLD else "Unknown"), best_score

y_true, y_pred = [], []
results = []

print("Evaluating...\n")
for person_folder in sorted(os.listdir(DATASET_PATH)):
    person_path = os.path.join(DATASET_PATH, person_folder)
    if not os.path.isdir(person_path):
        continue

    images = [f for f in os.listdir(person_path)
              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # Use last 30% as test set
    test_start = int(len(images) * (1 - TEST_SPLIT))
    test_images = images[test_start:]

    if not test_images:
        test_images = images  # fallback if too few images

    correct = 0
    for img_file in test_images:
        img = cv2.imread(os.path.join(person_path, img_file))
        if img is None:
            continue

        faces = app.get(img)
        if faces:
            name, score = recognize(faces[0].embedding)
        else:
            name, score = "Unknown", 0.0

        y_true.append(person_folder)
        y_pred.append(name)
        if name == person_folder:
            correct += 1

    acc = correct / len(test_images) * 100 if test_images else 0
    print(f"  {person_folder}: {correct}/{len(test_images)} correct ({acc:.1f}%)")
    results.append({
        "Person"   : person_folder,
        "Correct"  : correct,
        "Total"    : len(test_images),
        "Accuracy" : f"{acc:.1f}%"
    })

# ── Overall Metrics ──────────────────────────────────────────────────────
labels = sorted(database.keys())
p  = precision_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0)
r  = recall_score   (y_true, y_pred, labels=labels, average='weighted', zero_division=0)
f1 = f1_score       (y_true, y_pred, labels=labels, average='weighted', zero_division=0)
ac = accuracy_score (y_true, y_pred)

print(f"""
─────────────── Recognition Metrics ───────────────
  Total Test Images: {len(y_true)}
  Threshold        : {THRESHOLD}

  Accuracy         : {ac:.4f}  ({ac*100:.2f}%)
  Precision        : {p:.4f}  ({p*100:.2f}%)
  Recall           : {r:.4f}  ({r*100:.2f}%)
  F1 Score         : {f1:.4f}  ({f1*100:.2f}%)
────────────────────────────────────────────────────
""")

# ── Per person report ────────────────────────────────────────────────────
print("Per-Person Classification Report:")
print(classification_report(y_true, y_pred, labels=labels, zero_division=0))

# ── Export ───────────────────────────────────────────────────────────────
df_results = pd.DataFrame(results)
df_overall = pd.DataFrame([{
    "Threshold" : THRESHOLD,
    "Accuracy"  : f"{ac*100:.2f}%",
    "Precision" : f"{p*100:.2f}%",
    "Recall"    : f"{r*100:.2f}%",
    "F1 Score"  : f"{f1*100:.2f}%",
    "Total Images": len(y_true)
}])

with pd.ExcelWriter("evaluation_recognition.xlsx") as writer:
    df_overall.to_excel(writer, sheet_name="Overall Metrics", index=False)
    df_results.to_excel(writer, sheet_name="Per Person",      index=False)

print("Saved → evaluation_recognition.xlsx")