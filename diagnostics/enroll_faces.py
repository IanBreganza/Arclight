import insightface
import numpy as np
import cv2
import os
from insightface.app import FaceAnalysis

app = FaceAnalysis(allowed_modules=['recognition', 'detection'])
app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.2)

dataset_path = r"C:\Users\Ian\Desktop\RPI\Faces4Arclight"
database = {}

for person_folder in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person_folder)
    if not os.path.isdir(person_path):
        continue

    embeddings = []
    image_files = [f for f in os.listdir(person_path)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Enrolling {person_folder} ({len(image_files)} images)...")

    for img_file in image_files:
        img_path = os.path.join(person_path, img_file)
        img = cv2.imread(img_path)
        if img is None:
            continue
        faces = app.get(img)
        if faces:
            embeddings.append(faces[0].embedding)

    if embeddings:
        database[person_folder] = np.mean(embeddings, axis=0)
        print(f"  ✓ Enrolled using {len(embeddings)} valid faces")
    else:
        print(f"  ✗ No faces detected, skipping")

# Save — overwrites old database completely
np.save(r"C:\Users\Ian\Desktop\RPI\face_database.npy", database)
print(f"\nDone! {len(database)} people enrolled → face_database.npy")