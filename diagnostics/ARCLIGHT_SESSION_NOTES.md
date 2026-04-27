# Arclight Face Recognition System — Session Notes
### April 2026 — UI Development & Evaluation

---

## Project Context
Face recognition attendance system using **YOLOv5-Face + ArcFace (InsightFace)** deployed locally on Windows PC, intended for Raspberry Pi 4 deployment. 20 enrolled people stored in `face_database.npy`.

**Project path:** `C:\Users\Ian\Desktop\RPI\`

---

## 1. Face Dataset & Enrollment

- New dataset added at `C:\Users\Ian\Desktop\RPI\Faces4Arclight\`
- 21 people enrolled: Allen, Angela, Brian, Bussleh, Carlo, Cristel, Erwin, Eubert, Eunice, Francis, Fritz, Jairus, Jamaica, Jasmine, Jenelle, Jong, JP, Justin, RB, Ryan, Sam
- **Francis** was added last-minute by re-running `enroll_faces.py`
- **Brian** has an empty folder — 0 images, skipped during enrollment
- Some images were `.heic` (Apple format) — ffmpeg botched the conversion
- Fixed using `pillow-heif` library with `convert_heic.py`

---

## 2. Recognition Metrics (ArcFace)

**Script:** `evaluate_recognition.py`
**Dataset:** `Faces4Arclight` (972 test images)
**Threshold:** 0.5

| Metric | Score |
|---|---|
| Accuracy | 94.65% |
| Precision | 99.60% |
| Recall | 94.65% |
| F1 Score | 96.75% |

**Low performers:**
- Jamaica — 61.1% (only 18 images)
- Erwin — 63.6%
- Fritz — 69.0%

---

## 3. Detection Metrics (YOLOv5n-Face on WIDERFace)

**Script:** `evaluate_detection_widerface.py`
**Dataset:** WIDERFace val (4,952 images) at `C:\Users\Ian\Desktop\yolov5-face-master\VOC\`
**Device:** CUDA (RTX 4060 Laptop)

| Threshold | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|
| 0.50 | 100% | 39.30% | 56.42% | 39.30% |
| 0.25 | 100% | 47.52% | 64.42% | 47.52% |
| 0.10 | 100% | 57.53% | 73.04% | 57.53% |
| 0.05 | 100% | 66.11% | 79.60% | 66.11% |
| 0.01 | 100% | 83.97% | 91.28% | 83.97% |

**Recommended threshold:** `0.01` — best F1 with zero false positives across all thresholds.

---

## 4. UI Development

### UI_face_manager.py — CRUD Interface
- Tkinter dark-theme GUI for managing enrolled faces
- **Add** — enter name → camera opens → captures 30 frames → saves embedding
- **Update** — select person → rescan via camera → replaces embedding
- **Delete** — select person → confirms → removes from database and dataset folder
- Camera uses `root.after()` polling (threading caused display issues)
- Fixed `AssertionError` by using `allowed_modules=['detection', 'recognition']`

### UI_face_recognize.py — Main Recognition Interface
- Live webcam feed on left panel (640x480)
- Attendance log panel on right (Time, Name, Confidence)
- ▶ START / ■ STOP / ↓ SAVE LOGS buttons
- **Camera display fix:** switched from `tk.Label` to `tk.Canvas` with `create_image()`
- Removed threading in favor of `root.after(30, self._poll)` for reliable display

---

## 5. Attendance Logging Logic

### Cooldown System
- `COOLDOWN_SEC = 60` — logs each person once per 60 seconds
- After cooldown resets, a new entry is created

### Confidence Update During Cooldown
- If a **higher confidence** is detected during the cooldown window, it **replaces** the existing entry in both SQLite and the log panel
- Uses `UPDATE logs SET confidence = ? WHERE id = ?` to update in-place
- `_add_log_row` uses `reversed(self.log_rows)` to update only the **most recent** entry for that person — prevents older entries from being overwritten

### Bug Fixed
- Both log entries were showing the same timestamp and confidence on export
- Caused by updating **all** entries with matching name instead of only the latest
- Fixed by iterating `reversed(self.log_rows)` and breaking after first match

---

## 6. Key File Locations

| File | Path |
|---|---|
| Main recognition UI | `C:\Users\Ian\Desktop\RPI\recognize_ui.py` |
| Face manager CRUD | `C:\Users\Ian\Desktop\RPI\face_manager.py` |
| Enrollment script | `C:\Users\Ian\Desktop\RPI\enroll_faces.py` |
| Recognition evaluation | `C:\Users\Ian\Desktop\RPI\evaluate_recognition.py` |
| Detection evaluation | `C:\Users\Ian\Desktop\yolov5-face-master\yolov5-face-master\evaluate_detection_widerface.py` |
| Face database | `C:\Users\Ian\Desktop\RPI\face_database.npy` |
| Attendance DB | `C:\Users\Ian\Desktop\RPI\attendance.db` |
| YOLOv5 weights | `C:\Users\Ian\Desktop\RPI\weights\yolov5n-face.pt` |

---

## 7. Important Config Values

```python
CONFIDENCE   = 0.5    # ArcFace cosine similarity threshold
COOLDOWN_SEC = 60     # seconds before re-logging same person
CONF_THRES   = 0.01   # YOLOv5 detection threshold (best result)
```

---

## 8. Known Issues / Notes

- **Brian** has no images — folder exists but enrollment skipped
- **sys.path.insert** required in recognize_ui.py to import YOLOv5 modules from RPI folder
- Windows path strings require `r''` prefix to avoid unicode escape errors
- `weights_only=False` required in `torch.load()` for PyTorch 2.6+ compatibility
- `np.int` → `int` fix required across codebase for NumPy compatibility
- ArcFace requires `allowed_modules=['detection', 'recognition']` — recognition alone causes AssertionError

---

*Session date: April 25–27, 2026*
