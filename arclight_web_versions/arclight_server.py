import cv2
import numpy as np
import sqlite3
import os
import sys
import asyncio
import json
import io
import traceback
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Optional

# Directory where this script lives — HTML must be here too
BASE_DIR = Path(__file__).parent

import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from insightface.app import FaceAnalysis
import pandas as pd

sys.path.insert(0, r'C:\Users\Ian\Desktop\RPI')
from models.experimental import attempt_load
from utils.general import non_max_suppression_face
from utils.datasets import letterbox

# ── CONFIG ───────────────────────────────────────────────────────────────
WEIGHTS      = r'C:\Users\Ian\Desktop\RPI\weights\yolov5n-face.pt'
DATABASE     = r'C:\Users\Ian\Desktop\RPI\face_database.npy'
DB_FILE      = r'C:\Users\Ian\Desktop\RPI\attendance.db'
DATASET      = r'C:\Users\Ian\Desktop\RPI\Faces4Arclight'
CONFIDENCE   = 0.5
COOLDOWN_SEC = 60
ENROLL_FRAMES = 30
# ─────────────────────────────────────────────────────────────────────────

# ── GLOBALS ───────────────────────────────────────────────────────────────
arc         = None
face_db     = {}
cap         = None
is_running  = False
last_seen   = {}
enroll_state = {
    "active": False,
    "name": None,
    "mode": None,        # 'add' or 'update'
    "embeddings": [],
    "count": 0,
    "done": False,
    "message": ""
}

# ── WebSocket connection manager ──────────────────────────────────────────
class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, data: dict):
        dead = []
        for ws in self.active:
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

ws_manager = ConnectionManager()

# ── STARTUP / SHUTDOWN ────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global arc, face_db
    print("Loading ArcFace...")
    arc = FaceAnalysis(allowed_modules=['detection', 'recognition'])
    arc.prepare(ctx_id=-1)
    face_db = load_database()
    print(f"  {len(face_db)} people loaded.")
    init_db()
    yield
    if cap:
        cap.release()

app = FastAPI(lifespan=lifespan)

# ── DATABASE HELPERS ──────────────────────────────────────────────────────
def load_database():
    if os.path.exists(DATABASE):
        return np.load(DATABASE, allow_pickle=True).item()
    return {}

def save_database(db):
    np.save(DATABASE, db)

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS logs (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        name       TEXT NOT NULL,
        timestamp  TEXT NOT NULL,
        date       TEXT NOT NULL,
        confidence REAL
    )''')
    conn.commit()
    conn.close()

# ── RECOGNITION ───────────────────────────────────────────────────────────
def recognize(embedding):
    best_name, best_score = "Unknown", -1.0
    for name, known_emb in face_db.items():
        score = float(np.dot(embedding, known_emb) / (
            np.linalg.norm(embedding) * np.linalg.norm(known_emb)))
        if score > best_score:
            best_score = score
            best_name  = name
    return (best_name if best_score >= CONFIDENCE else "Unknown"), best_score

def log_entry_sync(name, confidence):
    """Log attendance; returns dict describing what happened or None."""
    global last_seen
    if name == "Unknown":
        return None

    now = datetime.now()
    if name not in last_seen or (now - last_seen[name]['time']).seconds > COOLDOWN_SEC:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute(
            'INSERT INTO logs (name, timestamp, date, confidence) VALUES (?, ?, ?, ?)',
            (name, now.strftime('%Y-%m-%d %H:%M:%S'),
             now.strftime('%Y-%m-%d'), float(confidence))
        )
        conn.commit()
        rowid = c.lastrowid
        conn.close()
        last_seen[name] = {'time': now, 'confidence': float(confidence), 'rowid': rowid}
        return {"event": "new", "name": name,
                "confidence": round(float(confidence), 3),
                "time": now.strftime('%H:%M:%S')}

    elif float(confidence) > last_seen[name]['confidence']:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute(
            'UPDATE logs SET confidence = ?, timestamp = ? WHERE id = ?',
            (float(confidence), now.strftime('%Y-%m-%d %H:%M:%S'),
             last_seen[name]['rowid'])
        )
        conn.commit()
        conn.close()
        last_seen[name]['confidence'] = float(confidence)
        return {"event": "update", "name": name,
                "confidence": round(float(confidence), 3),
                "time": now.strftime('%H:%M:%S')}
    return None

# ── MJPEG STREAM ──────────────────────────────────────────────────────────
async def generate_frames():
    global cap, is_running, last_seen
    while is_running and cap and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            await asyncio.sleep(0.03)
            continue

        faces = arc.get(frame)
        for face in faces:
            bbox  = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            name, score = recognize(face.embedding)
            log_data = log_entry_sync(name, score)
            if log_data:
                asyncio.create_task(ws_manager.broadcast(log_data))

            color = (0, 255, 136) if name != "Unknown" else (0, 80, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{name}  {score:.2f}",
                        (x1, max(y1 - 8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # Also handle enrollment capture
        if enroll_state["active"] and not enroll_state["done"]:
            _enroll_tick(frame)

        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
               + buf.tobytes() + b'\r\n')
        await asyncio.sleep(0.03)

def _enroll_tick(frame):
    """Accumulate embeddings during enrollment (called from frame loop)."""
    if enroll_state["count"] >= ENROLL_FRAMES:
        return
    faces = arc.get(frame)
    if faces:
        enroll_state["embeddings"].append(faces[0].embedding)
        enroll_state["count"] += 1
        enroll_state["message"] = f"Captured {enroll_state['count']}/{ENROLL_FRAMES}"
        if enroll_state["count"] >= ENROLL_FRAMES:
            _finish_enrollment()

def _finish_enrollment():
    global face_db
    embs = enroll_state["embeddings"]
    if len(embs) >= 10:
        avg = np.mean(embs, axis=0)
        face_db[enroll_state["name"]] = avg
        save_database(face_db)
        folder = os.path.join(DATASET, enroll_state["name"])
        os.makedirs(folder, exist_ok=True)
        enroll_state["done"] = True
        action = "updated" if enroll_state["mode"] == "update" else "added"
        enroll_state["message"] = f"✓ '{enroll_state['name']}' {action} successfully!"
    else:
        enroll_state["done"] = True
        enroll_state["message"] = "✗ Not enough faces detected. Try again."

# ══════════════════════════════════════════════════════════════════════════
# REST ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = BASE_DIR / "arclight_ui.html"
    if not html_path.exists():
        return HTMLResponse(
            f"<pre>ERROR: arclight_ui.html not found at {html_path}\n"
            "Put both files in the same folder.</pre>", status_code=500)
    return HTMLResponse(html_path.read_text(encoding="utf-8"))

# ── Video stream ──────────────────────────────────────────────────────────
@app.get("/video_feed")
async def video_feed():
    if not is_running:
        raise HTTPException(status_code=503, detail="Camera not running")
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

# ── Camera control ────────────────────────────────────────────────────────
@app.post("/api/camera/start")
async def start_camera():
    global cap, is_running, last_seen
    if is_running:
        return {"status": "already running"}
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise HTTPException(status_code=500, detail="Cannot open camera")
    is_running = True
    last_seen  = {}
    return {"status": "started"}

@app.post("/api/camera/stop")
async def stop_camera():
    global cap, is_running
    is_running = False
    if cap:
        cap.release()
        cap = None
    return {"status": "stopped"}

@app.get("/api/camera/status")
async def camera_status():
    return {"running": is_running}

# ── Faces CRUD ────────────────────────────────────────────────────────────
@app.get("/api/faces")
async def list_faces():
    return [
        {"name": name, "dims": int(emb.shape[0])}
        for name, emb in sorted(face_db.items())
    ]

class EnrollRequest(BaseModel):
    name: str
    mode: str  # 'add' or 'update'

@app.post("/api/faces/enroll/start")
async def enroll_start(req: EnrollRequest):
    global cap, is_running
    name = req.name.strip()
    if req.mode == "add" and name in face_db:
        raise HTTPException(400, f"'{name}' already exists. Use update mode.")
    if not is_running:
        cap = cv2.VideoCapture(0)
        is_running = True
    enroll_state.update({
        "active": True, "name": name, "mode": req.mode,
        "embeddings": [], "count": 0, "done": False,
        "message": f"Ready — position face for '{name}'"
    })
    return {"status": "started", "name": name}

@app.get("/api/faces/enroll/status")
async def enroll_status():
    return {
        "active":   enroll_state["active"],
        "count":    enroll_state["count"],
        "total":    ENROLL_FRAMES,
        "done":     enroll_state["done"],
        "message":  enroll_state["message"],
    }

@app.post("/api/faces/enroll/cancel")
async def enroll_cancel():
    enroll_state.update({"active": False, "done": False, "count": 0,
                          "embeddings": [], "message": ""})
    return {"status": "cancelled"}

@app.delete("/api/faces/{name}")
async def delete_face(name: str):
    global face_db
    if name not in face_db:
        raise HTTPException(404, f"'{name}' not found")
    del face_db[name]
    save_database(face_db)
    folder = os.path.join(DATASET, name)
    if os.path.exists(folder):
        import shutil
        shutil.rmtree(folder)
    return {"status": "deleted", "name": name}

# ── Logs ──────────────────────────────────────────────────────────────────
@app.get("/api/logs")
async def get_logs():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT id, name, timestamp, confidence FROM logs ORDER BY timestamp DESC LIMIT 200")
    rows = c.fetchall()
    conn.close()
    return [{"id": r[0], "name": r[1], "time": r[2], "confidence": round(r[3], 3)}
            for r in rows]

@app.get("/api/logs/export")
async def export_logs():
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql("SELECT * FROM logs ORDER BY timestamp DESC", conn)
    conn.close()
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    buf.seek(0)
    fname = f"attendance_{datetime.now().strftime('%Y-%m-%d')}.xlsx"
    return StreamingResponse(
        buf,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename={fname}"}
    )

@app.delete("/api/logs")
async def clear_logs():
    conn = sqlite3.connect(DB_FILE)
    conn.execute("DELETE FROM logs")
    conn.commit()
    conn.close()
    return {"status": "cleared"}

# ── WebSocket ─────────────────────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws_manager.connect(ws)
    try:
        while True:
            try:
                # Wait for ping with timeout; disconnect if client gone
                await asyncio.wait_for(ws.receive_text(), timeout=30)
            except asyncio.TimeoutError:
                # No ping received — send a keepalive pong back
                try:
                    await ws.send_json({"event": "ping"})
                except Exception:
                    break
    except WebSocketDisconnect:
        pass
    finally:
        ws_manager.disconnect(ws)

# ── MAIN ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("arclight_server:app", host="0.0.0.0", port=8000, reload=False)