import cv2
import torch
import numpy as np
import sqlite3
import threading
import tkinter as tk
from tkinter import ttk, simpledialog, messagebox
from PIL import Image, ImageTk
from datetime import datetime
from insightface.app import FaceAnalysis
import sys
import os

sys.path.insert(0, r'C:\Users\Ian\Desktop\RPI')
from models.experimental import attempt_load
from utils.general import non_max_suppression_face
from utils.datasets import letterbox

# ── CONFIG ───────────────────────────────────────────────────────────────
WEIGHTS      = r'C:\Users\Ian\Desktop\RPI\weights\yolov5n-face.pt'
DATABASE     = r'C:\Users\Ian\Desktop\RPI\face_database.npy'
DB_FILE      = r'C:\Users\Ian\Desktop\RPI\attendance.db'
CONFIDENCE   = 0.5
COOLDOWN_SEC = 60
# ─────────────────────────────────────────────────────────────────────────

# ── INIT DB ───────────────────────────────────────────────────────────────
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

def log_entry(name, confidence, last_seen, log_callback):
    now = datetime.now()
    if name == "Unknown":
        return last_seen

    if name not in last_seen or (now - last_seen[name]['time']).seconds > COOLDOWN_SEC:
        # New entry — log it
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute(
            'INSERT INTO logs (name, timestamp, date, confidence) VALUES (?, ?, ?, ?)',
            (name, now.strftime('%Y-%m-%d %H:%M:%S'),
             now.strftime('%Y-%m-%d'), float(confidence))
        )
        conn.commit()
        last_seen[name] = {'time': now, 'confidence': confidence, 'rowid': c.lastrowid}
        conn.close()
        log_callback(name, confidence, now)

    elif confidence > last_seen[name]['confidence']:
        # Higher confidence found during cooldown — update existing row
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute(
            'UPDATE logs SET confidence = ?, timestamp = ? WHERE id = ?',
            (float(confidence), now.strftime('%Y-%m-%d %H:%M:%S'),
             last_seen[name]['rowid'])
        )
        conn.commit()
        conn.close()
        last_seen[name]['confidence'] = confidence
        log_callback(name, confidence, now, update=True)

    return last_seen

# ── LOAD MODELS ───────────────────────────────────────────────────────────
print("Loading models...")
device = torch.device('cpu')
yolo   = attempt_load(WEIGHTS, map_location=device)
yolo.eval()

arc = FaceAnalysis(allowed_modules=['detection', 'recognition'])
arc.prepare(ctx_id=-1)

face_db = np.load(DATABASE, allow_pickle=True).item()
print(f"  {len(face_db)} people loaded.")

def recognize(embedding):
    best_name, best_score = "Unknown", -1
    for name, known_emb in face_db.items():
        score = np.dot(embedding, known_emb) / (
            np.linalg.norm(embedding) * np.linalg.norm(known_emb))
        if score > best_score:
            best_score = score
            best_name  = name
    return (best_name if best_score >= CONFIDENCE else "Unknown"), best_score

# ══════════════════════════════════════════════════════════════════════════
class RecognizeApp:
    def __init__(self, root):
        self.root       = root
        self.root.title("Arclight — Face Recognition")
        self.root.geometry("1100x660")
        self.root.configure(bg="#0d0d0d")
        self.root.resizable(False, False)

        self.running    = False
        self.cap        = None
        self.last_seen  = {}
        self.log_rows   = []   # in-memory log for export

        init_db()
        self._build_ui()

    # ── BUILD UI ──────────────────────────────────────────────────────────
    def _build_ui(self):
        # ── Left: camera + controls ──
        left = tk.Frame(self.root, bg="#0d0d0d")
        left.pack(side="left", fill="both", expand=True, padx=(16, 8), pady=16)

        # Title bar
        title_row = tk.Frame(left, bg="#0d0d0d")
        title_row.pack(fill="x", pady=(0, 10))
        tk.Label(title_row, text="ARCLIGHT", font=("Courier", 16, "bold"),
                 bg="#0d0d0d", fg="#00ff88").pack(side="left")
        tk.Label(title_row, text="  Face Recognition", font=("Courier", 10),
                 bg="#0d0d0d", fg="#555").pack(side="left", pady=4)

        # Camera canvas
        self.canvas = tk.Canvas(left, width=640, height=480, bg="#111",
                                highlightthickness=0)
        self.canvas.pack()
        self.canvas_image = self.canvas.create_image(0, 0, anchor="nw")

        # Status bar
        self.status = tk.Label(left, text="● STOPPED", font=("Courier", 9),
                               bg="#0d0d0d", fg="#ff4444")
        self.status.pack(anchor="w", pady=(6, 0))

        # Control buttons
        btn_row = tk.Frame(left, bg="#0d0d0d")
        btn_row.pack(fill="x", pady=10)

        self.start_btn = tk.Button(
            btn_row, text="▶  START", font=("Courier", 10, "bold"),
            bg="#00ff88", fg="#0d0d0d", bd=0, pady=8, padx=20,
            cursor="hand2", command=self._start)
        self.start_btn.pack(side="left", padx=(0, 8))

        self.stop_btn = tk.Button(
            btn_row, text="■  STOP", font=("Courier", 10, "bold"),
            bg="#1a1a1a", fg="#ff4444", relief="solid", bd=1,
            pady=8, padx=20, cursor="hand2",
            command=self._stop, state="disabled")
        self.stop_btn.pack(side="left", padx=(0, 8))

        tk.Button(
            btn_row, text="↓  SAVE LOGS", font=("Courier", 10, "bold"),
            bg="#1a1a1a", fg="#00aaff", relief="solid", bd=1,
            pady=8, padx=20, cursor="hand2",
            command=self._save_logs).pack(side="left")

        # ── Right: logs panel ──
        right = tk.Frame(self.root, bg="#111", width=340)
        right.pack(side="right", fill="y", padx=(8, 16), pady=16)
        right.pack_propagate(False)

        # Log header
        log_header = tk.Frame(right, bg="#111")
        log_header.pack(fill="x", padx=12, pady=(12, 6))

        tk.Label(log_header, text="ATTENDANCE LOG",
                 font=("Courier", 11, "bold"),
                 bg="#111", fg="#00ff88").pack(side="left")

        self.log_count = tk.Label(log_header, text="0 entries",
                                  font=("Courier", 8),
                                  bg="#111", fg="#555")
        self.log_count.pack(side="right")

        tk.Frame(right, bg="#222", height=1).pack(fill="x", padx=12)

        # Log list
        list_frame = tk.Frame(right, bg="#111")
        list_frame.pack(fill="both", expand=True, padx=4, pady=4)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Log.Treeview",
                        background="#111", foreground="#ccc",
                        fieldbackground="#111", font=("Courier", 9),
                        rowheight=32, borderwidth=0)
        style.configure("Log.Treeview.Heading",
                        background="#1a1a1a", foreground="#00ff88",
                        font=("Courier", 9, "bold"), relief="flat")
        style.map("Log.Treeview",
                  background=[("selected", "#00ff8822")],
                  foreground=[("selected", "#00ff88")])

        cols = ("Time", "Name", "Conf")
        self.log_tree = ttk.Treeview(list_frame, columns=cols,
                                     show="headings", style="Log.Treeview")
        self.log_tree.heading("Time", text="TIME")
        self.log_tree.heading("Name", text="NAME")
        self.log_tree.heading("Conf", text="CONF")
        self.log_tree.column("Time", width=80)
        self.log_tree.column("Name", width=130)
        self.log_tree.column("Conf", width=60)

        sb = ttk.Scrollbar(list_frame, orient="vertical",
                           command=self.log_tree.yview)
        self.log_tree.configure(yscroll=sb.set)
        self.log_tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        # Clear log button
        tk.Button(right, text="✕  CLEAR LOG",
                  font=("Courier", 9), bg="#111", fg="#555",
                  relief="flat", bd=0, cursor="hand2",
                  command=self._clear_log).pack(pady=(4, 12))

    # ── CAMERA LOOP ───────────────────────────────────────────────────────
    def _start(self):
        self.running = True
        self.cap = cv2.VideoCapture(0)
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.status.config(text="● RUNNING", fg="#00ff88")
        self._poll()

    def _stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.canvas.itemconfig(self.canvas_image, image="")
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.status.config(text="● STOPPED", fg="#ff4444")

    def _poll(self):
        if not self.running:
            return
        ret, frame = self.cap.read()
        if ret:
            faces = arc.get(frame)
            for face in faces:
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                name, score = recognize(face.embedding)
                self.last_seen = log_entry(
                    name, score, self.last_seen, self._add_log_row)
                color = (0, 255, 136) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{name}  {score:.2f}",
                            (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img).resize((640, 480))
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas.imgtk = imgtk
            self.canvas.itemconfig(self.canvas_image, image=imgtk)

        self.root.after(30, self._poll)

    def _loop(self):
        pass  # no longer used

    def _update_display(self):
        if not self.running:
            return
        frame = getattr(self, 'current_frame', None)
        if frame is not None:
            img   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img   = Image.fromarray(img).resize((640, 480))
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas.imgtk = imgtk
            self.canvas.config(image=imgtk)
        self.root.after(30, self._update_display)

    # ── LOG OPERATIONS ────────────────────────────────────────────────────
    def _add_log_row(self, name, confidence, timestamp, update=False):
        time_str = timestamp.strftime('%H:%M:%S')
        conf_str = f"{confidence:.2f}"

        if update:
            rowid = None
            # Find the matching rowid from last_seen
            for row in self.log_tree.get_children():
                vals = self.log_tree.item(row)["values"]
                if vals[1] == name:
                    self.log_tree.item(row, values=(time_str, name, conf_str))
                    break
            # Only update the LAST entry for this person in memory
            for entry in reversed(self.log_rows):
                if entry["name"] == name:
                    entry["confidence"] = confidence
                    entry["time"] = timestamp.strftime('%Y-%m-%d %H:%M:%S')
                    break
            return

        # New entry
        self.log_rows.append({
            "time": timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            "name": name,
            "confidence": confidence
        })
        self.log_tree.insert("", 0, values=(time_str, name, conf_str))
        self.log_count.config(text=f"{len(self.log_rows)} entries")

    def _clear_log(self):
        for row in self.log_tree.get_children():
            self.log_tree.delete(row)
        self.log_rows.clear()
        self.log_count.config(text="0 entries")

    def _save_logs(self):
        if not self.log_rows:
            # Load from DB if no in-memory logs
            conn = sqlite3.connect(DB_FILE)
            import pandas as pd
            df = pd.read_sql("SELECT * FROM logs ORDER BY timestamp DESC", conn)
            conn.close()
            if df.empty:
                messagebox.showinfo("No Logs", "No attendance logs to export.")
                return
        else:
            import pandas as pd
            df = pd.DataFrame(self.log_rows)

        # Ask for filename
        default_name = f"attendance_{datetime.now().strftime('%Y-%m-%d')}"
        filename = simpledialog.askstring(
            "Save Logs",
            "Enter filename (leave blank for default):",
            initialvalue=default_name,
            parent=self.root
        )
        if filename is None:
            return  # cancelled
        if not filename.strip():
            filename = default_name
        if not filename.endswith('.xlsx'):
            filename += '.xlsx'

        save_path = os.path.join(r'C:\Users\Ian\Desktop\RPI', filename)
        df.to_excel(save_path, index=False)
        messagebox.showinfo("Saved", f"Logs saved to:\n{save_path}")

    def on_close(self):
        self._stop()
        self.root.destroy()

# ── MAIN ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    app  = RecognizeApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()