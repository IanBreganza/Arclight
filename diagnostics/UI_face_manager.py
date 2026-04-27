import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from PIL import Image, ImageTk
import threading
import os
from insightface.app import FaceAnalysis

# ── CONFIG ──────────────────────────────────────────────────────────────
DATABASE  = r'C:\Users\Ian\Desktop\RPI\face_database.npy'
DATASET   = r'C:\Users\Ian\Desktop\RPI\Faces4Arclight'
THRESHOLD = 0.5
ENROLL_FRAMES = 30   # number of frames to capture per enrollment
# ────────────────────────────────────────────────────────────────────────

# ── Load ArcFace ─────────────────────────────────────────────────────────
print("Loading ArcFace...")
arc = FaceAnalysis(allowed_modules=['detection', 'recognition'])
arc.prepare(ctx_id=-1)

# ── Load Database ─────────────────────────────────────────────────────────
def load_database():
    if os.path.exists(DATABASE):
        return np.load(DATABASE, allow_pickle=True).item()
    return {}

def save_database(db):
    np.save(DATABASE, db)

database = load_database()

# ══════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════
class FaceManagerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Manager — Arclight")
        self.root.geometry("1000x650")
        self.root.configure(bg="#0f0f0f")
        self.root.resizable(False, False)

        self.camera_active = False
        self.cap = None
        self.camera_thread = None
        self.captured_embeddings = []
        self.current_mode = None   # 'add' or 'update'
        self.update_target = None  # name being updated

        self._build_ui()
        self._refresh_list()

    # ── UI BUILDER ────────────────────────────────────────────────────────
    def _build_ui(self):
        # ── Sidebar ──
        sidebar = tk.Frame(self.root, bg="#1a1a1a", width=260)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        tk.Label(sidebar, text="Arclight", font=("Courier", 14, "bold"),
                 bg="#1a1a1a", fg="#00ff88").pack(pady=(30, 4))
        tk.Label(sidebar, text="Face Management System", font=("Courier", 8),
                 bg="#1a1a1a", fg="#555").pack()

        tk.Frame(sidebar, bg="#333", height=1).pack(fill="x", padx=20, pady=20)

        # Enrolled count
        self.count_label = tk.Label(sidebar, text="", font=("Courier", 10),
                                    bg="#1a1a1a", fg="#888")
        self.count_label.pack(pady=(0, 16))

        # Action buttons
        btn_cfg = {"font": ("Courier", 10, "bold"), "bd": 0, "pady": 10,
                   "cursor": "hand2", "width": 22}

        tk.Button(sidebar, text="＋  ADD FACE", bg="#00ff88", fg="#0f0f0f",
                  command=self._open_add, **btn_cfg).pack(pady=4)

        tk.Button(sidebar, text="✎  UPDATE FACE", bg="#1a1a1a", fg="#00ff88",
                  relief="solid", bd=1, command=self._open_update,
                  font=("Courier", 10, "bold"), pady=10,
                  cursor="hand2", width=22).pack(pady=4)

        tk.Button(sidebar, text="✕  DELETE FACE", bg="#1a1a1a", fg="#ff4444",
                  relief="solid", bd=1, command=self._delete_face,
                  font=("Courier", 10, "bold"), pady=10,
                  cursor="hand2", width=22).pack(pady=4)

        tk.Frame(sidebar, bg="#333", height=1).pack(fill="x", padx=20, pady=20)

        tk.Button(sidebar, text="↺  REFRESH LIST", bg="#1a1a1a", fg="#888",
                  relief="solid", bd=1, command=self._refresh_list,
                  font=("Courier", 9), pady=8,
                  cursor="hand2", width=22).pack(pady=2)

        # Status
        self.status_label = tk.Label(sidebar, text="Ready", font=("Courier", 8),
                                     bg="#1a1a1a", fg="#555", wraplength=220)
        self.status_label.pack(side="bottom", pady=20)

        # ── Main Panel ──
        main = tk.Frame(self.root, bg="#0f0f0f")
        main.pack(side="left", fill="both", expand=True)

        # Header
        header = tk.Frame(main, bg="#0f0f0f")
        header.pack(fill="x", padx=30, pady=(24, 0))
        tk.Label(header, text="Enrolled Faces", font=("Courier", 16, "bold"),
                 bg="#0f0f0f", fg="#fff").pack(side="left")

        # List
        list_frame = tk.Frame(main, bg="#0f0f0f")
        list_frame.pack(fill="both", expand=True, padx=30, pady=16)

        cols = ("Name", "Embedding", "Status")
        self.tree = ttk.Treeview(list_frame, columns=cols, show="headings",
                                 selectmode="browse", height=20)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Treeview", background="#1a1a1a", foreground="#ccc",
                        fieldbackground="#1a1a1a", font=("Courier", 10),
                        rowheight=36, borderwidth=0)
        style.configure("Treeview.Heading", background="#252525",
                        foreground="#00ff88", font=("Courier", 10, "bold"),
                        relief="flat")
        style.map("Treeview", background=[("selected", "#00ff8833")],
                  foreground=[("selected", "#00ff88")])

        self.tree.heading("Name",      text="NAME")
        self.tree.heading("Embedding", text="EMBEDDING DIMS")
        self.tree.heading("Status",    text="STATUS")
        self.tree.column("Name",      width=200)
        self.tree.column("Embedding", width=160)
        self.tree.column("Status",    width=120)

        scrollbar = ttk.Scrollbar(list_frame, orient="vertical",
                                  command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)

        self.tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # ── Camera Window (hidden by default) ──
        self.camera_frame = tk.Frame(self.root, bg="#0a0a0a",
                                     width=400, height=400)
        self.camera_label = tk.Label(self.camera_frame, bg="#0a0a0a")
        self.camera_label.pack(fill="both", expand=True)

        self.cam_info = tk.Label(self.camera_frame, text="",
                                 font=("Courier", 10), bg="#0a0a0a", fg="#00ff88")
        self.cam_info.pack(pady=4)

        self.progress_var = tk.IntVar()
        self.progress = ttk.Progressbar(self.camera_frame,
                                        variable=self.progress_var,
                                        maximum=ENROLL_FRAMES, length=340)
        self.progress.pack(pady=4)

        cam_btns = tk.Frame(self.camera_frame, bg="#0a0a0a")
        cam_btns.pack(pady=8)

        self.start_btn = tk.Button(cam_btns, text="⬤  START CAPTURE",
                                   font=("Courier", 10, "bold"),
                                   bg="#00ff88", fg="#0f0f0f",
                                   bd=0, pady=8, padx=16,
                                   cursor="hand2",
                                   command=self._start_capture)
        self.start_btn.pack(side="left", padx=4)

        tk.Button(cam_btns, text="✕  CANCEL",
                  font=("Courier", 10, "bold"),
                  bg="#1a1a1a", fg="#ff4444",
                  relief="solid", bd=1, pady=8, padx=16,
                  cursor="hand2",
                  command=self._close_camera).pack(side="left", padx=4)

    # ── REFRESH LIST ──────────────────────────────────────────────────────
    def _refresh_list(self):
        global database
        database = load_database()
        for row in self.tree.get_children():
            self.tree.delete(row)

        for name, emb in sorted(database.items()):
            dims = f"{emb.shape[0]}D vector"
            self.tree.insert("", "end", values=(name, dims, "✓ Enrolled"))

        self.count_label.config(
            text=f"{len(database)} people enrolled")
        self._set_status("Database refreshed.")

    # ── ADD FACE ──────────────────────────────────────────────────────────
    def _open_add(self):
        name = simpledialog.askstring("Add Face", "Enter person's name:",
                                      parent=self.root)
        if not name:
            return
        name = name.strip()
        if name in database:
            messagebox.showerror("Error", f"'{name}' already exists. Use Update instead.")
            return

        self.current_mode = 'add'
        self.update_target = name
        self._open_camera()

    # ── UPDATE FACE ───────────────────────────────────────────────────────
    def _open_update(self):
        selected = self.tree.selection()
        if not selected:
            messagebox.showwarning("Select a face",
                                   "Please select a person from the list first.")
            return
        name = self.tree.item(selected[0])["values"][0]
        confirm = messagebox.askyesno("Update Face",
                                      f"Rescan face for '{name}'?")
        if not confirm:
            return

        self.current_mode = 'update'
        self.update_target = name
        self._open_camera()

    # ── DELETE FACE ───────────────────────────────────────────────────────
    def _delete_face(self):
        selected = self.tree.selection()
        if not selected:
            messagebox.showwarning("Select a face",
                                   "Please select a person from the list first.")
            return
        name = self.tree.item(selected[0])["values"][0]
        confirm = messagebox.askyesno("Delete Face",
                                      f"Delete '{name}' from the database?")
        if not confirm:
            return

        global database
        del database[name]
        save_database(database)

        # Also delete folder if exists
        folder = os.path.join(DATASET, name)
        if os.path.exists(folder):
            import shutil
            shutil.rmtree(folder)

        self._refresh_list()
        self._set_status(f"'{name}' deleted successfully.")
        messagebox.showinfo("Deleted", f"'{name}' has been removed.")

    # ── CAMERA ────────────────────────────────────────────────────────────
    def _open_camera(self):
        self.captured_embeddings = []
        self.progress_var.set(0)
        self.cam_info.config(
            text=f"Preparing to enroll: {self.update_target}\nClick START CAPTURE when ready.")
        self.start_btn.config(state="normal", text="⬤  START CAPTURE")
        self.camera_frame.place(relx=0.5, rely=0.5, anchor="center",
                                width=420, height=440)
        self.camera_frame.lift()
        self.camera_active = True
        self.cap = cv2.VideoCapture(0)
        self._update_camera_feed()

    def _update_camera_feed(self):
        if not self.camera_active:
            return
        ret, frame = self.cap.read()
        if ret:
            # Detect faces and draw box
            faces = arc.get(frame)
            display = frame.copy()
            for face in faces:
                bbox = face.bbox.astype(int)
                cv2.rectangle(display,
                              (bbox[0], bbox[1]),
                              (bbox[2], bbox[3]),
                              (0, 255, 136), 2)

            # Convert to tkinter image
            img = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = img.resize((400, 300))
            imgtk = ImageTk.PhotoImage(image=img)
            self.camera_label.imgtk = imgtk
            self.camera_label.config(image=imgtk)

        self.root.after(30, self._update_camera_feed)

    def _start_capture(self):
        self.start_btn.config(state="disabled", text="Capturing...")
        self.captured_embeddings = []
        self.cam_info.config(text=f"Capturing {ENROLL_FRAMES} frames for {self.update_target}...")
        threading.Thread(target=self._capture_loop, daemon=True).start()

    def _capture_loop(self):
        count = 0
        while count < ENROLL_FRAMES:
            if not self.camera_active:
                break
            ret, frame = self.cap.read()
            if not ret:
                continue
            faces = arc.get(frame)
            if faces:
                self.captured_embeddings.append(faces[0].embedding)
                count += 1
                self.progress_var.set(count)
                self.cam_info.config(
                    text=f"Captured {count}/{ENROLL_FRAMES} frames...")

        if len(self.captured_embeddings) >= 10:
            self._save_enrollment()
        else:
            self.cam_info.config(text="Not enough faces captured. Try again.")
            self.start_btn.config(state="normal", text="⬤  START CAPTURE")

    def _save_enrollment(self):
        global database
        avg_embedding = np.mean(self.captured_embeddings, axis=0)
        database[self.update_target] = avg_embedding
        save_database(database)

        # Save captured images to dataset folder
        folder = os.path.join(DATASET, self.update_target)
        os.makedirs(folder, exist_ok=True)

        self._close_camera()
        self._refresh_list()

        action = "updated" if self.current_mode == 'update' else "added"
        self._set_status(
            f"'{self.update_target}' {action} with "
            f"{len(self.captured_embeddings)} frames.")
        messagebox.showinfo("Success",
                            f"'{self.update_target}' has been {action} successfully!")

    def _close_camera(self):
        self.camera_active = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.camera_frame.place_forget()
        self.camera_label.config(image="")

    def _set_status(self, msg):
        self.status_label.config(text=msg)

    def on_close(self):
        self._close_camera()
        self.root.destroy()

# ── MAIN ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    app = FaceManagerApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
