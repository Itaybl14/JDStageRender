#!/usr/bin/env python3
"""
JD Stage Render - small GUI around stage_render.py.

Pick a mashup map's mainsequence, choose quality, preview a frame, render.
The GUI runs stage_render.py as a subprocess (same pipeline as the CLI).
"""

import os
import re
import subprocess
import sys
import tempfile
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

HERE = os.path.dirname(os.path.abspath(__file__))
CLI = os.path.join(HERE, "stage_render.py")
DOWNLOADS = os.path.join(os.path.expanduser("~"), "Downloads")

RESOLUTIONS = ["426x240", "640x360", "1280x720", "1920x1080"]
QUALITIES = {  # label -> (crf, x264 preset)
    "Lowest (fastest, smallest)": (35, "ultrafast"),
    "Low": (28, "veryfast"),
    "Medium": (23, "fast"),
    "High": (18, "medium"),
}
MASHUP_COLORS = ["as authored", "blue", "orange", "green", "purple"]

PROGRESS_RE = re.compile(r"\[render\] (\d+)/(\d+) frames\s+([\d.]+) fps\s+eta\s+(\d+)s")
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def map_name_from(path):
    base = os.path.basename(path)
    i = base.lower().find("_mainsequence")
    return base[:i] if i > 0 else os.path.splitext(base)[0]


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("JD Stage Render")
        self.minsize(760, 640)
        self.proc = None
        self.preview_img = None
        self._build()

    # -- layout -----------------------------------------------------------------------
    def _build(self):
        pad = {"padx": 6, "pady": 3}
        root = ttk.Frame(self, padding=10)
        root.pack(fill="both", expand=True)
        root.columnconfigure(1, weight=1)

        self.v_seq = tk.StringVar()
        self.v_coach = tk.StringVar()
        self.v_audio = tk.StringVar()
        self.v_out = tk.StringVar()
        self.v_no_coach = tk.BooleanVar(value=False)
        self.v_no_audio = tk.BooleanVar(value=False)

        r = 0
        ttk.Label(root, text="Mainsequence").grid(row=r, column=0, sticky="w", **pad)
        ttk.Entry(root, textvariable=self.v_seq).grid(row=r, column=1, sticky="ew", **pad)
        ttk.Button(root, text="Browse…", command=self.pick_seq).grid(row=r, column=2, **pad)
        r += 1
        ttk.Label(root, text="Coach video").grid(row=r, column=0, sticky="w", **pad)
        ttk.Entry(root, textvariable=self.v_coach).grid(row=r, column=1, sticky="ew", **pad)
        f = ttk.Frame(root)
        f.grid(row=r, column=2, sticky="w")
        ttk.Button(f, text="Browse…", command=lambda: self.pick_file(self.v_coach, "Coach video",
                   [("Video", "*.webm *.mp4 *.mkv *.mov"), ("All", "*.*")])).pack(side="left")
        ttk.Checkbutton(f, text="none", variable=self.v_no_coach).pack(side="left", padx=4)
        r += 1
        ttk.Label(root, text="Audio").grid(row=r, column=0, sticky="w", **pad)
        ttk.Entry(root, textvariable=self.v_audio).grid(row=r, column=1, sticky="ew", **pad)
        f = ttk.Frame(root)
        f.grid(row=r, column=2, sticky="w")
        ttk.Button(f, text="Browse…", command=lambda: self.pick_file(self.v_audio, "Audio",
                   [("Audio", "*.ogg *.wav *.opus *.mp3 *.m4a *.flac"), ("All", "*.*")])).pack(side="left")
        ttk.Checkbutton(f, text="none", variable=self.v_no_audio).pack(side="left", padx=4)
        r += 1
        ttk.Label(root, text="(leave coach/audio empty to auto-detect them in the map folder)",
                  foreground="#666").grid(row=r, column=1, sticky="w", padx=6)
        r += 1
        ttk.Label(root, text="Output").grid(row=r, column=0, sticky="w", **pad)
        ttk.Entry(root, textvariable=self.v_out).grid(row=r, column=1, sticky="ew", **pad)
        ttk.Button(root, text="Save as…", command=self.pick_out).grid(row=r, column=2, **pad)
        r += 1

        opts = ttk.LabelFrame(root, text="Settings", padding=8)
        opts.grid(row=r, column=0, columnspan=3, sticky="ew", **pad)
        self.v_res = self._combo(opts, "Resolution", RESOLUTIONS, 0, 0, "640x360")
        self.v_fps = self._combo(opts, "FPS", ["25", "30", "50", "60"], 0, 2, "25", width=6)
        self.v_quality = self._combo(opts, "Quality", list(QUALITIES), 1, 0, "Low", width=24)
        self.v_workers = tk.StringVar(value="4")
        ttk.Label(opts, text="Workers").grid(row=1, column=2, sticky="w", padx=4)
        ttk.Spinbox(opts, from_=1, to=os.cpu_count() or 8, textvariable=self.v_workers, width=6).grid(
            row=1, column=3, sticky="w", padx=4)
        self.v_mcolor = self._combo(opts, "Mashup colour", MASHUP_COLORS, 2, 0, MASHUP_COLORS[0])
        self.v_song_preview = tk.BooleanVar(value=False)
        ttk.Checkbutton(opts, text="Song preview only (30 s from the musictrack's previewLoopStart)",
                        variable=self.v_song_preview).grid(row=3, column=0, columnspan=6, sticky="w", padx=4, pady=(4, 0))
        r += 1

        act = ttk.Frame(root)
        act.grid(row=r, column=0, columnspan=3, sticky="ew", **pad)
        ttk.Label(act, text="Preview at (s)").pack(side="left")
        self.v_ptime = tk.StringVar(value="30")
        ttk.Entry(act, textvariable=self.v_ptime, width=7).pack(side="left", padx=4)
        self.b_preview = ttk.Button(act, text="Preview frame", command=self.preview)
        self.b_preview.pack(side="left", padx=4)
        ttk.Button(act, text="Mainsequence editor…", command=self.open_editor).pack(side="left", padx=12)
        self.b_stop = ttk.Button(act, text="Stop", command=self.stop, state="disabled")
        self.b_stop.pack(side="right", padx=4)
        self.b_render = ttk.Button(act, text="Render video", command=self.render)
        self.b_render.pack(side="right", padx=4)
        r += 1

        self.progress = ttk.Progressbar(root, maximum=100)
        self.progress.grid(row=r, column=0, columnspan=3, sticky="ew", **pad)
        r += 1
        self.v_status = tk.StringVar(value="Pick a mainsequence (.tape.ckd) to start.")
        ttk.Label(root, textvariable=self.v_status).grid(row=r, column=0, columnspan=3, sticky="w", **pad)
        r += 1

        bottom = ttk.Frame(root)
        bottom.grid(row=r, column=0, columnspan=3, sticky="nsew", **pad)
        root.rowconfigure(r, weight=1)
        bottom.columnconfigure(1, weight=1)
        bottom.rowconfigure(0, weight=1)
        self.canvas = tk.Label(bottom, background="#111", width=48, height=13, text="preview",
                               foreground="#777")
        self.canvas.grid(row=0, column=0, sticky="nsw", padx=(0, 6))
        self.log = tk.Text(bottom, height=12, wrap="none", font=("Consolas", 9))
        self.log.grid(row=0, column=1, sticky="nsew")

    def _combo(self, parent, label, values, row, col, default, width=14):
        v = tk.StringVar(value=default)
        ttk.Label(parent, text=label).grid(row=row, column=col, sticky="w", padx=4, pady=3)
        ttk.Combobox(parent, textvariable=v, values=values, state="readonly", width=width).grid(
            row=row, column=col + 1, sticky="w", padx=4, pady=3)
        return v

    # -- mainsequence editor ------------------------------------------------------------
    def open_editor(self):
        from mainseq_editor import EditorWindow
        from jdstage.timeline import map_dir_of
        seq = self.v_seq.get().strip()
        map_dir = map_dir_of(seq) if seq and os.path.isfile(seq) else ""

        def saved(path):
            self.v_seq.set(path)
            if not self.v_out.get():
                self.v_out.set(os.path.join(DOWNLOADS, f"{map_name_from(path)}.mp4"))
            self.status(f"Mainsequence saved: {path}")
        EditorWindow(self, map_dir, on_saved=saved)

    # -- pickers ----------------------------------------------------------------------
    def pick_seq(self):
        p = filedialog.askopenfilename(title="Mainsequence",
                                       filetypes=[("Tape", "*.tape.ckd *.tape"), ("All", "*.*")])
        if p:
            self.v_seq.set(p)
            if not self.v_out.get():
                self.v_out.set(os.path.join(DOWNLOADS, f"{map_name_from(p)}.mp4"))
            self.status(f"Map: {map_name_from(p)}")

    def pick_file(self, var, title, types):
        p = filedialog.askopenfilename(title=title, filetypes=types)
        if p:
            var.set(p)

    def pick_out(self):
        p = filedialog.asksaveasfilename(title="Output video", defaultextension=".mp4",
                                         filetypes=[("MP4", "*.mp4"), ("MKV", "*.mkv"), ("WebM", "*.webm"),
                                                    ("PNG sequence", "*.png")],
                                         initialdir=DOWNLOADS)
        if p:
            self.v_out.set(p)

    # -- command building -------------------------------------------------------------
    def base_args(self):
        seq = self.v_seq.get().strip()
        if not seq or not os.path.isfile(seq):
            messagebox.showerror("JD Stage Render", "Pick a valid mainsequence file first.")
            return None
        args = [sys.executable, "-u", CLI, seq]
        if self.v_no_coach.get():
            args.append("--no-coach")
        elif self.v_coach.get().strip():
            args += ["--coach", self.v_coach.get().strip()]
        if self.v_no_audio.get():
            args.append("--no-audio")
        elif self.v_audio.get().strip():
            args += ["--audio", self.v_audio.get().strip()]
        if self.v_mcolor.get() != MASHUP_COLORS[0]:
            args += ["--mashup-color", self.v_mcolor.get()]
        args += ["--fps", self.v_fps.get()]
        return args

    # -- actions ----------------------------------------------------------------------
    def preview(self):
        args = self.base_args()
        if args is None:
            return
        try:
            t = float(self.v_ptime.get())
        except ValueError:
            messagebox.showerror("JD Stage Render", "Preview time must be a number of seconds.")
            return
        tmp = os.path.join(tempfile.gettempdir(), "jdstage_preview.png")
        args += ["--still", str(t), "--size", "640x360", "-o", tmp]
        expected = os.path.join(tempfile.gettempdir(), f"jdstage_preview_{t:.2f}s.png")
        self.run(args, on_done=lambda ok: self.show_preview(expected) if ok else None, label="Preview")

    def render(self):
        args = self.base_args()
        if args is None:
            return
        song_preview = self.v_song_preview.get()
        suffix = "_preview" if song_preview else ""
        out = self.v_out.get().strip() or os.path.join(DOWNLOADS, f"{map_name_from(self.v_seq.get())}{suffix}.mp4")
        self.v_out.set(out)
        crf, preset = QUALITIES[self.v_quality.get()]
        if song_preview:
            args.append("--preview")
        args += ["--size", self.v_res.get(), "--crf", str(crf), "--preset", preset,
                 "--workers", self.v_workers.get(), "-o", out]
        self.run(args, on_done=lambda ok: self.status(f"Done: {out}" if ok else "Render failed or stopped."),
                 label="Render")

    def show_preview(self, path):
        if not os.path.isfile(path):
            self.status("Preview failed (see log).")
            return
        img = tk.PhotoImage(file=path)
        # fit into ~320px wide
        factor = max(1, img.width() // 320)
        img = img.subsample(factor, factor)
        self.preview_img = img
        self.canvas.configure(image=img, text="", width=img.width(), height=img.height())
        self.status(f"Preview at {self.v_ptime.get()} s")

    def run(self, args, on_done, label):
        if self.proc is not None:
            return
        self.log.delete("1.0", "end")
        self.append(" ".join(f'"{a}"' if " " in a else a for a in args[2:]) + "\n")
        self.progress["value"] = 0
        self.status(f"{label} running…")
        self.b_render.configure(state="disabled")
        self.b_preview.configure(state="disabled")
        self.b_stop.configure(state="normal")
        self.proc = subprocess.Popen(args, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     creationflags=NO_WINDOW)
        threading.Thread(target=self._pump, args=(self.proc, on_done), daemon=True).start()

    def _pump(self, proc, on_done):
        buf = b""
        fd = proc.stdout.fileno()
        while True:
            chunk = os.read(fd, 4096)
            if not chunk:
                break
            buf += chunk
            parts = re.split(rb"[\r\n]", buf)
            buf = parts.pop()
            for p in parts:
                if p.strip():
                    self.after(0, self._line, p.decode("utf-8", "replace"))
        code = proc.wait()
        self.after(0, self._finished, code == 0, on_done)

    def _line(self, line):
        m = PROGRESS_RE.search(line)
        if m:
            done, total, fps, eta = int(m.group(1)), int(m.group(2)), m.group(3), int(m.group(4))
            self.progress["value"] = 100.0 * done / max(1, total)
            self.status(f"Rendering {done}/{total} frames  {fps} fps  ~{eta // 60}m{eta % 60:02d}s left")
            return
        if "no tape labelled" in line:
            return
        self.append(line + "\n")

    def _finished(self, ok, on_done):
        self.proc = None
        self.b_render.configure(state="normal")
        self.b_preview.configure(state="normal")
        self.b_stop.configure(state="disabled")
        if ok:
            self.progress["value"] = 100
        on_done(ok)

    def stop(self):
        if self.proc is None:
            return
        # kill the whole tree (render worker pool + ffmpeg)
        subprocess.run(["taskkill", "/PID", str(self.proc.pid), "/T", "/F"], capture_output=True,
                       creationflags=NO_WINDOW)
        self.status("Stopped.")

    def append(self, text):
        self.log.insert("end", text)
        self.log.see("end")

    def status(self, text):
        self.v_status.set(text)


if __name__ == "__main__":
    App().mainloop()
