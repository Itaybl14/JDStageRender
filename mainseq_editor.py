#!/usr/bin/env python3
"""
Mainsequence editor for custom mashup (MU) maps.

Each section is just a colour, a start and an end. Scroll direction and the duo layout
are filled in automatically the way official MU maps do it (scroll alternates left/right,
duo layout on). The live preview shows the coach video cut out with its alpha mask over a
solid background in the colour of the section at the preview time.

Styled after the Dance Editor timeline (tools/framework/JD.Timeline StyleSheet / ClipWidget /
BeatsTimeline): dark beat ruler, dark track header, gradient track row, clips with a
+10/-10 gradient and a -30 edge, selected clip +100 with dark text and a white edge,
red time cursor.

Saving writes only <map>/cinematics/<map>_mainsequence.tape.ckd (old one backed up).
"""

import os
import queue
import random
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from jdstage import mashup_seq as MS  # noqa: E402
from jdstage import video as V  # noqa: E402
from jdstage.assets import parse_data  # noqa: E402
from jdstage.timeline import Timeline, find_coach_video, find_musictrack, load_musictrack  # noqa: E402

UI = os.path.join(HERE, "templates", "ui")
PREVIEW_W, PREVIEW_H = 480, 270
PLAY_FPS = 20.0


# ----------------------------------------------------------------------------------------
# Dance Editor palette
# ----------------------------------------------------------------------------------------

def rgb(r, g, b):
    return f"#{max(0, min(255, r)):02x}{max(0, min(255, g)):02x}{max(0, min(255, b)):02x}"


def add(color, d):
    return rgb(int(color[1:3], 16) + d, int(color[3:5], 16) + d, int(color[5:7], 16) + d)


BACKGROUND = rgb(78, 78, 78)
LINE = rgb(60, 60, 60)
TRACK_LIGHT, TRACK_DARK = rgb(78, 78, 78), rgb(70, 70, 70)
TRACK_FORE = rgb(180, 180, 180)
HEADER_BG = rgb(35, 35, 35)
RULER_BLOCK = rgb(80, 80, 80)
PAST_END = "#b22222"
WHITESMOKE, LIGHTGRAY = "#f5f5f5", "#d3d3d3"
CURSOR = "#ff3b3b"
SELECTED_TEXT = rgb(60, 60, 60)
CLIP_COLORS = {"purple": rgb(122, 62, 170), "green": rgb(46, 140, 72), "blue": rgb(40, 96, 178),
               "orange": rgb(196, 112, 34)}
# preview backgrounds: the stage's colour sets
PREVIEW_BG = {"purple": (124, 40, 200), "green": (30, 170, 70), "blue": (30, 110, 230), "orange": (235, 120, 20),
              None: (40, 40, 44)}
CLIP_FONT = ("Microsoft Sans Serif", 7)
UI_FONT = ("Microsoft Sans Serif", 8)

HEADER_W, RULER_H, TRACK_H = 90, 26, 34
EDGE_GRAB = 5

COLOR_ORDERS = {
    "purple → green → blue → orange": ["purple", "green", "blue", "orange"],
    "blue → orange → purple → green": ["blue", "orange", "purple", "green"],
    "green → purple → orange → blue": ["green", "purple", "orange", "blue"],
    "random (no repeats)": None,
}


def fmt_time(sec):
    sign = "-" if sec < 0 else ""
    sec = abs(sec)
    return f"{sign}{int(sec // 60)}:{sec % 60:05.2f}"


MIN_LEN = MS.TICK  # shortest section: one tick (1/24 beat)


def parse_beat(text):
    """'b12', 'b12.5', 'b12:5' (= beat 12 + 5/24) -> beats, or None if not a beat entry."""
    text = text.strip().lower()
    if not text.startswith("b"):
        return None
    text = text[4:] if text.startswith("beat") else text[1:]
    text = text.strip()
    if ":" in text:
        b, k = text.split(":", 1)
        return int(b or 0) + int(k or 0) / MS.FRAMES_PER_BEAT
    return float(text)


def parse_time(text):
    text = text.strip()
    if ":" in text:
        m, s = text.split(":", 1)
        return int(m or 0) * 60 + float(s)
    return float(text)


def read_map_name(map_dir):
    for f in os.listdir(map_dir):
        if f.lower() in ("songdesc.tpl.ckd", "songdesc.tpl"):
            try:
                with open(os.path.join(map_dir, f), "rb") as fh:
                    d = parse_data(fh.read(), f)
                for c in d.get("COMPONENTS", []) or []:
                    if c.get("MapName"):
                        return c["MapName"]
            except Exception:
                pass
    return os.path.basename(os.path.normpath(map_dir))


def icon(name, shrink=1):
    try:
        img = tk.PhotoImage(file=os.path.join(UI, name))
        return img.subsample(shrink, shrink) if shrink > 1 else img
    except tk.TclError:
        return None


def normalize(sections):
    """Automatic per-section settings, as official MU maps: scroll alternates, duo layout on."""
    sections.sort(key=lambda s: s.start)
    for i, s in enumerate(sections):
        s.scroll = MS.SCROLLS[i % 2]
        s.duo = True


# ----------------------------------------------------------------------------------------
# Coach video source (background thread)
# ----------------------------------------------------------------------------------------

class CoachSource:
    """Decodes masked coach frames (RGBA, preview size) for single seeks and for playback."""

    def __init__(self):
        self.ffmpeg = None
        self.info = None
        self.video_start = 0.0
        self.requests = queue.Queue()
        self.results = queue.Queue()
        self._play = None  # (t0, generation)
        self._gen = 0
        threading.Thread(target=self._loop, daemon=True).start()

    def open(self, path, video_start):
        self.info = None
        if not path:
            return "no coach video found"
        try:
            self.ffmpeg = V.find_ffmpeg()
            self.info = V.CoachInfo(path, V.find_ffprobe(self.ffmpeg))
            self.video_start = video_start
            return f"coach: {os.path.basename(path)} ({self.info.mode})"
        except Exception as exc:
            return f"coach video unreadable: {exc}"

    def seek(self, t):
        self._gen += 1
        self.requests.put(("seek", t, self._gen))

    def play(self, t):
        self._gen += 1
        self.requests.put(("play", t, self._gen))

    def stop(self):
        self._gen += 1
        self.requests.put(("stop", 0, self._gen))

    def _reader(self, t, fps):
        return V.CoachReader(self.ffmpeg, self.info, fps, t - self.video_start, PREVIEW_W, PREVIEW_H)

    def _frame(self, reader):
        f = reader.next_native()
        if f is None:
            return None
        return cv2.resize(f, (PREVIEW_W, PREVIEW_H), interpolation=cv2.INTER_AREA)

    def _loop(self):
        while True:
            kind, t, gen = self.requests.get()
            while not self.requests.empty():  # only the latest request matters
                kind, t, gen = self.requests.get()
            if kind == "stop" or self.info is None:
                if kind != "stop":
                    self.results.put((gen, t, None))
                continue
            if kind == "seek":
                r = self._reader(t, 25)
                frame = self._frame(r)
                r.close()
                self.results.put((gen, t, frame))
                continue
            # play: stream until a newer request arrives
            r = self._reader(t, PLAY_FPS)
            start_wall = time.perf_counter()
            n = 0
            while self.requests.empty():
                frame = self._frame(r)
                ft = t + n / PLAY_FPS
                self.results.put((gen, ft, frame))
                n += 1
                if frame is None and r.done:
                    break
                ahead = start_wall + n / PLAY_FPS - time.perf_counter()
                if ahead > 0:
                    time.sleep(ahead)
            r.close()


# ----------------------------------------------------------------------------------------
# Timeline (single Colour track, Dance Editor style)
# ----------------------------------------------------------------------------------------

class TimelineView(ttk.Frame):
    def __init__(self, master, ed):
        super().__init__(master)
        self.ed = ed
        self.ppb, self.offset = 8.0, 0.0
        self.drag = None
        self.canvas = tk.Canvas(self, background=BACKGROUND, highlightthickness=0, height=RULER_H + TRACK_H + 4)
        self.canvas.grid(row=0, column=0, sticky="ew")
        self.hbar = ttk.Scrollbar(self, orient="horizontal", command=self._on_scrollbar)
        self.hbar.grid(row=1, column=0, sticky="ew")
        self.columnconfigure(0, weight=1)
        self.tape_icon = icon("Icon_20x20_tape.png")
        c = self.canvas
        c.bind("<Configure>", lambda e: self.redraw())
        c.bind("<Button-1>", self.on_press)
        c.bind("<B1-Motion>", self.on_drag)
        c.bind("<ButtonRelease-1>", self.on_release)
        c.bind("<Double-Button-1>", self.on_double)
        c.bind("<Button-3>", self.on_context)
        c.bind("<Motion>", self.on_motion)
        c.bind("<MouseWheel>", self.on_wheel)

    @property
    def width(self):
        return max(self.canvas.winfo_width(), 300)

    def x_of(self, b):
        return HEADER_W + (b - self.offset) * self.ppb

    def beat_of(self, x):
        return self.offset + (x - HEADER_W) / self.ppb

    def total(self):
        return max(self.ed.song_end, self.ed.seq.total_end(), 16.0) + 4

    def fit(self):
        self.ppb = max(1.0, (self.width - HEADER_W - 8) / self.total())
        self.offset = 0.0
        self.redraw()

    def zoom(self, f, ax=None):
        ax = ax if ax is not None else HEADER_W + (self.width - HEADER_W) / 2
        b = self.beat_of(ax)
        self.ppb = min(1200.0, max(1.0, self.ppb * f))
        self.offset = b - (ax - HEADER_W) / self.ppb
        self._clamp()
        self.redraw()

    def _clamp(self):
        vis = (self.width - HEADER_W) / self.ppb
        self.offset = max(-2.0, min(self.offset, max(0.0, self.total() - vis)))

    def _on_scrollbar(self, *a):
        vis = (self.width - HEADER_W) / self.ppb
        if a[0] == "moveto":
            self.offset = float(a[1]) * self.total()
        else:
            self.offset += int(a[1]) * vis * (0.9 if a[2] == "pages" else 0.1)
        self._clamp()
        self.redraw()

    def ensure_visible(self, b0, b1):
        vis = (self.width - HEADER_W) / self.ppb
        if b0 < self.offset or b1 > self.offset + vis:
            self.offset = b0 - vis * 0.1
            self._clamp()

    # -- paint ---------------------------------------------------------------------------
    def _grad(self, x0, y0, x1, y1, top, bottom, bands=5):
        t = [int(top[i:i + 2], 16) for i in (1, 3, 5)]
        b = [int(bottom[i:i + 2], 16) for i in (1, 3, 5)]
        h = (y1 - y0) / bands
        for k in range(bands):
            f = k / max(1, bands - 1)
            self.canvas.create_rectangle(x0, y0 + k * h, x1, y0 + (k + 1) * h + 1, outline="",
                                         fill=rgb(*[int(t[j] + (b[j] - t[j]) * f) for j in range(3)]))

    def redraw(self):
        c = self.canvas
        c.delete("all")
        W, ed = self.width, self.ed
        self._grad(HEADER_W, RULER_H, W, RULER_H + TRACK_H, TRACK_LIGHT, TRACK_DARK)
        c.create_line(HEADER_W, RULER_H + TRACK_H - 1, W, RULER_H + TRACK_H - 1, fill=LINE)
        # ruler: measure blocks, WhiteSmoke measure ticks + numbers, LightGray beat ticks
        first = max(0, int(self.offset // 4) * 4)
        last = self.beat_of(W)
        m = first
        while m <= last:
            c.create_rectangle(self.x_of(m), 0, self.x_of(m + 4), RULER_H - 1, outline=LINE,
                               fill=PAST_END if m >= ed.song_end else RULER_BLOCK)
            m += 4
        every = 1
        while self.ppb * 4 * every < 34:
            every *= 2
        b = first
        while b <= last:
            x = self.x_of(b)
            if b % 4 == 0:
                c.create_line(x, 0, x, RULER_H * 0.25, fill=WHITESMOKE)
                c.create_line(x, RULER_H - 1, x, RULER_H * 0.75, fill=WHITESMOKE)
                if (b // 4) % every == 0:
                    c.create_text(x + 3, RULER_H / 2, text=str(int(b // 4) + 1), anchor="w", fill=WHITESMOKE,
                                  font=UI_FONT)
            elif self.ppb > 8:
                c.create_line(x, RULER_H - 1, x, RULER_H * 0.8, fill=LIGHTGRAY)
            # 1/24-beat ticks once they are far enough apart to see
            tick_px = self.ppb / MS.FRAMES_PER_BEAT
            if tick_px >= 5:
                step = 1 if tick_px >= 10 else 2
                for k in range(step, MS.FRAMES_PER_BEAT, step):
                    xt = x + k * tick_px
                    c.create_line(xt, RULER_H - 1, xt, RULER_H * (0.82 if k % 6 == 0 else 0.9), fill=LIGHTGRAY)
            b += 1
        # clips
        y0, y1 = RULER_H + 3, RULER_H + TRACK_H - 4
        for i, s in enumerate(ed.seq.sections):
            x0, x1 = self.x_of(s.start), self.x_of(s.end)
            if x1 < HEADER_W or x0 > W:
                continue
            on = i == ed.selected
            base = add(CLIP_COLORS[s.color], 100) if on else CLIP_COLORS[s.color]
            xa, xb = max(x0, HEADER_W), min(x1, W + 2)
            self._grad(xa, y0, xb, y1, add(base, 10), add(base, -10), bands=3)
            c.create_rectangle(x0, y0, x1, y1, outline=WHITESMOKE if on else add(base, -30))
            tx = xa + 2
            if self.tape_icon is not None and xb - xa > 26:
                c.create_image(tx, (y0 + y1) / 2 + 1, image=self.tape_icon, anchor="w")
                tx += 22
            room = xb - tx - 3
            if room > 14:
                text = f"{i + 1}. {s.color}"
                n = int(room / 5)
                c.create_text(tx, (y0 + y1) / 2, anchor="w", font=CLIP_FONT,
                              fill=SELECTED_TEXT if on else "white",
                              text=text if len(text) <= n else text[:max(1, n - 1)] + "…")
        # song end, cursor, header
        xe = self.x_of(ed.song_end)
        if HEADER_W <= xe <= W:
            c.create_line(xe, RULER_H, xe, RULER_H + TRACK_H, fill=PAST_END, dash=(4, 3))
        xc = self.x_of(ed.cursor)
        if xc >= HEADER_W:
            c.create_line(xc, 0, xc, RULER_H + TRACK_H, fill=CURSOR, width=2)
        c.create_rectangle(0, 0, HEADER_W, RULER_H, fill=HEADER_BG, outline=LINE)
        c.create_text(8, RULER_H / 2, text="Beats", anchor="w", fill=TRACK_FORE, font=UI_FONT)
        c.create_rectangle(0, RULER_H, HEADER_W, RULER_H + TRACK_H, fill=HEADER_BG, outline=LINE)
        c.create_text(8, RULER_H + TRACK_H / 2, text="Colour", anchor="w", fill=TRACK_FORE, font=UI_FONT)
        tot, vis = self.total(), (W - HEADER_W) / self.ppb
        self.hbar.set(max(0.0, self.offset / tot), min(1.0, (self.offset + vis) / tot))

    # -- interaction ----------------------------------------------------------------------
    def hit(self, x, y):
        if not (RULER_H <= y <= RULER_H + TRACK_H) or x < HEADER_W:
            return None
        for i, s in enumerate(self.ed.seq.sections):
            x0, x1 = self.x_of(s.start), self.x_of(s.end)
            if x0 - EDGE_GRAB <= x <= x1 + EDGE_GRAB:
                if abs(x - x1) <= EDGE_GRAB:
                    return i, "end"
                if abs(x - x0) <= EDGE_GRAB:
                    return i, "start"
                if x0 <= x <= x1:
                    return i, "body"
        return None

    def on_press(self, e):
        self.canvas.focus_set()
        if e.y < RULER_H and e.x >= HEADER_W:
            self.drag = {"part": "scrub"}
            self.ed.set_cursor(self.beat_of(e.x))
            return
        h = self.hit(e.x, e.y)
        if h is None:
            return
        i, part = h
        self.ed.select(i, move_cursor=part == "body")
        s = self.ed.seq.sections[i]
        self.drag = {"i": i, "part": part, "x0": e.x, "start": s.start, "end": s.end, "moved": False}

    def on_drag(self, e):
        d = self.drag
        if not d:
            return
        if d["part"] == "scrub":
            self.ed.set_cursor(self.beat_of(e.x))
            return
        # snap to whole beats; hold Shift for 1/24-beat ticks
        step = MS.TICK if e.state & 0x0001 else 1.0
        delta = (e.x - d["x0"]) / self.ppb
        s = self.ed.seq.sections[d["i"]]
        if d["part"] == "end":
            s.length = max(MIN_LEN, MS.snap(d["end"] + delta, step) - s.start)
        elif d["part"] == "start":
            new_start = min(MS.snap(d["start"] + delta, step), d["end"] - MIN_LEN)
            s.start, s.length = new_start, d["end"] - new_start
        else:
            s.start = MS.snap(d["start"] + delta, step)
        d["moved"] = True
        self.ed.live_edit(s)

    def on_release(self, e):
        d, self.drag = self.drag, None
        if d and d.get("moved"):
            self.ed.commit(self.ed.seq.sections[d["i"]])

    def on_double(self, e):
        if self.hit(e.x, e.y) is None and RULER_H <= e.y <= RULER_H + TRACK_H and e.x >= HEADER_W:
            self.ed.add_section_at(MS.snap(self.beat_of(e.x), MS.TICK if e.state & 0x0001 else 1.0))

    def on_context(self, e):
        h = self.hit(e.x, e.y)
        if h is None:
            return
        self.ed.select(h[0])
        m = tk.Menu(self, tearoff=0)
        for col in MS.COLORS:
            m.add_command(label=col.capitalize(), command=lambda c=col: self.ed.set_color(c))
        m.add_separator()
        m.add_command(label="Delete", command=self.ed.delete_section)
        m.tk_popup(e.x_root, e.y_root)

    def on_motion(self, e):
        h = self.hit(e.x, e.y)
        self.canvas.configure(cursor="sb_h_double_arrow" if h and h[1] != "body" else ("fleur" if h else ""))
        if e.x >= HEADER_W:
            self.ed.show_hover(self.beat_of(e.x))

    def on_wheel(self, e):
        if e.state & 0x4:
            self.zoom(1.25 if e.delta > 0 else 0.8, e.x)
        else:
            self.offset -= (e.delta / 120) * 40 / self.ppb
            self._clamp()
            self.redraw()


# ----------------------------------------------------------------------------------------
# Editor window
# ----------------------------------------------------------------------------------------

class EditorWindow(tk.Toplevel):
    def __init__(self, master=None, map_dir="", on_saved=None):
        super().__init__(master)
        self.title("Mainsequence Editor")
        self.geometry("1040x640")
        self.minsize(860, 560)
        try:
            self.iconbitmap(os.path.join(UI, "DanceEditor.ico"))
        except tk.TclError:
            pass
        self.on_saved = on_saved
        self.map_dir = ""
        self.timeline = Timeline()
        self.seq = MS.MashupSequence("map")
        self.song_end = 256.0
        self.selected = None
        self.cursor = 0.0
        self.dirty = False
        self.playing = False
        self._coach = None          # last decoded coach frame (RGBA, preview size)
        self._coach_t = None
        self._photo = None
        self._icons = {}
        self._loading = False
        self.coach = CoachSource()
        self._build()
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.bind("<Delete>", lambda e: self.delete_section() if not isinstance(e.widget, ttk.Entry) else None)
        self.bind("<Control-s>", lambda e: self.save())
        self.bind("<space>", lambda e: self.toggle_play() if not isinstance(e.widget, ttk.Entry) else None)
        # Enter always pauses/resumes, wherever the focus is (in the Start/End fields it
        # also applies the typed time first: the field's own binding runs before this one)
        self.bind("<Return>", lambda e: self.toggle_play())
        self.bind("<KP_Enter>", lambda e: self.toggle_play())
        self.after(40, self._poll)
        self.after(50, self.view.fit)
        if map_dir:
            self.after(60, lambda: self.load_map(map_dir))

    # -- layout ----------------------------------------------------------------------------
    def _build(self):
        mb = tk.Menu(self)
        fm = tk.Menu(mb, tearoff=0)
        fm.add_command(label="Open map folder…", command=self.pick_dir)
        fm.add_command(label="Save", accelerator="Ctrl+S", command=self.save)
        fm.add_separator()
        fm.add_command(label="Close", command=self.close)
        mb.add_cascade(label="File", menu=fm)
        self.configure(menu=mb)

        tb = ttk.Frame(self, padding=(4, 3))
        tb.pack(side="top", fill="x")

        def tbtn(text, img, cmd, shrink=1):
            ic = icon(img, shrink) if img else None
            self._icons[text] = ic
            ttk.Button(tb, text=text, image=ic, compound="left" if ic else "none", command=cmd).pack(side="left", padx=1)

        tbtn("Open map", "Icon_30x30_open_file.png", self.pick_dir, 2)
        tbtn("Save", "save.png", self.save)
        ttk.Separator(tb, orient="vertical").pack(side="left", fill="y", padx=6, pady=2)
        tbtn("Add section", "Icon_16x16_AddModel.png", self.add_section)
        tbtn("Delete", "Icons_16x16_Remove.png", self.delete_section)
        tbtn("Fill song…", "Icon_20x20_color.png", self.autofill_dialog)
        ttk.Button(tb, text="Fit", command=lambda: self.view.fit()).pack(side="right", padx=1)
        ttk.Button(tb, text="−", width=3, command=lambda: self.view.zoom(0.8)).pack(side="right", padx=1)
        ttk.Button(tb, text="+", width=3, command=lambda: self.view.zoom(1.25)).pack(side="right", padx=1)

        self.v_status = tk.StringVar(value="Open a map folder to start.")
        self.v_pos = tk.StringVar()
        sb = ttk.Frame(self, relief="sunken", padding=(6, 2))
        sb.pack(side="bottom", fill="x")
        ttk.Label(sb, textvariable=self.v_status).pack(side="left")
        ttk.Label(sb, textvariable=self.v_pos).pack(side="right")

        self.v_info = tk.StringVar(value="No map loaded")
        ttk.Label(self, textvariable=self.v_info, padding=(6, 2)).pack(side="top", anchor="w")
        self.view = TimelineView(self, self)
        self.view.pack(side="top", fill="x", padx=4)

        body = ttk.Frame(self, padding=(4, 8, 4, 4))
        body.pack(side="top", fill="both", expand=True)

        # section: colour / start / end
        props = ttk.LabelFrame(body, text="Section", padding=10)
        props.pack(side="left", fill="y")
        self.v_index = tk.StringVar(value="none selected")
        ttk.Label(props, textvariable=self.v_index).grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 8))
        ttk.Label(props, text="Colour").grid(row=1, column=0, sticky="w")
        sw = ttk.Frame(props)
        sw.grid(row=1, column=1, columnspan=3, sticky="w", pady=4)
        self.swatches = {}
        for col in MS.COLORS:
            b = tk.Button(sw, width=8, height=2, text=col.capitalize(), fg="white", bg=CLIP_COLORS[col],
                          activebackground=add(CLIP_COLORS[col], 30), relief="raised", bd=2,
                          command=lambda c=col: self.set_color(c))
            b.pack(side="left", padx=2)
            self.swatches[col] = b
        self.v_start = tk.StringVar()
        self.v_end = tk.StringVar()
        self.v_start_beat = tk.StringVar()
        self.v_end_beat = tk.StringVar()
        for r, (label, var, bvar, which) in enumerate((("Start", self.v_start, self.v_start_beat, "start"),
                                                       ("End", self.v_end, self.v_end_beat, "end")), start=2):
            ttk.Label(props, text=label).grid(row=r, column=0, sticky="w", pady=6)
            ent = ttk.Entry(props, textvariable=var, width=10, font=("Consolas", 11))
            ent.grid(row=r, column=1, sticky="w")
            ent.bind("<Return>", lambda e, w=which: self.apply_time(w))
            ent.bind("<FocusOut>", lambda e, w=which: self.apply_time(w))
            nb = ttk.Frame(props)
            nb.grid(row=r, column=2, sticky="w", padx=6)
            for txt, d in (("◀◀", -1.0), ("◀", -MS.TICK), ("▶", MS.TICK), ("▶▶", 1.0)):
                ttk.Button(nb, text=txt, width=3 if len(txt) == 1 else 4,
                           command=lambda w=which, d=d: self.nudge(w, d)).pack(side="left")
            ttk.Label(props, textvariable=bvar, foreground="#666").grid(row=r, column=3, sticky="w")
        ttk.Label(props, text="Type a time (m:ss.ss or seconds) or a beat (b12, b12:5 = beat 12 + 5/24).\n"
                              "Everything snaps to 1/24 beat (one tape tick).\n"
                              "◀ ▶ move by 1/24 beat, ◀◀ ▶▶ by one beat.\n"
                              "Dragging snaps to beats; hold Shift for 1/24 beat.",
                  foreground="#666", justify="left").grid(row=4, column=0, columnspan=4, sticky="w", pady=(10, 0))

        # live preview
        pv = ttk.LabelFrame(body, text="Live preview", padding=8)
        pv.pack(side="left", fill="both", expand=True, padx=(8, 0))
        self.preview = tk.Label(pv, background="#111")
        self.preview.pack(side="top")
        ctl = ttk.Frame(pv)
        ctl.pack(side="top", fill="x", pady=(6, 0))
        self.b_play = ttk.Button(ctl, text="▶ Play", width=11, command=self.toggle_play)
        self.b_play.pack(side="left")
        self.v_ptime = tk.StringVar(value="0:00.00")
        ttk.Label(ctl, textvariable=self.v_ptime, font=("Consolas", 11)).pack(side="left", padx=10)
        self.v_coach = tk.StringVar()
        ttk.Label(ctl, textvariable=self.v_coach, foreground="#666").pack(side="right")
        self._show(self._compose())

    # -- map ---------------------------------------------------------------------------
    def pick_dir(self):
        d = filedialog.askdirectory(title="Map folder", parent=self)
        if d:
            self.load_map(d)

    def load_map(self, d):
        d = os.path.abspath(d) if d else ""
        if not d or not os.path.isdir(d):
            messagebox.showerror("Mainsequence Editor", "Pick an existing map folder.", parent=self)
            return
        if self.dirty and not messagebox.askyesno("Mainsequence Editor", "Discard unsaved changes?", parent=self):
            return
        self.stop_play()
        self.map_dir = d
        name = read_map_name(d)
        note = ""
        mt = find_musictrack(d, name)
        try:
            self.timeline = load_musictrack(mt) if mt else Timeline()
        except Exception as exc:
            self.timeline = Timeline()
            note = f"  (musictrack unreadable: {exc})"
        if self.timeline.has_markers and self.timeline.end_beat:
            self.song_end = float(self.timeline.end_beat - 1)
        else:
            self.song_end = 256.0
            note += "  (no musictrack: assuming 120 BPM)"
        self.seq = MS.MashupSequence(name)
        existing = MS.mainsequence_path(d, name)
        if os.path.isfile(existing):
            try:
                self.seq = MS.load(existing, name)
            except Exception as exc:
                note += f"  (existing mainsequence unreadable: {exc})"
        self.song_end = max(self.song_end, self.seq.song_end)
        self.v_coach.set(self.coach.open(find_coach_video(d, name), self.timeline.video_start))
        self.title(f"Mainsequence Editor — {name}")
        self.v_info.set(f"{name}    {fmt_time(self._t(self.song_end))} ({self.song_end:g} beats)    "
                        f"{len(self.seq.sections)} sections{note}")
        self.dirty = False
        self.selected = None
        self.view.fit()
        if self.seq.sections:
            self.select(0)
        else:
            self.set_cursor(0.0)
        self.validate()

    def _t(self, beat):
        return self.timeline.time_at_beat(beat)

    def _b(self, sec):
        return self.timeline.beat_at(sec)

    # -- selection / editing -------------------------------------------------------------
    def select(self, i, move_cursor=True):
        if i is None or not (0 <= i < len(self.seq.sections)):
            self.selected = None
            self.v_index.set("none selected")
            self.view.redraw()
            return
        self.selected = i
        s = self.seq.sections[i]
        self._fill_fields(s)
        self.view.ensure_visible(s.start, s.end)
        if move_cursor and not (s.start <= self.cursor < s.end):
            self.set_cursor(s.start + min(2.0, s.length / 2))
        else:
            self.view.redraw()
            self._refresh_preview()

    def _fill_fields(self, s):
        i = self.seq.sections.index(s)
        self.v_index.set(f"Section {i + 1} of {len(self.seq.sections)}")
        self.v_start.set(fmt_time(self._t(s.start)))
        self.v_end.set(fmt_time(self._t(s.end)))
        self.v_start_beat.set(f"beat {MS.fmt_beat(s.start)}")
        self.v_end_beat.set(f"beat {MS.fmt_beat(s.end)}")
        for col, b in self.swatches.items():
            b.configure(relief="sunken" if col == s.color else "raised", bd=4 if col == s.color else 2)

    def live_edit(self, s):
        self.dirty = True
        self._fill_fields(s)
        self.view.redraw()
        self._refresh_preview()

    def commit(self, s):
        self.dirty = True
        normalize(self.seq.sections)
        self.select(self.seq.sections.index(s), move_cursor=False)
        self.validate()

    def set_color(self, col):
        if self.selected is None:
            return
        s = self.seq.sections[self.selected]
        s.color = col
        self.commit(s)

    def apply_time(self, which):
        if self.selected is None:
            return
        s = self.seq.sections[self.selected]
        var = self.v_start if which == "start" else self.v_end
        try:
            text = var.get()
            pb = parse_beat(text)
            beat = MS.snap(pb if pb is not None else self._b(parse_time(text)))
        except ValueError:
            self._fill_fields(s)
            return
        end = s.end
        if which == "start":
            beat = min(beat, end - MIN_LEN)
            s.start, s.length = beat, end - beat
        else:
            s.length = max(MIN_LEN, beat - s.start)
        self.commit(s)

    def nudge(self, which, beats):
        if self.selected is None:
            return
        s = self.seq.sections[self.selected]
        end = s.end
        if which == "start":
            ns = min(MS.snap(s.start + beats), end - MIN_LEN)
            s.start, s.length = ns, end - ns
        else:
            s.length = max(MIN_LEN, MS.snap(s.end + beats) - s.start)
        self.commit(s)

    def _next_color(self, prev):
        return MS.COLORS[(MS.COLORS.index(prev) + 1) % len(MS.COLORS)] if prev in MS.COLORS else "purple"

    def add_section(self):
        secs = self.seq.sections
        prev = secs[self.selected] if self.selected is not None else (secs[-1] if secs else None)
        start = prev.end if prev else 0.0
        nxt = min((x.start for x in secs if x.start >= start and x is not prev), default=None)
        length = 16.0 if nxt is None or nxt - start >= 16 else max(MIN_LEN, nxt - start)
        s = MS.Section(start, length, self._next_color(prev.color if prev else None))
        secs.append(s)
        self.commit(s)
        self.select(secs.index(s))

    def add_section_at(self, beat):
        secs = self.seq.sections
        prev = max((x for x in secs if x.start <= beat), key=lambda x: x.start, default=None)
        nxt = min((x.start for x in secs if x.start > beat), default=None)
        s = MS.Section(beat, 16.0 if nxt is None else max(MIN_LEN, min(16.0, nxt - beat)),
                       self._next_color(prev.color if prev else None))
        secs.append(s)
        self.commit(s)
        self.select(secs.index(s))

    def delete_section(self):
        if self.selected is None:
            return
        i = self.selected
        del self.seq.sections[i]
        self.dirty = True
        normalize(self.seq.sections)
        self.select(min(i, len(self.seq.sections) - 1) if self.seq.sections else None)
        self.validate()

    def autofill_dialog(self):
        dlg = tk.Toplevel(self)
        dlg.title("Fill song")
        dlg.transient(self)
        dlg.resizable(False, False)
        f = ttk.Frame(dlg, padding=12)
        f.pack()
        v_block = tk.DoubleVar(value=16.0)
        v_order = tk.StringVar(value=list(COLOR_ORDERS)[0])
        ttk.Label(f, text="Section length (beats)").grid(row=0, column=0, sticky="w", pady=3, padx=(0, 10))
        ttk.Spinbox(f, from_=1, to=128, increment=1, textvariable=v_block, width=8).grid(row=0, column=1, sticky="w")
        ttk.Label(f, text="Colours").grid(row=1, column=0, sticky="w", pady=3)
        ttk.Combobox(f, textvariable=v_order, values=list(COLOR_ORDERS), state="readonly", width=30).grid(
            row=1, column=1, sticky="w")
        ttk.Label(f, text="Replaces all sections, from the start to the end of the song.", foreground="#a33").grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(8, 0))

        def ok():
            try:
                block = float(v_block.get())
            except (tk.TclError, ValueError):
                return
            colors = COLOR_ORDERS[v_order.get()]
            if colors is None:
                colors, last = [], None
                for _ in range(int(self.song_end / max(block, 1)) + 2):
                    last = random.choice([x for x in MS.COLORS if x != last])
                    colors.append(last)
            self.seq.sections = MS.autofill(0.0, self.song_end, block, colors)
            normalize(self.seq.sections)
            self.dirty = True
            dlg.destroy()
            self.view.fit()
            self.select(0)
            self.validate()

        bf = ttk.Frame(f)
        bf.grid(row=3, column=0, columnspan=2, sticky="e", pady=(10, 0))
        ttk.Button(bf, text="Fill", command=ok).pack(side="left", padx=4)
        ttk.Button(bf, text="Cancel", command=dlg.destroy).pack(side="left")
        dlg.grab_set()

    def validate(self):
        problems = []
        secs = self.seq.sections
        for a, b in zip(secs, secs[1:]):
            if b.start < a.end - 1e-6:
                problems.append(f"overlap at {fmt_time(self._t(b.start))}")
            elif b.start > a.end + 1e-6:
                problems.append(f"gap {fmt_time(self._t(a.end))}–{fmt_time(self._t(b.start))}")
        if secs and secs[-1].end > self.song_end + 1e-6:
            problems.append("last section runs past the song end")
        mark = "  •  unsaved changes" if self.dirty else ""
        self.v_status.set(("⚠ " + "; ".join(problems[:3]) + (" …" if len(problems) > 3 else ""))
                          if problems else f"{len(secs)} sections OK{mark}")

    # -- preview ---------------------------------------------------------------------------
    def color_at(self, beat):
        for s in self.seq.sections:
            if s.start <= beat < s.end:
                return s.color
        return None

    def set_cursor(self, beat):
        self.stop_play()
        self.cursor = max(0.0, MS.snap(beat, 0.05))
        self.view.redraw()
        self._refresh_preview(seek=True)

    def show_hover(self, beat):
        self.v_pos.set(f"{fmt_time(self._t(beat))}  (beat {MS.fmt_beat(MS.snap(beat))})")

    def _refresh_preview(self, seek=False):
        t = self._t(self.cursor)
        self.v_ptime.set(fmt_time(t))
        if seek or self._coach_t is None or abs(self._coach_t - t) > 0.02:
            if not self.playing:
                self.coach.seek(t)
        self._show(self._compose())

    def _compose(self):
        bg = np.empty((PREVIEW_H, PREVIEW_W, 3), np.float32)
        bg[:] = np.array(PREVIEW_BG[self.color_at(self.cursor)], np.float32) / 255.0
        if self._coach is not None:
            f = self._coach.astype(np.float32) / 255.0
            a = f[..., 3:4]
            bg = f[..., :3] * a + bg * (1.0 - a)
        return (np.clip(bg, 0, 1) * 255 + 0.5).astype(np.uint8)

    def _show(self, img):
        ppm = b"P6 %d %d 255\n" % (img.shape[1], img.shape[0]) + img.tobytes()
        self._photo = tk.PhotoImage(data=ppm, format="PPM")
        self.preview.configure(image=self._photo)

    def _poll(self):
        latest = None
        try:
            while True:
                latest = self.coach.results.get_nowait()
        except queue.Empty:
            pass
        if latest is not None:
            gen, t, frame = latest
            if gen == self.coach._gen:
                self._coach, self._coach_t = frame, t
                if self.playing:
                    self.cursor = self._b(t)
                    self.v_ptime.set(fmt_time(t))
                    self.view.redraw()
                    if self.cursor >= self.song_end:
                        self.stop_play()
                self._show(self._compose())
        self.after(15, self._poll)

    def toggle_play(self):
        if self.playing:
            self.stop_play()
        elif self.map_dir:
            self.playing = True
            self.b_play.configure(text="❚❚ Pause")
            self.coach.play(self._t(self.cursor))

    def stop_play(self):
        if self.playing:
            self.playing = False
            self.coach.stop()
            self.b_play.configure(text="▶ Play")

    # -- save -----------------------------------------------------------------------------
    def save(self):
        if not self.map_dir:
            return
        if not self.seq.sections:
            messagebox.showerror("Save", "Add at least one section.", parent=self)
            return
        normalize(self.seq.sections)
        self.seq.song_end = max(self.song_end, self.seq.total_end())
        try:
            done = MS.write_map(self.map_dir, self.seq)
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc), parent=self)
            return
        self.dirty = False
        self.validate()
        messagebox.showinfo("Saved", "\n".join(f"{act}: {os.path.relpath(p, self.map_dir)}" for p, act in done),
                            parent=self)
        tape = next(p for p, act in done if act == "written")
        if self.on_saved:
            self.on_saved(tape)

    def close(self):
        if self.dirty and not messagebox.askyesno("Mainsequence Editor", "Close without saving?", parent=self):
            return
        self.stop_play()
        self.destroy()


def main():
    app = tk.Tk()
    app.withdraw()
    ed = EditorWindow(app, sys.argv[1] if len(sys.argv) > 1 else "")
    ed.bind("<Destroy>", lambda e: app.destroy() if e.widget is ed else None)
    app.mainloop()


if __name__ == "__main__":
    main()
