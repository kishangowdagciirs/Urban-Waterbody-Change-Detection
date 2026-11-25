import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import threading
import traceback
import queue
import os
import sys

from waterbody_app.analysis import discover_band, run_analysis

BG = "#F2F7FC"
PANEL = "#FFFFFF"
PANEL2 = "#DCEBFA"
ENTRY = "#F8FBFF"
WHITE = "#18324B"
CYAN = "#2878B8"
GREEN = "#23845B"
RED = "#C65362"
MUTED = "#60778D"
BORDER = "#D5E3F0"
ACCENT_PALE = "#EAF3FC"

APP_TITLE = "Sentinel-2 Waterbody Change Analysis | GEE + OFFLINE NDWI"


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1320x850")
        self.minsize(1120, 700)
        self.configure(bg=BG)

        self.earlier_year = tk.StringVar(value="2017")
        self.later_year = tk.StringVar(value="2026")
        self.old_b3 = tk.StringVar()
        self.old_b8 = tk.StringVar()
        self.new_b3 = tk.StringVar()
        self.new_b8 = tk.StringVar()
        self.old_water_mask = tk.StringVar()
        self.new_water_mask = tk.StringVar()
        self.detection_method = tk.StringVar(value="Maximum observed extent (GEE)")
        self.boundary = tk.StringVar()
        self.output = tk.StringVar()
        self.gee_project = tk.StringVar()
        self.earlier_start = tk.StringVar(value="2017-10-01")
        self.earlier_end = tk.StringVar(value="2018-01-01")
        self.later_start = tk.StringVar(value="2026-10-01")
        self.later_end = tk.StringVar(value="2027-01-01")
        self.max_scene_cloud = tk.StringVar(value="80")
        self.threshold = tk.StringVar(value="0.00")
        self.min_pixels = tk.StringVar(value="25")
        self.events = queue.Queue()
        self.worker = None
        self.gee_worker = None
        self.gee_auth_button = None
        self.gee_fetch_button = None
        self.run_button = None
        self.log = None
        self.status = None
        self.progress = None
        self.results_tab = None
        self.dashboard = None
        self.map_image = None
        self.stats_tree = None
        self.kpi_values = {}
        self.chart_canvas = None
        self.map_preview = None

        self._build()
        self.after(100, self._poll_events)

    def _build(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TFrame", background=BG)
        style.configure("P.TFrame", background=PANEL)
        style.configure("TLabel", background=PANEL, foreground=WHITE, font=("Segoe UI", 10))
        style.configure("H.TLabel", background=PANEL, foreground=CYAN, font=("Segoe UI", 12, "bold"))
        style.configure("Title.TLabel", background=BG, foreground=CYAN, font=("Segoe UI", 23, "bold"))
        style.configure("Sub.TLabel", background=BG, foreground=MUTED, font=("Segoe UI", 10))
        style.configure("TButton", background=PANEL2, foreground="#24435F", font=("Segoe UI", 9, "bold"), padding=7)
        style.map("TButton", background=[("active", "#C8DFF4")], foreground=[("active", "#18324B")])
        style.configure("TEntry", fieldbackground=ENTRY, foreground="#18324B", insertcolor="#18324B")
        style.configure("TCheckbutton", background=PANEL, foreground="#18324B")
        style.configure("LogFoot.TLabel", background=PANEL, foreground=MUTED, font=("Segoe UI", 9))
        style.configure("TNotebook", background=PANEL, borderwidth=0)
        style.configure("TNotebook.Tab", background="#E5EFF8", foreground="#35536D", padding=(14, 8), font=("Segoe UI", 9, "bold"))
        style.map("TNotebook.Tab", background=[("selected", "#FFFFFF")], foreground=[("selected", "#2878B8")])
        style.configure("Treeview", background="#FFFFFF", fieldbackground="#FFFFFF", foreground="#24435F", rowheight=26)
        style.configure("Treeview.Heading", background="#EAF3FC", foreground="#24435F", font=("Segoe UI", 9, "bold"))

        ttk.Label(self, text="SENTINEL-2 WATERBODY CHANGE ANALYSIS", style="Title.TLabel").pack(anchor="w", padx=22, pady=(16, 2))
        ttk.Label(self, text="ONLINE GEE COMPOSITES OR LOCAL FILES • NDWI (B3/B8) • 10 m • Any study boundary", style="Sub.TLabel").pack(anchor="w", padx=22, pady=(0, 14))

        main = ttk.Frame(self)
        main.pack(fill="both", expand=True, padx=16, pady=(0, 16))
        left_outer = ttk.Frame(main, style="P.TFrame")
        left_outer.pack(side="left", fill="y", padx=(0, 12))
        left_canvas = tk.Canvas(left_outer, bg=PANEL, highlightthickness=0, width=440)
        left_scrollbar = ttk.Scrollbar(left_outer, orient="vertical", command=left_canvas.yview)
        left_canvas.configure(yscrollcommand=left_scrollbar.set)
        left_scrollbar.pack(side="right", fill="y")
        left_canvas.pack(side="left", fill="y", expand=True)
        left = ttk.Frame(left_canvas, style="P.TFrame", padding=14)
        left_window = left_canvas.create_window((0, 0), window=left, anchor="nw")
        left.bind("<Configure>", lambda _event: left_canvas.configure(scrollregion=left_canvas.bbox("all")))
        left_canvas.bind("<Configure>", lambda event: left_canvas.itemconfigure(left_window, width=event.width))
        left_canvas.bind_all("<MouseWheel>", lambda event: left_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units"))
        right = ttk.Frame(main, style="P.TFrame", padding=14)
        right.pack(side="left", fill="both", expand=True)

        self._section(left, "1  COMPARISON YEARS")
        year_frame = ttk.Frame(left, style="P.TFrame")
        year_frame.pack(fill="x", pady=(0, 10))
        ttk.Label(year_frame, text="Earlier year").grid(row=0, column=0, sticky="w", padx=(0, 8))
        ttk.Entry(year_frame, textvariable=self.earlier_year, width=9).grid(row=0, column=1, sticky="w", padx=(0, 14))
        ttk.Label(year_frame, text="Later year").grid(row=0, column=2, sticky="w", padx=(0, 8))
        ttk.Entry(year_frame, textvariable=self.later_year, width=9).grid(row=0, column=3, sticky="w")

        self._section(left, "2  ONLINE IMAGE FETCHING (GOOGLE EARTH ENGINE)")
        ttk.Label(left, text="Google Cloud Project ID (Earth Engine enabled)", foreground=MUTED, background=PANEL, wraplength=350).pack(anchor="w", pady=(0, 2))
        ttk.Entry(left, textvariable=self.gee_project, width=42).pack(fill="x", pady=(0, 6))
        ttk.Label(left, text="Earlier-year date window (start inclusive, end exclusive)", foreground=MUTED, background=PANEL, wraplength=350).pack(anchor="w", pady=(0, 2))
        date_frame1 = ttk.Frame(left, style="P.TFrame")
        date_frame1.pack(fill="x", pady=(0, 6))
        ttk.Entry(date_frame1, textvariable=self.earlier_start, width=13).pack(side="left", fill="x", expand=True)
        ttk.Label(date_frame1, text="to", foreground=MUTED, background=PANEL).pack(side="left", padx=4)
        ttk.Entry(date_frame1, textvariable=self.earlier_end, width=13).pack(side="left", fill="x", expand=True)
        ttk.Label(left, text="Later-year date window (start inclusive, end exclusive)", foreground=MUTED, background=PANEL, wraplength=350).pack(anchor="w", pady=(0, 2))
        date_frame2 = ttk.Frame(left, style="P.TFrame")
        date_frame2.pack(fill="x", pady=(0, 6))
        ttk.Entry(date_frame2, textvariable=self.later_start, width=13).pack(side="left", fill="x", expand=True)
        ttk.Label(date_frame2, text="to", foreground=MUTED, background=PANEL).pack(side="left", padx=4)
        ttk.Entry(date_frame2, textvariable=self.later_end, width=13).pack(side="left", fill="x", expand=True)
        ttk.Label(left, text="Maximum scene cloud metadata (%) — SCL still masks clouds per pixel", foreground=MUTED, background=PANEL, wraplength=350).pack(anchor="w", pady=(0, 2))
        ttk.Entry(left, textvariable=self.max_scene_cloud, width=10).pack(anchor="w", pady=(0, 7))
        self.gee_auth_button = ttk.Button(left, text="AUTHENTICATE / TEST GEE", command=self.authenticate_gee)
        self.gee_auth_button.pack(fill="x", pady=(0, 5))
        self.gee_fetch_button = ttk.Button(left, text="FETCH CLOUD-MASKED COMPOSITES", command=self.fetch_gee)
        self.gee_fetch_button.pack(fill="x", pady=(0, 8))
        ttk.Label(left, text="Fetches independent date-window products: median B3/B8, valid-observation count, per-pixel water frequency, and maximum observed water extent from per-scene NDWI after SCL cloud/shadow masking.", foreground=MUTED, background=PANEL, wraplength=350, font=("Segoe UI", 8)).pack(anchor="w", pady=(0, 10))

        self._section(left, "3  EARLIER-YEAR SENTINEL-2 INPUTS")
        self._file_row(left, "B3 Green (10 m)", self.old_b3, lambda: self._browse_raster(self.old_b3, "Select earlier-year B3 Green raster"))
        self._file_row(left, "B8 NIR (10 m)", self.old_b8, lambda: self._browse_raster(self.old_b8, "Select earlier-year B8 NIR raster"))
        ttk.Button(left, text="FIND B03 + B08 IN SAFE/FOLDER", command=lambda: self._find_bands(self.old_b3, self.old_b8, "earlier year")).pack(fill="x", pady=(0, 10))

        self._section(left, "4  LATER-YEAR SENTINEL-2 INPUTS")
        self._file_row(left, "B3 Green (10 m)", self.new_b3, lambda: self._browse_raster(self.new_b3, "Select later-year B3 Green raster"))
        self._file_row(left, "B8 NIR (10 m)", self.new_b8, lambda: self._browse_raster(self.new_b8, "Select later-year B8 NIR raster"))
        ttk.Button(left, text="FIND B03 + B08 IN SAFE/FOLDER", command=lambda: self._find_bands(self.new_b3, self.new_b8, "later year")).pack(fill="x", pady=(0, 10))

        self._section(left, "5  STUDY BOUNDARY AND OUTPUT")
        self._file_row(left, "District/study boundary (.shp)", self.boundary, self._browse_boundary)
        self._file_row(left, "Output folder", self.output, self._browse_output)

        self._section(left, "6  PROCESSING PARAMETERS")
        ttk.Label(left, text="Water detection method", foreground=MUTED, background=PANEL).pack(anchor="w", pady=(2, 2))
        ttk.Combobox(left, textvariable=self.detection_method, state="readonly", values=[
            "Maximum observed extent (GEE)", "Median composite NDWI"
        ]).pack(fill="x", pady=(0, 7))
        self._param_row(left, "NDWI threshold (water if NDWI >)", self.threshold)
        self._param_row(left, "Minimum connected pixels", self.min_pixels, enabled=False)
        ttk.Label(left, text="NDWI = (B3 − B8) / (B3 + B8)\nMaximum extent combines water detections across clear scenes in each separate window. 10 m pixels • 8-connectivity • minimum 25 pixels (0.25 ha).", foreground=MUTED, background=PANEL, wraplength=350, font=("Segoe UI", 8)).pack(anchor="w", pady=(0, 8))

        self.masked_confirm = tk.BooleanVar(value=False)
        tk.Checkbutton(
            left,
            text="Inputs are already cloud/shadow masked; invalid pixels are NoData.",
            variable=self.masked_confirm,
            wraplength=350,
            justify="left",
            anchor="w",
            background=PANEL,
            foreground=WHITE,
            activebackground=PANEL,
            activeforeground=WHITE,
            selectcolor=PANEL2,
            font=("Segoe UI", 9),
            relief="flat",
            highlightthickness=0,
        ).pack(anchor="w", pady=(0, 7))
        ttk.Label(left, text="Do not use untouched SAFE bands unless cloud/shadow pixels have been masked beforehand.", foreground=RED, background=PANEL, wraplength=350, font=("Segoe UI", 8, "bold")).pack(anchor="w", pady=(0, 10))

        self.run_button = ttk.Button(left, text="RUN OFFLINE ANALYSIS", command=self.run)
        self.run_button.pack(fill="x", pady=(4, 5))
        ttk.Button(left, text="CLEAR INPUTS", command=self.clear).pack(fill="x")
        ttk.Button(left, text="OPEN OUTPUT FOLDER", command=self.open_output).pack(fill="x", pady=(5, 0))

        self.status = ttk.Label(right, text="READY — select inputs and run an analysis.", style="H.TLabel", wraplength=780)
        self.status.pack(anchor="w", fill="x", pady=(0, 8))
        self.progress = ttk.Progressbar(right, mode="determinate", maximum=100)
        self.progress.pack(fill="x", pady=(0, 10))

        tabs = ttk.Notebook(right)
        tabs.pack(fill="both", expand=True)
        self.dashboard = ttk.Frame(tabs, padding=12)
        activity = ttk.Frame(tabs, padding=10)
        tabs.add(self.dashboard, text="Results dashboard")
        tabs.add(activity, text="Activity log")
        self.results_tab = tabs
        self._build_dashboard(self.dashboard)

        self.log = tk.Text(activity, bg="#F8FBFF", fg="#233D55", insertbackground="#233D55",
                           relief="flat", font=("Consolas", 9), wrap="word", padx=10, pady=10,
                           highlightthickness=1, highlightbackground=BORDER)
        self.log.pack(fill="both", expand=True)
        self.log.insert("end", "READY\n\nThis application supports local Sentinel-2 B3/B8 files and optional online image fetching through Google Earth Engine.\nGEE mode requires internet access, an Earth Engine-enabled Google Cloud project, and one-time authentication.\n\nFor online mode, select a study boundary/output folder, set the two independent date windows, then click FETCH CLOUD-MASKED COMPOSITES. GEE produces independent products for each date window.\nFor local mode, select B3/B8 rasters and choose Median composite NDWI.\nMaximum observed extent is the union of per-scene water detections, not a legal pond boundary; seasonal water-level changes and false detections still need review.\n")
        self.log.configure(state="disabled")
        ttk.Label(right, text="Outputs: NDWI rasters, water masks, pixel-change raster, waterbody GeoPackage, CSV/Excel statistics, PNG/PDF maps and run metadata.", style="LogFoot.TLabel", wraplength=780).pack(anchor="w", pady=(9, 0))

    def _build_dashboard(self, parent):
        header = ttk.Frame(parent, style="P.TFrame")
        header.pack(fill="x", pady=(0, 10))
        ttk.Label(header, text="Analysis overview", style="H.TLabel").pack(side="left")
        ttk.Label(header, text="Urban ponds & lakes • Sentinel-2", foreground=MUTED, background=PANEL).pack(side="right")

        cards = ttk.Frame(parent, style="P.TFrame")
        cards.pack(fill="x", pady=(0, 10))
        for title, key in [("Earlier water area", "earlier"), ("Later water area", "later"),
                           ("Net area change", "net"), ("Potential water loss", "loss")]:
            card = tk.Frame(cards, bg=ACCENT_PALE, highlightbackground=BORDER, highlightthickness=1, padx=10, pady=9)
            card.pack(side="left", fill="both", expand=True, padx=(0, 7))
            tk.Label(card, text=title.upper(), bg=ACCENT_PALE, fg=MUTED, font=("Segoe UI", 8, "bold")).pack(anchor="w")
            value = tk.Label(card, text="—", bg=ACCENT_PALE, fg="#245F91", font=("Segoe UI", 16, "bold"))
            value.pack(anchor="w", pady=(5, 0))
            self.kpi_values[key] = value

        body = ttk.Frame(parent, style="P.TFrame")
        body.pack(fill="both", expand=True)
        left = ttk.Frame(body, style="P.TFrame")
        left.pack(side="left", fill="both", expand=True, padx=(0, 10))
        right = ttk.Frame(body, style="P.TFrame")
        right.pack(side="left", fill="both", expand=True)
        ttk.Label(left, text="Water area comparison", style="H.TLabel").pack(anchor="w", pady=(0, 5))
        self.chart_canvas = tk.Canvas(left, height=190, bg="#FFFFFF", highlightthickness=1, highlightbackground=BORDER)
        self.chart_canvas.pack(fill="x")
        ttk.Label(left, text="Change map preview", style="H.TLabel").pack(anchor="w", pady=(10, 5))
        self.map_preview = tk.Label(left, text="Run an analysis to display the change map here.", bg="#F8FBFF", fg=MUTED, anchor="center", height=8)
        self.map_preview.pack(fill="both", expand=True)
        ttk.Label(right, text="Key statistics", style="H.TLabel").pack(anchor="w", pady=(0, 5))
        self.stats_tree = ttk.Treeview(right, columns=("metric", "value"), show="headings", height=12)
        self.stats_tree.heading("metric", text="Metric")
        self.stats_tree.heading("value", text="Value")
        self.stats_tree.column("metric", width=175, anchor="w")
        self.stats_tree.column("value", width=145, anchor="e")
        self.stats_tree.pack(fill="both", expand=True)
        actions = ttk.Frame(right, style="P.TFrame")
        actions.pack(fill="x", pady=(8, 0))
        ttk.Button(actions, text="Open output folder", command=self.open_output).pack(side="left")
        ttk.Button(actions, text="Refresh dashboard", command=self.refresh_dashboard).pack(side="right")

    def refresh_dashboard(self):
        out_text = self.output.get().strip()
        if not out_text:
            return
        out = Path(out_text)
        summary_path = out / "06_Statistics" / "Summary.json"
        if summary_path.exists():
            try:
                import json
                with summary_path.open(encoding="utf-8") as f:
                    self._display_results({"summary": json.load(f), "output_dir": str(out)})
            except Exception as exc:
                self.write("Dashboard refresh failed: " + str(exc))
        else:
            self.write("No Summary.json found in the selected output folder yet.")

    def _display_results(self, result):
        summary = result.get("summary", {})
        old = float(summary.get("water_area_earlier_ha", 0) or 0)
        new = float(summary.get("water_area_later_ha", 0) or 0)
        net = float(summary.get("net_area_change_ha", new-old) or 0)
        loss = float(summary.get("potential_loss_m2", 0) or 0) / 10000.0
        for key, value in [("earlier", f"{old:,.2f} ha"), ("later", f"{new:,.2f} ha"),
                           ("net", f"{net:+,.2f} ha"), ("loss", f"{loss:,.2f} ha")]:
            color = RED if ((key == "net" and net < 0) or (key == "loss" and loss > 0)) else "#245F91"
            self.kpi_values[key].configure(text=value, fg=color)
        for item in self.stats_tree.get_children():
            self.stats_tree.delete(item)
        rows = [
            ("Earlier waterbodies", summary.get("earlier_waterbody_count", 0)),
            ("Later waterbodies", summary.get("later_waterbody_count", 0)),
            ("Persistent water", f"{float(summary.get('persistent_water_m2', 0) or 0)/10000:.2f} ha"),
            ("Potential water gain", f"{float(summary.get('gain_m2', 0) or 0)/10000:.2f} ha"),
            ("Potential water loss", f"{loss:.2f} ha"),
            ("Earlier valid pixels", f"{int(summary.get('valid_pixels_earlier', 0) or 0):,}"),
            ("Later valid pixels", f"{int(summary.get('valid_pixels_later', 0) or 0):,}"),
            ("Detection method", summary.get("detection_method", "—")),
            ("NDWI threshold", summary.get("ndwi_threshold", "—")),
            ("Pixel size", f"{summary.get('pixel_size_m', 10)} m"),
            ("CRS", summary.get("target_crs", "—")),
        ]
        for metric, value in rows:
            self.stats_tree.insert("", "end", values=(metric, value))
        self._draw_area_chart(old, new, summary.get("earlier_year", "Earlier"), summary.get("later_year", "Later"))
        map_path = Path(result.get("output_dir", "")) / "07_Maps" / f"WaterChange_{summary.get('earlier_year')}_{summary.get('later_year')}.png"
        if map_path.exists():
            try:
                image = tk.PhotoImage(file=str(map_path))
                max_w, max_h = 620, 260
                factor = max(1, (image.width()+max_w-1)//max_w, (image.height()+max_h-1)//max_h)
                if factor > 1:
                    image = image.subsample(factor, factor)
                self.map_image = image
                self.map_preview.configure(image=self.map_image, text="", height=0)
            except Exception as exc:
                self.map_preview.configure(image="", text=f"Map preview unavailable: {exc}")
        self.results_tab.select(self.dashboard)

    def _draw_area_chart(self, old, new, old_year, new_year):
        canvas = self.chart_canvas
        canvas.delete("all")
        canvas.update_idletasks()
        width, height = max(canvas.winfo_width(), 400), 190
        canvas.create_text(14, 12, text="Mapped water area (hectares)", anchor="w", fill="#60778D", font=("Segoe UI", 9, "bold"))
        maximum = max(float(old), float(new), 0.01) * 1.18
        baseline, bar_width = height - 30, 65
        for x, value, label, color in [(width*0.32, old, str(old_year), "#79B4E2"), (width*0.68, new, str(new_year), "#3E83BD")]:
            bar_height = max(2, (float(value)/maximum)*(height-65))
            canvas.create_rectangle(x-bar_width/2, baseline-bar_height, x+bar_width/2, baseline, fill=color, outline="")
            canvas.create_text(x, baseline-bar_height-10, text=f"{float(value):.2f}", fill="#29465F", font=("Segoe UI", 9, "bold"))
            canvas.create_text(x, baseline+13, text=label, fill="#60778D", font=("Segoe UI", 9))

    def _section(self, parent, text):
        ttk.Label(parent, text=text, style="H.TLabel").pack(anchor="w", pady=(5, 8))

    def _file_row(self, parent, label, variable, command):
        ttk.Label(parent, text=label).pack(anchor="w", pady=(2, 2))
        frame = ttk.Frame(parent, style="P.TFrame")
        frame.pack(fill="x", pady=(0, 7))
        ttk.Entry(frame, textvariable=variable, width=36).pack(side="left", fill="x", expand=True)
        ttk.Button(frame, text="Browse", command=command).pack(side="right", padx=(5, 0))

    def _param_row(self, parent, label, variable, enabled=True):
        ttk.Label(parent, text=label).pack(anchor="w", pady=(2, 2))
        ttk.Entry(parent, textvariable=variable, width=12, state="normal" if enabled else "disabled").pack(anchor="w", pady=(0, 7))

    def _browse_raster(self, variable, title):
        path = filedialog.askopenfilename(title=title, filetypes=[("Raster files", "*.tif *.tiff *.jp2"), ("All files", "*.*")])
        if path:
            variable.set(path)

    def _find_bands(self, b3_var, b8_var, label):
        folder = filedialog.askdirectory(title=f"Select Sentinel-2 SAFE product or folder for {label}")
        if not folder:
            return
        try:
            b3 = discover_band(folder, "B03")
            b8 = discover_band(folder, "B08")
            b3_var.set(b3)
            b8_var.set(b8)
            self.write(f"✓ Found {label} bands:\n  B03: {b3}\n  B08: {b8}")
            messagebox.showinfo("Bands found", f"B3 and B8 found for {label}.\n\nPlease verify that both inputs are the intended pre-masked rasters.")
        except Exception as exc:
            self.write("ERROR finding bands: " + str(exc))
            messagebox.showerror("Band discovery failed", str(exc))

    def _browse_boundary(self):
        path = filedialog.askopenfilename(title="Select district/study boundary shapefile", filetypes=[("Shapefile", "*.shp")])
        if path:
            self.boundary.set(path)

    def _browse_output(self):
        path = filedialog.askdirectory(title="Select output folder")
        if path:
            self.output.set(path)

    def write(self, message):
        if not self.log:
            return
        self.log.configure(state="normal")
        self.log.insert("end", message + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def clear(self):
        for var in [self.old_b3, self.old_b8, self.new_b3, self.new_b8, self.old_water_mask, self.new_water_mask, self.boundary, self.output]:
            var.set("")
        self.progress.configure(value=0)
        self.status.configure(text="Inputs cleared. Select files to begin.")
        self.write("Inputs cleared.")

    def open_output(self):
        path = self.output.get().strip()
        if path and Path(path).exists():
            try:
                os.startfile(path)
            except Exception:
                messagebox.showinfo("Output folder", path)
        else:
            messagebox.showinfo("Output folder", "Choose an output folder first.")

    def _set_gee_buttons(self, enabled):
        state = "normal" if enabled else "disabled"
        if self.gee_auth_button:
            self.gee_auth_button.configure(state=state)
        if self.gee_fetch_button:
            self.gee_fetch_button.configure(state=state)
        if self.run_button:
            self.run_button.configure(state=state)

    def authenticate_gee(self):
        if self.gee_worker and self.gee_worker.is_alive():
            messagebox.showwarning("GEE busy", "An Earth Engine operation is already running.")
            return
        project = self.gee_project.get().strip()
        if not project:
            messagebox.showwarning("Project ID required", "Enter your Google Cloud Project ID. It must be registered for Earth Engine and have the Earth Engine API enabled.")
            return
        self._set_gee_buttons(False)
        self.status.configure(text="Connecting to Google Earth Engine…")
        self.write("\nChecking Google Earth Engine authentication/project access…")
        self.gee_worker = threading.Thread(target=self._worker_gee_auth, args=(project,), daemon=True)
        self.gee_worker.start()

    def _worker_gee_auth(self, project):
        try:
            from waterbody_app.gee_fetch import test_connection
            message = test_connection(project)
            self.events.put(("gee_auth_success", message))
        except Exception:
            self.events.put(("gee_error", traceback.format_exc()))

    def fetch_gee(self):
        if self.gee_worker and self.gee_worker.is_alive():
            messagebox.showwarning("GEE busy", "An Earth Engine operation is already running.")
            return
        try:
            project = self.gee_project.get().strip()
            if not project:
                raise ValueError("Enter your Google Cloud Project ID first.")
            boundary = self.boundary.get().strip()
            if not boundary or not Path(boundary).is_file():
                raise ValueError("Select a valid study boundary shapefile first.")
            output = self.output.get().strip()
            if not output:
                raise ValueError("Select an output folder first.")
            max_cloud = float(self.max_scene_cloud.get())
            if not 0 <= max_cloud <= 100:
                raise ValueError("Maximum scene cloud metadata must be between 0 and 100.")
            from datetime import date
            specs = [
                (self.earlier_start.get().strip(), self.earlier_end.get().strip(), "earlier"),
                (self.later_start.get().strip(), self.later_end.get().strip(), "later"),
            ]
            for start_text, end_text, label in specs:
                start = date.fromisoformat(start_text)
                end = date.fromisoformat(end_text)
                if end <= start:
                    raise ValueError(f"{label.title()} date window: end date must be after start date.")
            if date.fromisoformat(specs[0][0]).year != int(self.earlier_year.get()):
                raise ValueError("Earlier date-window start year must match the Earlier year field.")
            if date.fromisoformat(specs[1][0]).year != int(self.later_year.get()):
                raise ValueError("Later date-window start year must match the Later year field.")
            config = {
                "project_id": project, "boundary": boundary, "output_dir": output,
                "earlier_start": specs[0][0], "earlier_end": specs[0][1],
                "later_start": specs[1][0], "later_end": specs[1][1],
                "max_scene_cloud": max_cloud,
                "water_threshold": float(self.threshold.get()),
            }
        except Exception as exc:
            messagebox.showwarning("GEE input validation", str(exc))
            return
        self._set_gee_buttons(False)
        self.progress.configure(value=0)
        self.status.configure(text="Starting Google Earth Engine composite fetch…")
        self.write("\nStarting GEE fetch. Cloud-masked composite downloads will be saved locally.")
        self.gee_worker = threading.Thread(target=self._worker_gee_fetch, args=(config,), daemon=True)
        self.gee_worker.start()

    def _worker_gee_fetch(self, config):
        try:
            from waterbody_app.gee_fetch import fetch_composites
            result = fetch_composites(
                config["project_id"], config["boundary"], config["output_dir"],
                config["earlier_start"], config["earlier_end"],
                config["later_start"], config["later_end"],
                config["max_scene_cloud"],
                progress=lambda pct, msg: self.events.put(("progress", pct, msg)),
                authenticate_if_needed=True,
                water_threshold=config["water_threshold"],
            )
            self.events.put(("gee_fetch_success", result))
        except Exception:
            self.events.put(("gee_error", traceback.format_exc()))

    def run(self):
        if self.gee_worker and self.gee_worker.is_alive():
            messagebox.showwarning("GEE busy", "Wait for Earth Engine fetching/authentication to finish first.")
            return
        if self.worker and self.worker.is_alive():
            messagebox.showwarning("Already running", "An analysis is already running.")
            return
        try:
            y0 = self.earlier_year.get().strip()
            y1 = self.later_year.get().strip()
            if not (y0.isdigit() and len(y0) == 4 and y1.isdigit() and len(y1) == 4):
                raise ValueError("Enter both comparison years as four digits, for example 2017 and 2024.")
            if y0 == y1:
                raise ValueError("Earlier and later years must be different.")
            paths = [self.old_b3.get().strip(), self.old_b8.get().strip(), self.new_b3.get().strip(), self.new_b8.get().strip()]
            if not all(paths):
                raise ValueError("Select B3 and B8 rasters for both years, or use FIND B03 + B08 IN SAFE/FOLDER.")
            for p in paths:
                if not Path(p).is_file():
                    raise FileNotFoundError(f"Raster file not found:\n{p}")
            boundary = self.boundary.get().strip()
            if not boundary or not Path(boundary).is_file():
                raise ValueError("Select a valid district/study boundary shapefile (.shp).")
            output = self.output.get().strip()
            if not output:
                raise ValueError("Select an output folder.")
            threshold = float(self.threshold.get())
            if not -1 <= threshold <= 1:
                raise ValueError("NDWI threshold must be between -1 and 1.")
            use_extent = self.detection_method.get() == "Maximum observed extent (GEE)"
            if use_extent and not (self.old_water_mask.get().strip() and self.new_water_mask.get().strip()):
                raise ValueError("Maximum observed extent requires the per-scene water masks created by FETCH CLOUD-MASKED COMPOSITES. Fetch the GEE composites first, or select Median composite NDWI for local B3/B8 files.")
            if use_extent and (not Path(self.old_water_mask.get()).is_file() or not Path(self.new_water_mask.get()).is_file()):
                raise FileNotFoundError("A GEE maximum-extent water mask is missing. Fetch the two date windows again.")
            if int(self.min_pixels.get()) != 25:
                raise ValueError("The approved minimum waterbody size is fixed at 25 pixels.")
            if not self.masked_confirm.get():
                raise ValueError("Confirm that B3/B8 inputs are already cloud/shadow masked and invalid pixels are NoData.")
            config = {
                "earlier_year": y0, "later_year": y1,
                "earlier_b3": paths[0], "earlier_b8": paths[1],
                "later_b3": paths[2], "later_b8": paths[3],
                "boundary": boundary, "threshold": threshold, "min_pixels": 25,
                "earlier_water_mask": self.old_water_mask.get().strip() if use_extent else None,
                "later_water_mask": self.new_water_mask.get().strip() if use_extent else None,
                "output_dir": output,
            }
        except Exception as exc:
            messagebox.showwarning("Input validation", str(exc))
            return

        self.run_button.configure(state="disabled")
        self.progress.configure(value=0)
        self.status.configure(text="Starting analysis…")
        self.write("\nStarting analysis…")
        self.worker = threading.Thread(target=self._worker_run, args=(config,), daemon=True)
        self.worker.start()

    def _worker_run(self, config):
        try:
            result = run_analysis(config, lambda pct, msg: self.events.put(("progress", pct, msg)))
            self.events.put(("success", result))
        except Exception:
            self.events.put(("error", traceback.format_exc()))

    def _poll_events(self):
        try:
            while True:
                event = self.events.get_nowait()
                if event[0] == "progress":
                    _, pct, message = event
                    self.progress.configure(value=pct)
                    self.status.configure(text=f"[{pct:3d}%] {message}")
                    self.write(f"[{pct:3d}%] {message}")
                elif event[0] == "success":
                    result = event[1]
                    summary = result["summary"]
                    self.run_button.configure(state="normal")
                    self.status.configure(text="Analysis completed successfully.")
                    self.write("\nSUCCESS — analysis complete.")
                    self.write(f"Earlier water area: {summary['water_area_earlier_ha']:.3f} ha")
                    self.write(f"Later water area:   {summary['water_area_later_ha']:.3f} ha")
                    self.write(f"Net change:         {summary['net_area_change_ha']:+.3f} ha")
                    self.write(f"Earlier waterbodies: {summary['earlier_waterbody_count']}")
                    self.write(f"Later waterbodies:   {summary['later_waterbody_count']}")
                    self.write("OUTPUT: " + result["output_dir"])
                    self._display_results(result)
                    messagebox.showinfo("Analysis complete", f"Analysis complete.\n\nResults dashboard updated.\nResults saved to:\n{result['output_dir']}")
                elif event[0] == "error":
                    details = event[1]
                    self.run_button.configure(state="normal")
                    self.status.configure(text="Analysis failed. See the full traceback in the activity log.")
                    self.write("ERROR — full traceback:\n" + details)
                    messagebox.showerror("Analysis failed", "Analysis failed. The complete traceback is in the activity log.")
                elif event[0] == "gee_auth_success":
                    self._set_gee_buttons(True)
                    self.status.configure(text="Google Earth Engine connection is ready.")
                    self.write("\n✓ " + event[1])
                    messagebox.showinfo("GEE connected", event[1])
                elif event[0] == "gee_fetch_success":
                    result = event[1]
                    self._set_gee_buttons(True)
                    self.old_b3.set(result["earlier"]["b3"])
                    self.old_b8.set(result["earlier"]["b8"])
                    self.new_b3.set(result["later"]["b3"])
                    self.new_b8.set(result["later"]["b8"])
                    self.old_water_mask.set(result["earlier"]["max_water"])
                    self.new_water_mask.set(result["later"]["max_water"])
                    self.earlier_year.set(str(result["earlier"]["year"]))
                    self.later_year.set(str(result["later"]["year"]))
                    self.masked_confirm.set(True)
                    self.progress.configure(value=100)
                    self.status.configure(text="GEE products downloaded. Run analysis using the selected water detection method.")
                    self.write("\n✓ GEE composites fetched successfully.")
                    self.write(f"Earlier ({result['earlier']['year']}): {result['earlier']['scene_count']} scenes; {len(result['earlier']['dates'])} acquisition dates")
                    self.write(f"  B03: {result['earlier']['b3']}")
                    self.write(f"  B08: {result['earlier']['b8']}")
                    self.write(f"  Valid observations: {result['earlier']['valid_obs']}")
                    self.write(f"  Maximum observed water extent: {result['earlier']['max_water']}")
                    self.write(f"  Water frequency (per-mille): {result['earlier']['water_frequency']}")
                    self.write(f"Later ({result['later']['year']}): {result['later']['scene_count']} scenes; {len(result['later']['dates'])} acquisition dates")
                    self.write(f"  B03: {result['later']['b3']}")
                    self.write(f"  B08: {result['later']['b8']}")
                    self.write(f"  Valid observations: {result['later']['valid_obs']}")
                    self.write(f"  Maximum observed water extent: {result['later']['max_water']}")
                    self.write(f"  Water frequency (per-mille): {result['later']['water_frequency']}")
                    self.write("Metadata: " + result["metadata"])
                    messagebox.showinfo("GEE fetch complete", "Independent date-window products are saved locally, including maximum observed water extent masks.\n\nThe GUI is set to Maximum observed extent (GEE) by default. Click RUN OFFLINE ANALYSIS to compare the two footprints. You can select Median composite NDWI to compare the conventional approach.")
                elif event[0] == "gee_error":
                    details = event[1]
                    self._set_gee_buttons(True)
                    self.status.configure(text="GEE operation failed. See full traceback in activity log.")
                    self.write("GEE ERROR — full traceback:\n" + details)
                    messagebox.showerror("GEE operation failed", "The Earth Engine operation failed. See the full traceback in the activity log.")
        except queue.Empty:
            pass
        self.after(100, self._poll_events)


if __name__ == "__main__":
    App().mainloop()
