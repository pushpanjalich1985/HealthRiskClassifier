# ─────────────────────────────────────────────
#  ui.py
#  Desktop GUI for the Health Risk Classifier
#  Built with Tkinter — no extra installs needed
#
#  Run with:
#      python src/ui.py
# ─────────────────────────────────────────────

import tkinter as tk
from tkinter import ttk
import sys
import os
import joblib
import numpy as np

# ── Tell Python where src/ is so imports work ────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))


# ── Colours — Dark Mode ───────────────────────────────────────────────────────
BG           = "#0F1117"
CARD_BG      = "#1A1D27"
BORDER       = "#2E3144"
TEXT_DARK    = "#F1F3F9"
TEXT_MUTED   = "#8B92A9"
ACCENT_BLUE  = "#4F8EF7"
HEADER_BG    = "#161B2E"

LOW_BG       = "#0D2318"
LOW_FG       = "#4ADE80"
LOW_BORDER   = "#166534"

MED_BG       = "#231A08"
MED_FG       = "#FCD34D"
MED_BORDER   = "#854D0E"

HIGH_BG      = "#200D0D"
HIGH_FG      = "#F87171"
HIGH_BORDER  = "#7F1D1D"

BAR_COLOURS  = [
    "#4F8EF7", "#A78BFA", "#34D399",
    "#FBBF24", "#F87171", "#22D3EE",
    "#A3E635", "#94A3B8"
]


# ─────────────────────────────────────────────────────────────────────────────
class HealthRiskApp:

    def __init__(self, root):
        self.root = root
        self._setup_window()
        self._load_model()
        self._build_ui()
        self._predict()


    # ── Window setup ──────────────────────────────────────────────────────────
    def _setup_window(self):
        self.root.title("Health Risk Classifier")
        self.root.geometry("900x720")
        self.root.resizable(True, True)
        self.root.configure(bg=BG)
        self.root.minsize(800, 650)

        self.root.update_idletasks()
        w = self.root.winfo_width()
        h = self.root.winfo_height()
        x = (self.root.winfo_screenwidth()  // 2) - (w // 2)
        y = (self.root.winfo_screenheight() // 2) - (h // 2)
        self.root.geometry(f"+{x}+{y}")


    # ── Load model ────────────────────────────────────────────────────────────
    def _load_model(self):
        base_dir   = os.path.dirname(os.path.dirname(__file__))
        model_path = os.path.join(base_dir, "output", "decision_tree.pkl")

        if not os.path.exists(model_path):
            self.model        = None
            self.model_status = "Model not found — run main.py first"
        else:
            self.model        = joblib.load(model_path)
            self.model_status = "Model loaded successfully"

        self.label_map = {0: "Low", 1: "Medium", 2: "High"}


    # ── Build UI ──────────────────────────────────────────────────────────────
    def _build_ui(self):

        # Header
        header = tk.Frame(self.root, bg=HEADER_BG, height=56)
        header.pack(fill="x")
        header.pack_propagate(False)

        tk.Label(
            header,
            text="Health Risk Classifier",
            bg=HEADER_BG, fg=TEXT_DARK,
            font=("Helvetica", 16, "bold")
        ).pack(side="left", padx=20, pady=14)

        status_colour = "#4ADE80" if self.model else "#F87171"
        tk.Label(
            header,
            text=f"● {self.model_status}",
            bg=HEADER_BG, fg=status_colour,
            font=("Helvetica", 10)
        ).pack(side="right", padx=20)

        # Scrollable canvas
        canvas_frame = tk.Frame(self.root, bg=BG)
        canvas_frame.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(canvas_frame, bg=BG, highlightthickness=0)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "Dark.Vertical.TScrollbar",
            background=BORDER,
            troughcolor=BG,
            bordercolor=BG,
            arrowcolor=TEXT_MUTED,
            darkcolor=BORDER,
            lightcolor=BORDER
        )

        scrollbar = ttk.Scrollbar(
            canvas_frame, orient="vertical",
            command=self.canvas.yview,
            style="Dark.Vertical.TScrollbar"
        )
        self.canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.inner = tk.Frame(self.canvas, bg=BG)
        self.canvas_window = self.canvas.create_window(
            (0, 0), window=self.inner, anchor="nw"
        )

        self.inner.bind("<Configure>", self._on_frame_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.bind_all(
            "<MouseWheel>",
            lambda e: self.canvas.yview_scroll(int(-1*(e.delta/120)), "units")
        )

        # Sections
        self._build_inputs()
        self._build_result_panel()
        self._build_flags_panel()
        self._build_importance_panel()
        self._build_stats_panel()

        tk.Label(
            self.inner,
            text="Decision Tree Classifier  ·  Scikit-learn  ·  Trained on 1000 synthetic patients",
            bg=BG, fg=TEXT_MUTED,
            font=("Helvetica", 9)
        ).pack(pady=(4, 16))


    # ── Input fields panel ────────────────────────────────────────────────────
    def _build_inputs(self):
        """Compact entry boxes in a 4-column grid."""

        outer = self._make_card(self.inner, "Patient Inputs")

        field_configs = [
            ("Age",               "age",   18,  90,  54,   1  ),
            ("BMI",               "bmi",   10,  50,  27.5, 0.5),
            ("Blood Pressure",    "bp",    80,  200, 120,  1  ),
            ("Cholesterol",       "chol",  100, 400, 200,  1  ),
            ("Blood Sugar",       "sugar", 70,  300, 100,  1  ),
            ("Physical Activity", "act",   0,   20,  3.0,  0.5),
            ("Sleep Hours",       "sleep", 3,   10,  7.0,  0.5),
        ]

        units = {
            "age":   "yrs",
            "bmi":   "",
            "bp":    "mmHg",
            "chol":  "mg/dL",
            "sugar": "mg/dL",
            "act":   "h/wk",
            "sleep": "hrs",
        }

        self.vars    = {}
        self.entries = {}

        # 4-column grid
        grid = tk.Frame(outer, bg=CARD_BG)
        grid.pack(fill="x", padx=4, pady=4)

        for col in range(4):
            grid.columnconfigure(col, weight=1)

        for i, (label, key, lo, hi, default, res) in enumerate(field_configs):
            row = i // 4
            col = i % 4

            cell = tk.Frame(grid, bg=CARD_BG, padx=6, pady=6)
            cell.grid(row=row, column=col, sticky="ew", padx=4, pady=4)

            # Label + unit row
            top = tk.Frame(cell, bg=CARD_BG)
            top.pack(fill="x")

            tk.Label(
                top,
                text=label,
                bg=CARD_BG, fg=TEXT_MUTED,
                font=("Helvetica", 9)
            ).pack(side="left")

            unit = units[key]
            if unit:
                tk.Label(
                    top,
                    text=unit,
                    bg=CARD_BG, fg=BORDER,
                    font=("Helvetica", 8)
                ).pack(side="right")

            # Entry box with glowing border frame
            var = tk.StringVar(value=str(default))

            entry_frame = tk.Frame(
                cell,
                bg=BORDER,
                highlightbackground=ACCENT_BLUE,
                highlightthickness=0,
                padx=1, pady=1
            )
            entry_frame.pack(fill="x", pady=(4, 0))

            entry = tk.Entry(
                entry_frame,
                textvariable=var,
                bg="#0D1120",
                fg=ACCENT_BLUE,
                insertbackground=ACCENT_BLUE,
                font=("Helvetica", 13, "bold"),
                relief="flat",
                bd=6,
                justify="center",
                width=8
            )
            entry.pack(fill="x")

            # Glow border on focus
            entry.bind("<FocusIn>",
                       lambda e, f=entry_frame: f.config(
                           highlightbackground=ACCENT_BLUE,
                           highlightthickness=2))
            entry.bind("<FocusOut>",
                       lambda e, f=entry_frame: f.config(
                           highlightthickness=0))

            # Validate + predict on every keystroke
            entry.bind("<KeyRelease>",
                       lambda e, k=key, lo=lo, hi=hi, r=res:
                           self._on_entry(k, lo, hi, r))

            self.vars[key]    = var
            self.entries[key] = entry

        # ── Smoking toggle — 8th cell ─────────────────────────────────────────
        smoke_cell = tk.Frame(grid, bg=CARD_BG, padx=6, pady=6)
        smoke_cell.grid(row=1, column=3, sticky="ew", padx=4, pady=4)

        tk.Label(
            smoke_cell,
            text="Smoking",
            bg=CARD_BG, fg=TEXT_MUTED,
            font=("Helvetica", 9)
        ).pack(anchor="w")

        btn_row = tk.Frame(smoke_cell, bg=CARD_BG)
        btn_row.pack(fill="x", pady=(4, 0))

        self.smoking_var = tk.IntVar(value=0)

        self.smoke_no_btn = tk.Button(
            btn_row, text="No",
            bg=ACCENT_BLUE, fg=BG,
            font=("Helvetica", 11, "bold"),
            relief="flat", bd=0,
            padx=10, pady=6,
            cursor="hand2",
            command=lambda: self._set_smoking(0)
        )
        self.smoke_no_btn.pack(side="left", fill="x", expand=True, padx=(0, 3))

        self.smoke_yes_btn = tk.Button(
            btn_row, text="Yes",
            bg="#0D1120", fg=TEXT_MUTED,
            font=("Helvetica", 11),
            relief="flat", bd=0,
            padx=10, pady=6,
            cursor="hand2",
            command=lambda: self._set_smoking(1)
        )
        self.smoke_yes_btn.pack(side="left", fill="x", expand=True, padx=(3, 0))

        # Range hints
        tk.Label(
            outer,
            text="Valid ranges:  Age 18–90  ·  BMI 10–50  ·  BP 80–200  ·  "
                 "Cholesterol 100–400  ·  Blood Sugar 70–300  ·  Activity 0–20  ·  Sleep 3–10",
            bg=CARD_BG, fg=BORDER,
            font=("Helvetica", 8),
            wraplength=820, justify="left"
        ).pack(anchor="w", padx=10, pady=(0, 6))

        # Predict button
        tk.Button(
            outer,
            text="Predict Risk  →",
            bg=ACCENT_BLUE, fg=BG,
            font=("Helvetica", 12, "bold"),
            relief="flat", bd=0,
            padx=24, pady=10,
            cursor="hand2",
            command=self._predict
        ).pack(pady=(10, 4))


    # ── Result panel ──────────────────────────────────────────────────────────
    def _build_result_panel(self):

        outer = self._make_card(self.inner, "Prediction Result")

        self.result_frame = tk.Frame(
            outer,
            bg=LOW_BG,
            highlightbackground=LOW_BORDER,
            highlightthickness=2
        )
        self.result_frame.pack(fill="x", padx=4, pady=4)

        self.risk_label = tk.Label(
            self.result_frame,
            text="LOW RISK",
            bg=LOW_BG, fg=LOW_FG,
            font=("Helvetica", 28, "bold"),
            pady=10
        )
        self.risk_label.pack()

        self.risk_sub = tk.Label(
            self.result_frame,
            text="No major risk factors detected",
            bg=LOW_BG, fg=LOW_FG,
            font=("Helvetica", 11),
            pady=4
        )
        self.risk_sub.pack()


    # ── Risk flags panel ──────────────────────────────────────────────────────
    def _build_flags_panel(self):

        outer = self._make_card(self.inner, "Active Risk Flags")
        self.flags_frame = tk.Frame(outer, bg=CARD_BG)
        self.flags_frame.pack(fill="x", padx=4)


    # ── Feature importance panel ──────────────────────────────────────────────
    def _build_importance_panel(self):

        outer = self._make_card(self.inner, "Feature Importances  (from trained model)")

        importances = [
            ("BMI",               0.2055),
            ("Cholesterol",       0.1556),
            ("Blood Sugar",       0.1465),
            ("Smoking",           0.1352),
            ("Blood Pressure",    0.1239),
            ("Age",               0.1182),
            ("Physical Activity", 0.1090),
            ("Sleep Hours",       0.0062),
        ]

        max_imp = importances[0][1]

        for i, (name, val) in enumerate(importances):
            row = tk.Frame(outer, bg=CARD_BG)
            row.pack(fill="x", padx=4, pady=3)

            tk.Label(
                row, text=name,
                bg=CARD_BG, fg=TEXT_DARK,
                font=("Helvetica", 10),
                width=18, anchor="w"
            ).pack(side="left")

            bar_bg = tk.Frame(row, bg=BORDER, height=10)
            bar_bg.pack(side="left", fill="x", expand=True, padx=(4, 8))
            bar_bg.pack_propagate(False)

            bar_fill = tk.Frame(bar_bg, bg=BAR_COLOURS[i], height=10)
            bar_fill.place(relwidth=val / max_imp, relheight=1)

            tk.Label(
                row,
                text=f"{val*100:.1f}%",
                bg=CARD_BG, fg=TEXT_MUTED,
                font=("Helvetica", 9),
                width=5
            ).pack(side="right")


    # ── Model stats panel ─────────────────────────────────────────────────────
    def _build_stats_panel(self):

        outer = self._make_card(self.inner, "Model Summary")

        stats_row = tk.Frame(outer, bg=CARD_BG)
        stats_row.pack(fill="x", padx=4)

        stats = [
            ("Test Accuracy",  "90.0%"),
            ("Tree Depth",     "7"),
            ("Training Rows",  "800"),
            ("Features",       "8"),
        ]

        for label, value in stats:
            box = tk.Frame(stats_row, bg=BG, padx=16, pady=10)
            box.pack(side="left", expand=True, fill="x", padx=4)

            tk.Label(
                box, text=value,
                bg=BG, fg=ACCENT_BLUE,
                font=("Helvetica", 20, "bold")
            ).pack()

            tk.Label(
                box, text=label,
                bg=BG, fg=TEXT_MUTED,
                font=("Helvetica", 9)
            ).pack()


    # ─────────────────────────────────────────────────────────────────────────
    #  EVENT HANDLERS
    # ─────────────────────────────────────────────────────────────────────────

    def _on_entry(self, key, lo, hi, resolution):
        """Validate entry is within range then predict."""
        try:
            val = float(self.vars[key].get())
            # Turn entry red if out of range, blue if valid
            if val < lo or val > hi:
                self.entries[key].config(fg=HIGH_FG)
            else:
                self.entries[key].config(fg=ACCENT_BLUE)
        except ValueError:
            # Still typing — not a valid number yet
            self.entries[key].config(fg=HIGH_FG)
            return
        self._predict()


    def _set_smoking(self, val):
        """Toggle smoking buttons and re-predict."""
        self.smoking_var.set(val)
        if val == 0:
            self.smoke_no_btn.config(
                bg=ACCENT_BLUE, fg=BG,
                font=("Helvetica", 11, "bold")
            )
            self.smoke_yes_btn.config(
                bg="#0D1120", fg=TEXT_MUTED,
                font=("Helvetica", 11)
            )
        else:
            self.smoke_no_btn.config(
                bg="#0D1120", fg=TEXT_MUTED,
                font=("Helvetica", 11)
            )
            self.smoke_yes_btn.config(
                bg=HIGH_BG, fg=HIGH_FG,
                font=("Helvetica", 11, "bold")
            )
        self._predict()


    def _predict(self):
        """
        Read all entry values, run prediction, update the UI.
        Uses the real loaded model if available, otherwise
        falls back to rule-based logic from generate_data.py.
        """

        # ── Read values ───────────────────────────────────────────────────────
        try:
            age   = float(self.vars["age"].get())
            bmi   = float(self.vars["bmi"].get())
            bp    = float(self.vars["bp"].get())
            chol  = float(self.vars["chol"].get())
            sugar = float(self.vars["sugar"].get())
            act   = float(self.vars["act"].get())
            sleep = float(self.vars["sleep"].get())
            smoke = self.smoking_var.get()
        except ValueError:
            # User is mid-typing — don't predict yet
            return

        # ── Run prediction ────────────────────────────────────────────────────
        if self.model:
            # Real trained Decision Tree
            # Column order must match training exactly:
            # Age, BMI, BloodPressure, Cholesterol,
            # BloodSugar, Smoking, PhysicalActivity, SleepHours
            features = np.array([[age, bmi, bp, chol, sugar, smoke, act, sleep]])
            pred_num = self.model.predict(features)[0]
            risk     = self.label_map[pred_num]
        else:
            # Fallback — mirrors generate_data.py label rules
            score = sum([
                bmi   > 30,
                bp    > 140,
                chol  > 240,
                sugar > 126,
                smoke == 1,
                age   > 60,
                act   < 1.5,
            ])
            risk = "Low" if score <= 1 else ("Medium" if score <= 3 else "High")

        # ── Update result card ────────────────────────────────────────────────
        colours = {
            "Low":    (LOW_BG,  LOW_FG,  LOW_BORDER,
                       "LOW RISK",    "No major risk factors detected"),
            "Medium": (MED_BG,  MED_FG,  MED_BORDER,
                       "MEDIUM RISK", "Some risk factors present"),
            "High":   (HIGH_BG, HIGH_FG, HIGH_BORDER,
                       "HIGH RISK",   "Multiple risk factors detected"),
        }
        bg, fg, border, title, sub = colours[risk]

        self.result_frame.config(bg=bg, highlightbackground=border)
        self.risk_label.config(bg=bg, fg=fg, text=title)
        self.risk_sub.config(bg=bg, fg=fg, text=sub)

        # ── Update flags ──────────────────────────────────────────────────────
        for widget in self.flags_frame.winfo_children():
            widget.destroy()

        flags = []
        if bmi   > 30:   flags.append((f"BMI {bmi:.1f} — obese range",               MED_BG,  MED_FG))
        if bp    > 140:  flags.append((f"BP {int(bp)} mmHg — hypertension",          HIGH_BG, HIGH_FG))
        if chol  > 240:  flags.append((f"Cholesterol {int(chol)} — elevated",        MED_BG,  MED_FG))
        if sugar > 126:  flags.append((f"Blood sugar {int(sugar)} — diabetic range", HIGH_BG, HIGH_FG))
        if smoke == 1:   flags.append(("Smoker",                                      MED_BG,  MED_FG))
        if age   > 60:   flags.append((f"Age {int(age)} — elevated age risk",         LOW_BG,  LOW_FG))
        if act   < 1.5:  flags.append((f"Activity {act:.1f} h/wk — very low",        LOW_BG,  LOW_FG))

        if not flags:
            tk.Label(
                self.flags_frame,
                text="No risk flags — all values in healthy range",
                bg=CARD_BG, fg=TEXT_MUTED,
                font=("Helvetica", 10)
            ).pack(anchor="w", pady=6)
        else:
            wrap = tk.Frame(self.flags_frame, bg=CARD_BG)
            wrap.pack(fill="x", pady=4)
            for text, fbg, ffg in flags:
                tk.Label(
                    wrap,
                    text=f"  {text}  ",
                    bg=fbg, fg=ffg,
                    font=("Helvetica", 10, "bold"),
                    padx=8, pady=4,
                    relief="flat"
                ).pack(side="left", padx=4, pady=2)


    # ─────────────────────────────────────────────────────────────────────────
    #  HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    def _make_card(self, parent, title):
        """Create a titled dark card panel."""
        outer = tk.Frame(parent, bg=BG, padx=16, pady=4)
        outer.pack(fill="x", pady=4)

        tk.Label(
            outer,
            text=title.upper(),
            bg=BG, fg=TEXT_MUTED,
            font=("Helvetica", 9, "bold")
        ).pack(anchor="w", pady=(4, 4))

        card = tk.Frame(
            outer,
            bg=CARD_BG,
            highlightbackground=BORDER,
            highlightthickness=1,
            padx=12, pady=12
        )
        card.pack(fill="x")
        return card


    def _on_frame_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))


    def _on_canvas_configure(self, event):
        self.canvas.itemconfig(self.canvas_window, width=event.width)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    app  = HealthRiskApp(root)
    root.mainloop()