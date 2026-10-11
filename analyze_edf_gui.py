#!/usr/bin/env python3
"""
Interactive EDF analysis setup GUI.

Change log:
- 2026-09-17: Added Step 5b for exactly two surface-EMG channels: session-wide
  95th-percentile normalization, RMS-envelope overlap co-contraction index (CCI),
  common-activation magnitude, simultaneous-activation percentage, and event-level plots.
- 2026-09-17: Added Step 5c for exactly two surface-EMG channels: intermuscular
  magnitude-squared coherence spectra and event-level band-mean summaries. The default
  4-12 Hz band is informed by PMID 39700629; this is a shared-drive/synchrony measure,
  not an amplitude co-contraction or joint-stiffness measure.
- 2026-09-25: Added Step 5d: per-event Welch power spectral-density plots for one or
  more selected bandpass-filtered channels.

Current scope:
1. Open a file chooser to select an EDF file.
2. Open a second GUI with four setup subparts:
   - Channel + bandpass filter limits
   - Analysis type: baseline, stimulation, or both
   - Baseline annotation/window selection and preview
   - Stimulation annotation/window selection and preview
   - Final signal analysis: bandpass, RMS smoothing, rectification, and box-whisker plot

Dependencies:
    pip install mne matplotlib numpy scipy

Recommended run from this project:
    python analyze_edf_gui.py
"""

from __future__ import annotations

import json
import sys
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np

try:
    import mne
except ImportError as exc:  # handled in main
    mne = None
    MNE_IMPORT_ERROR = exc
else:
    MNE_IMPORT_ERROR = None

try:
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    from matplotlib.figure import Figure
except ImportError as exc:  # handled in preview
    matplotlib = None
    FigureCanvasTkAgg = None
    NavigationToolbar2Tk = None
    Figure = None
    MPL_IMPORT_ERROR = exc
else:
    MPL_IMPORT_ERROR = None

try:
    from scipy.signal import coherence, welch
except ImportError as exc:  # handled in Step 5c
    coherence = None
    welch = None
    SCIPY_IMPORT_ERROR = exc
else:
    SCIPY_IMPORT_ERROR = None


@dataclass
class AnalysisConfig:
    edf_path: str
    channels: list[str]
    bandpass_low_hz: float
    bandpass_high_hz: float
    analysis_mode: str
    baseline_annotation: str | None
    baseline_window_s: tuple[float, float] | None
    stimulation_annotations: list[str]
    stimulation_window_s: tuple[float, float] | None
    baseline_boundary_annotations: list[str] | None = None
    stimulation_boundary_pairs: list[list[str]] | None = None
    baseline_window_mode: str = "relative"
    stimulation_window_mode: str = "relative"


class EDFAnalysisSetupGUI:
    """Second-stage EDF setup GUI."""

    def __init__(self, root: tk.Tk, edf_path: str):
        self.root = root
        self.edf_path = str(edf_path)
        self.raw = None
        self.preview_window: tk.Toplevel | None = None

        self.selected_channels: list[str] = []
        self.selected_channels_display = tk.StringVar(value="No channels selected")
        self.low_hz = tk.StringVar(value="20")
        self.high_hz = tk.StringVar(value="450")
        self.notch_60_enabled = tk.BooleanVar(value=False)
        self.analysis_mode = tk.StringVar(value="baseline_and_stimulation")
        self.baseline_tmin = tk.StringVar(value="-45")
        self.baseline_tmax = tk.StringVar(value="-1")
        self.stim_tmin = tk.StringVar(value="0")
        self.stim_tmax = tk.StringVar(value="60")
        self.baseline_window_mode = tk.StringVar(value="relative")
        self.stim_window_mode = tk.StringVar(value="relative")
        self.baseline_pair_indices: list[int] = []
        self.stim_pair_indices: list[tuple[int, int]] = []
        self.rms_window_ms = tk.StringVar(value="100")
        self.cci_threshold_pct = tk.StringVar(value="10")
        self.coherence_low_hz = tk.StringVar(value="4")
        self.coherence_high_hz = tk.StringVar(value="12")
        self.coherence_segment_s = tk.StringVar(value="1")
        self.psd_segment_s = tk.StringVar(value="0.5")
        self.psd_xmin_hz = tk.StringVar(value="1")
        self.psd_xmax_hz = tk.StringVar(value="45")
        self._session_p95_cache: dict[tuple[object, ...], dict[str, float]] = {}
        self.custom_plot_title = tk.StringVar(value="")
        self.custom_plot_titles: dict[str, str] = {}
        self.selected_baseline_annotation_indices: list[int] = []
        self.selected_stim_annotation_indices: list[int] = []
        self.baseline_annotation_display = tk.StringVar(value="No baseline annotation selected")
        self.stim_annotation_display = tk.StringVar(value="No stimulation annotations selected")
        self.status = tk.StringVar(value="Loading EDF...")

        self._build_gui()
        self._load_edf()

    # ------------------------- GUI construction -------------------------
    def _build_gui(self) -> None:
        self.root.title("EDF Analysis Setup")
        self.root.geometry("1000x780")
        self.root.minsize(900, 650)

        # Keep the complete setup visible on ordinary laptop displays.  Steps 5b-5d
        # extend the form beyond the initial window height, so place the
        # content in a vertically scrollable canvas rather than clipping its end.
        scroll_canvas = tk.Canvas(self.root, highlightthickness=0)
        y_scrollbar = ttk.Scrollbar(self.root, orient=tk.VERTICAL, command=scroll_canvas.yview)
        scroll_canvas.configure(yscrollcommand=y_scrollbar.set)
        y_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        scroll_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        outer = ttk.Frame(scroll_canvas, padding=12)
        outer_window = scroll_canvas.create_window((0, 0), window=outer, anchor=tk.NW)

        def _update_scroll_region(_event: tk.Event | None = None) -> None:
            scroll_canvas.configure(scrollregion=scroll_canvas.bbox("all"))

        def _fit_content_width(event: tk.Event) -> None:
            scroll_canvas.itemconfigure(outer_window, width=event.width)

        outer.bind("<Configure>", _update_scroll_region)
        scroll_canvas.bind("<Configure>", _fit_content_width)

        def _mousewheel(event: tk.Event) -> None:
            # Windows/macOS report delta; X11 reports mouse buttons 4 and 5.
            if event.delta:
                scroll_canvas.yview_scroll(-int(event.delta / 120), "units")
            elif event.num == 4:
                scroll_canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                scroll_canvas.yview_scroll(1, "units")

        scroll_canvas.bind_all("<MouseWheel>", _mousewheel)
        scroll_canvas.bind_all("<Button-4>", _mousewheel)
        scroll_canvas.bind_all("<Button-5>", _mousewheel)

        title = ttk.Label(
            outer,
            text=f"EDF analysis setup\n{self.edf_path}",
            font=("TkDefaultFont", 11, "bold"),
            justify=tk.LEFT,
        )
        title.pack(anchor=tk.W, pady=(0, 10))

        # Subpart 1: channel + bandpass
        channel_frame = ttk.LabelFrame(
            outer,
            text="1. Channel of interest and bandpass filter",
            padding=10,
        )
        channel_frame.pack(fill=tk.X, pady=5)

        ttk.Button(
            channel_frame,
            text="Choose channel...",
            command=self.choose_channel,
        ).grid(row=0, column=0, padx=(0, 8), pady=4, sticky=tk.W)
        ttk.Label(channel_frame, textvariable=self.selected_channels_display, width=45, wraplength=360).grid(
            row=0, column=1, padx=(0, 16), sticky=tk.W
        )

        ttk.Label(channel_frame, text="Bandpass low Hz:").grid(row=0, column=2, padx=(0, 4))
        ttk.Entry(channel_frame, textvariable=self.low_hz, width=8).grid(row=0, column=3, padx=(0, 12))
        ttk.Label(channel_frame, text="Bandpass high Hz:").grid(row=0, column=4, padx=(0, 4))
        ttk.Entry(channel_frame, textvariable=self.high_hz, width=8).grid(row=0, column=5)
        ttk.Checkbutton(channel_frame, text="Apply 60 Hz notch to raw signal (all analyses)",
                        variable=self.notch_60_enabled).grid(row=1, column=0, columnspan=6, sticky=tk.W, pady=(4, 0))

        # Subpart 2: analysis mode
        mode_frame = ttk.LabelFrame(
            outer,
            text="2. Analysis type",
            padding=10,
        )
        mode_frame.pack(fill=tk.X, pady=5)

        modes = [
            ("A) Baseline only", "baseline"),
            ("B) Stimulation only", "stimulation"),
            ("C) Baseline and stimulation", "baseline_and_stimulation"),
        ]
        for i, (label, value) in enumerate(modes):
            ttk.Radiobutton(
                mode_frame,
                text=label,
                value=value,
                variable=self.analysis_mode,
                command=self._update_enabled_sections,
            ).grid(row=0, column=i, padx=12, sticky=tk.W)

        # Subpart 3: baseline events/window/preview
        self.baseline_frame = ttk.LabelFrame(
            outer,
            text="3. Baseline annotation, analysis window, and preview",
            padding=10,
        )
        self.baseline_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self._build_baseline_section(self.baseline_frame)

        # Subpart 4: stimulation events/window/preview
        self.stim_frame = ttk.LabelFrame(
            outer,
            text="4. Stimulation annotation(s), analysis window, and preview",
            padding=10,
        )
        self.stim_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self._build_stimulation_section(self.stim_frame)

        self.analysis_frame = ttk.LabelFrame(outer, text="5a. RMS amplitude analysis", padding=10)
        self.analysis_frame.pack(fill=tk.X, pady=5)
        self._build_analysis_section(self.analysis_frame)

        self.cocontraction_frame = ttk.LabelFrame(outer, text="5b. Two-channel surface-EMG co-contraction", padding=10)
        self.cocontraction_frame.pack(fill=tk.X, pady=5)
        self._build_cocontraction_section(self.cocontraction_frame)

        self.coherence_frame = ttk.LabelFrame(outer, text="5c. Two-channel intermuscular coherence", padding=10)
        self.coherence_frame.pack(fill=tk.X, pady=5)
        self._build_coherence_section(self.coherence_frame)

        self.psd_frame = ttk.LabelFrame(outer, text="5d. Welch spectral density analysis", padding=10)
        self.psd_frame.pack(fill=tk.X, pady=5)
        self._build_psd_section(self.psd_frame)

        # Footer
        footer = ttk.Frame(outer)
        footer.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(footer, textvariable=self.status).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(footer, text="Print/Save current setup", command=self.save_config).pack(side=tk.RIGHT, padx=(8, 0))
        ttk.Button(footer, text="Close", command=self.root.destroy).pack(side=tk.RIGHT)

    def _build_baseline_section(self, parent: ttk.Frame) -> None:
        mode = ttk.Frame(parent); mode.pack(fill=tk.X)
        ttk.Radiobutton(mode, text="Relative time window", variable=self.baseline_window_mode, value="relative", command=self._refresh_window_controls).pack(side=tk.LEFT)
        ttk.Radiobutton(mode, text="Between annotation pair", variable=self.baseline_window_mode, value="pair", command=self._refresh_window_controls).pack(side=tk.LEFT, padx=10)
        self.baseline_relative_controls = ttk.Frame(parent); self.baseline_relative_controls.pack(fill=tk.X)
        ttk.Button(self.baseline_relative_controls, text="Choose baseline annotation...", command=self.choose_baseline_annotation).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Label(self.baseline_relative_controls, text="Window relative to annotation (s):").pack(side=tk.LEFT, padx=(8, 4))
        ttk.Entry(self.baseline_relative_controls, textvariable=self.baseline_tmin, width=8).pack(side=tk.LEFT)
        ttk.Label(self.baseline_relative_controls, text="to").pack(side=tk.LEFT, padx=4)
        ttk.Entry(self.baseline_relative_controls, textvariable=self.baseline_tmax, width=8).pack(side=tk.LEFT)
        self.baseline_pair_controls = ttk.Frame(parent); self.baseline_pair_controls.pack(fill=tk.X)
        ttk.Button(self.baseline_pair_controls, text="Choose start/end annotation pair...", command=self.choose_baseline_pair).pack(side=tk.LEFT)
        self.baseline_pair_display = tk.StringVar(value="No annotation pair selected")
        ttk.Label(self.baseline_pair_controls, textvariable=self.baseline_pair_display, wraplength=800).pack(side=tk.LEFT, padx=8)
        ttk.Button(parent, text="Preview baseline window", command=self.preview_baseline).pack(anchor=tk.W, pady=4)
        ttk.Label(parent, text="Pair mode uses the interval between two selected annotation onsets. Ignored for stimulation-only mode.", foreground="gray30").pack(anchor=tk.W)
        ttk.Label(parent, textvariable=self.baseline_annotation_display, wraplength=900).pack(anchor=tk.W, pady=(4, 0), fill=tk.X)

    def _build_stimulation_section(self, parent: ttk.Frame) -> None:
        mode = ttk.Frame(parent); mode.pack(fill=tk.X)
        ttk.Radiobutton(mode, text="Relative time window", variable=self.stim_window_mode, value="relative", command=self._refresh_window_controls).pack(side=tk.LEFT)
        ttk.Radiobutton(mode, text="Between annotation pairs", variable=self.stim_window_mode, value="pair", command=self._refresh_window_controls).pack(side=tk.LEFT, padx=10)
        self.stim_relative_controls = ttk.Frame(parent); self.stim_relative_controls.pack(fill=tk.X)
        ttk.Button(self.stim_relative_controls, text="Choose stimulation annotation(s)...", command=self.choose_stim_annotations).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Label(self.stim_relative_controls, text="Window relative to annotation (s):").pack(side=tk.LEFT, padx=(8, 4))
        ttk.Entry(self.stim_relative_controls, textvariable=self.stim_tmin, width=8).pack(side=tk.LEFT)
        ttk.Label(self.stim_relative_controls, text="to").pack(side=tk.LEFT, padx=4)
        ttk.Entry(self.stim_relative_controls, textvariable=self.stim_tmax, width=8).pack(side=tk.LEFT)
        self.stim_pair_controls = ttk.Frame(parent); self.stim_pair_controls.pack(fill=tk.X)
        ttk.Button(self.stim_pair_controls, text="Add annotation pair(s)...", command=self.choose_stim_pairs).pack(side=tk.LEFT)
        self.stim_pair_display = tk.StringVar(value="No annotation pairs selected")
        ttk.Label(self.stim_pair_controls, textvariable=self.stim_pair_display, wraplength=800).pack(side=tk.LEFT, padx=8)
        ttk.Button(parent, text="Preview stimulation window(s)", command=self.preview_stimulation).pack(anchor=tk.W, pady=4)
        ttk.Label(parent, text="Pair mode creates one segment per pair; segment lengths may differ. Ignored for baseline-only mode.", foreground="gray30").pack(anchor=tk.W)
        ttk.Label(parent, textvariable=self.stim_annotation_display, wraplength=900).pack(anchor=tk.W, pady=(4, 0), fill=tk.X)
        self._refresh_window_controls()

    def _refresh_window_controls(self) -> None:
        if hasattr(self, "baseline_relative_controls"):
            (self.baseline_relative_controls.pack if self.baseline_window_mode.get() == "relative" else self.baseline_relative_controls.pack_forget)()
            (self.baseline_pair_controls.pack if self.baseline_window_mode.get() == "pair" else self.baseline_pair_controls.pack_forget)()
        if hasattr(self, "stim_relative_controls"):
            (self.stim_relative_controls.pack if self.stim_window_mode.get() == "relative" else self.stim_relative_controls.pack_forget)()
            (self.stim_pair_controls.pack if self.stim_window_mode.get() == "pair" else self.stim_pair_controls.pack_forget)()

    def _annotation_pair_window(self, indices: tuple[int, int]) -> tuple[float, tuple[float, float], str]:
        first, second = (self._get_annotation_by_index(i) for i in indices)
        start, stop = first[0], second[0]
        if stop <= start:
            raise ValueError("The end annotation must occur after the start annotation.")
        return start, (0.0, stop-start), f"{first[2]} → {second[2]}"

    def choose_baseline_pair(self) -> None:
        selected=self._choose_annotations_dialog("Choose baseline boundary pair", "Select exactly two annotations in chronological order (start then end):", True, self.baseline_pair_indices)
        if selected is not None:
            if len(selected) != 2:
                messagebox.showwarning("Choose a pair", "Select exactly two annotations."); return
            try: self._annotation_pair_window(tuple(selected))
            except ValueError as exc: messagebox.showerror("Invalid pair", str(exc)); return
            self.baseline_pair_indices=selected
            self.baseline_pair_display.set(" → ".join(f"#{i} {self._get_annotation_by_index(i)[2]}" for i in selected))

    def choose_stim_pairs(self) -> None:
        selected=self._choose_annotations_dialog("Choose stimulation boundary annotations", "Select an even number of annotations; consecutive selections form start/end pairs in chronological order:", True, [i for pair in self.stim_pair_indices for i in pair])
        if selected is None: return
        if len(selected)<2 or len(selected)%2:
            messagebox.showwarning("Incomplete pairs", "Select at least two annotations and an even number overall."); return
        pairs=[tuple(selected[i:i+2]) for i in range(0,len(selected),2)]
        try:
            for pair in pairs: self._annotation_pair_window(pair)
        except ValueError as exc: messagebox.showerror("Invalid pair", str(exc)); return
        self.stim_pair_indices=pairs
        self.stim_pair_display.set("; ".join(f"#{a} {self._get_annotation_by_index(a)[2]} → #{b} {self._get_annotation_by_index(b)[2]}" for a,b in pairs))

    def _build_analysis_section(self, parent: ttk.Frame) -> None:
        top = ttk.Frame(parent)
        top.pack(fill=tk.X)
        ttk.Label(top, text="RMS smoothing window, ms:").pack(side=tk.LEFT, padx=(0, 4))
        ttk.Entry(top, textvariable=self.rms_window_ms, width=8).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Label(top, text="Default plot title:").pack(side=tk.LEFT, padx=(0, 4))
        ttk.Entry(top, textvariable=self.custom_plot_title, width=28).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(top, text="Set per-channel titles...", command=self.edit_custom_plot_titles).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Button(top, text="Run signal analysis and show box-whisker plot(s)", command=self.run_signal_analysis).pack(side=tk.LEFT)
        help_text = (
            "For each selected channel, extracts windows, applies the subpart 1 bandpass, computes a moving RMS envelope, "
            "rectifies it, and plots values in microvolts. Each selected channel produces a separate PNG-capable figure. "
            "First box: subpart 3 baseline ('pre-stimulus'); subsequent boxes: subpart 4 stimulation annotations."
        )
        ttk.Label(parent, text=help_text, foreground="gray30", wraplength=900).pack(anchor=tk.W, pady=(4, 0), fill=tk.X)

    def _build_cocontraction_section(self, parent: ttk.Frame) -> None:
        top = ttk.Frame(parent)
        top.pack(fill=tk.X)
        ttk.Label(top, text="Simultaneous-activation threshold (% of session P95):").pack(side=tk.LEFT, padx=(0, 4))
        ttk.Entry(top, textvariable=self.cci_threshold_pct, width=7).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Button(top, text="Run 5b co-contraction analysis", command=self.run_cocontraction_analysis).pack(side=tk.LEFT)
        ttk.Label(parent, text=(
            "Requires exactly two anatomically opposing surface-EMG channels. Each channel is filtered and RMS-smoothed, "
            "then divided by its session-wide 95th-percentile RMS value. Reports event-level CCI (overlap; 0-100%), "
            "common activation, and percent simultaneous activation. CCI is not joint stiffness or dystonia severity."
        ), foreground="gray30", wraplength=900).pack(anchor=tk.W, pady=(4, 0), fill=tk.X)

    def _build_coherence_section(self, parent: ttk.Frame) -> None:
        top = ttk.Frame(parent)
        top.pack(fill=tk.X)
        ttk.Label(top, text="Summary band (Hz):").pack(side=tk.LEFT, padx=(0, 4))
        ttk.Entry(top, textvariable=self.coherence_low_hz, width=6).pack(side=tk.LEFT)
        ttk.Label(top, text="to").pack(side=tk.LEFT, padx=4)
        ttk.Entry(top, textvariable=self.coherence_high_hz, width=6).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Label(top, text="Welch segment (s):").pack(side=tk.LEFT, padx=(0, 4))
        ttk.Entry(top, textvariable=self.coherence_segment_s, width=6).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Button(top, text="Run 5c coherence analysis", command=self.run_coherence_analysis).pack(side=tk.LEFT)
        ttk.Label(parent, text=(
            "Requires exactly two surface-EMG channels and matched event windows. Computes magnitude-squared intermuscular "
            "coherence on bandpass-filtered EMG (not RMS envelopes); the default 4-12 Hz summary band is informed by PMID 39700629. "
            "Coherence indexes frequency-specific synchrony/shared drive, not amplitude co-contraction."
        ), foreground="gray30", wraplength=900).pack(anchor=tk.W, pady=(4, 0), fill=tk.X)

    def _build_psd_section(self, parent: ttk.Frame) -> None:
        top = ttk.Frame(parent)
        top.pack(fill=tk.X)
        ttk.Label(top, text="Welch segment (s):").pack(side=tk.LEFT, padx=(0, 4))
        ttk.Entry(top, textvariable=self.psd_segment_s, width=6).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Label(top, text="X-axis (Hz):").pack(side=tk.LEFT, padx=(0, 4))
        ttk.Entry(top, textvariable=self.psd_xmin_hz, width=5).pack(side=tk.LEFT)
        ttk.Label(top, text="to").pack(side=tk.LEFT, padx=4)
        ttk.Entry(top, textvariable=self.psd_xmax_hz, width=5).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Button(top, text="Run 5d Welch PSD analysis", command=self.run_psd_analysis).pack(side=tk.LEFT)
        ttk.Label(parent, text=(
            "Uses the selected baseline/stimulation event windows and all selected channels. Computes Welch PSD from the raw signal; "
            "the optional 60 Hz notch in section 1 is applied only when its checkbox is enabled (subpart 1 bandpass settings are not applied). "
            "The segment length controls frequency resolution and averaging; it is capped at each window's available duration."
        ), foreground="gray30", wraplength=900).pack(anchor=tk.W, pady=(4, 0), fill=tk.X)

    # ------------------------- EDF loading -------------------------
    def _load_edf(self) -> None:
        try:
            self.raw = mne.io.read_raw_edf(self.edf_path, preload=False, verbose="ERROR")
        except Exception as exc:
            self.status.set("Failed to load EDF")
            messagebox.showerror("EDF load error", f"Could not load EDF file:\n{self.edf_path}\n\n{exc}")
            return

        sfreq = self.raw.info["sfreq"]
        duration = self.raw.n_times / sfreq if sfreq else float("nan")
        msg = (
            f"Loaded EDF: {len(self.raw.ch_names)} channels, "
            f"sampling rate {sfreq:g} Hz, duration {duration:.2f} s, "
            f"annotations {len(self.raw.annotations)}"
        )
        self.status.set(msg)
        if self.raw.ch_names:
            self.selected_channels = [self.raw.ch_names[0]]
            self._update_channel_display()

        self._update_annotation_displays()
        self._update_enabled_sections()

    # ------------------------- Helpers -------------------------
    def choose_channel(self) -> None:
        if self.raw is None:
            messagebox.showwarning("No EDF", "EDF has not been loaded yet.")
            return

        win = tk.Toplevel(self.root)
        win.title("Choose channels of interest")
        win.geometry("560x560")
        win.transient(self.root)
        win.grab_set()

        ttk.Label(win, text="Select one or more channels:").pack(anchor=tk.W, padx=10, pady=(10, 4))
        ttk.Label(win, text="Use Ctrl/Shift-click to select multiple channels. A vertical scrollbar is on the right.", foreground="gray30").pack(anchor=tk.W, padx=10)
        search_var = tk.StringVar(value="")
        search_entry = ttk.Entry(win, textvariable=search_var)
        search_entry.pack(fill=tk.X, padx=10, pady=(0, 6))

        list_frame = ttk.Frame(win)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)

        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL)
        listbox = tk.Listbox(
            list_frame,
            selectmode=tk.EXTENDED,
            exportselection=False,
            yscrollcommand=scrollbar.set,
        )
        scrollbar.config(command=listbox.yview)
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        filtered_channels: list[str] = []
        initial_set = set(self.selected_channels)

        def refresh_list(*_args):
            query = search_var.get().lower().strip()
            listbox.delete(0, tk.END)
            filtered_channels.clear()
            for ch in self.raw.ch_names:
                if not query or query in ch.lower():
                    filtered_channels.append(ch)
                    listbox.insert(tk.END, ch)
            for pos, ch in enumerate(filtered_channels):
                if ch in initial_set:
                    listbox.selection_set(pos)

        def accept_selection():
            selection = listbox.curselection()
            if not selection:
                messagebox.showwarning("No channel selected", "Please select one or more channels.", parent=win)
                return
            self.selected_channels = [filtered_channels[pos] for pos in selection]
            self._update_channel_display()
            self._session_p95_cache.clear()
            # Preserve existing per-channel titles for retained channels only.
            self.custom_plot_titles = {ch: self.custom_plot_titles.get(ch, "") for ch in self.selected_channels if ch in self.custom_plot_titles}
            win.destroy()

        def select_all_visible():
            listbox.selection_set(0, tk.END)

        button_frame = ttk.Frame(win)
        button_frame.pack(fill=tk.X, padx=10, pady=(2, 10))
        ttk.Button(button_frame, text="Use selected channel(s)", command=accept_selection).pack(side=tk.RIGHT)
        ttk.Button(button_frame, text="Select all visible", command=select_all_visible).pack(side=tk.RIGHT, padx=(0, 8))
        ttk.Button(button_frame, text="Cancel", command=win.destroy).pack(side=tk.RIGHT, padx=(0, 8))

        search_var.trace_add("write", refresh_list)
        refresh_list()
        listbox.bind("<Double-Button-1>", lambda _event: accept_selection())
        search_entry.focus_set()

    def _update_channel_display(self) -> None:
        if not self.selected_channels:
            self.selected_channels_display.set("No channels selected")
            return
        shown = ", ".join(self.selected_channels[:4])
        if len(self.selected_channels) > 4:
            shown += f"; ... +{len(self.selected_channels) - 4} more"
        self.selected_channels_display.set(f"Selected channels ({len(self.selected_channels)}): {shown}")

    def edit_custom_plot_titles(self) -> None:
        if not self.selected_channels:
            messagebox.showwarning("No channels selected", "Choose one or more channels of interest first.")
            return

        win = tk.Toplevel(self.root)
        win.title("Custom plot titles for selected channels")
        win.geometry("720x420")
        win.transient(self.root)
        win.grab_set()

        ttk.Label(win, text="Enter a distinct custom plot title for each selected channel. Blank uses the default title.").pack(anchor=tk.W, padx=10, pady=(10, 6))

        canvas = tk.Canvas(win, borderwidth=0)
        scrollbar = ttk.Scrollbar(win, orient=tk.VERTICAL, command=canvas.yview)
        inner = ttk.Frame(canvas)
        inner.bind("<Configure>", lambda _event: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0), pady=6)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 10), pady=6)

        title_vars: dict[str, tk.StringVar] = {}
        default_title = self.custom_plot_title.get().strip()
        for row, ch in enumerate(self.selected_channels):
            ttk.Label(inner, text=ch, width=28).grid(row=row, column=0, sticky=tk.W, padx=(0, 8), pady=3)
            title_vars[ch] = tk.StringVar(value=self.custom_plot_titles.get(ch, default_title))
            ttk.Entry(inner, textvariable=title_vars[ch], width=58).grid(row=row, column=1, sticky=tk.EW, pady=3)
        inner.columnconfigure(1, weight=1)

        def save_titles():
            self.custom_plot_titles = {ch: title_vars[ch].get().strip() for ch in self.selected_channels}
            win.destroy()

        button_frame = ttk.Frame(win)
        button_frame.pack(fill=tk.X, padx=10, pady=(2, 10))
        ttk.Button(button_frame, text="Save titles", command=save_titles).pack(side=tk.RIGHT)
        ttk.Button(button_frame, text="Cancel", command=win.destroy).pack(side=tk.RIGHT, padx=(0, 8))

    def _annotation_rows(self) -> list[tuple[int, float, float, str]]:
        if self.raw is None:
            return []
        rows = []
        for i, ann in enumerate(self.raw.annotations):
            rows.append((i, float(ann["onset"]), float(ann["duration"]), str(ann["description"])))
        return rows

    def choose_baseline_annotation(self) -> None:
        """Open a searchable pop-up to choose one baseline annotation."""
        selected = self._choose_annotations_dialog(
            title="Choose baseline annotation",
            instruction="Select one baseline annotation:",
            multiple=False,
            initial_indices=self.selected_baseline_annotation_indices,
        )
        if selected is not None:
            self.selected_baseline_annotation_indices = selected[:1]
            self._update_annotation_displays()

    def choose_stim_annotations(self) -> None:
        """Open a searchable pop-up to choose one or more stimulation annotations."""
        selected = self._choose_annotations_dialog(
            title="Choose stimulation annotation(s)",
            instruction="Select one or more stimulation annotations:",
            multiple=True,
            initial_indices=self.selected_stim_annotation_indices,
        )
        if selected is not None:
            self.selected_stim_annotation_indices = selected
            self._update_annotation_displays()

    def _choose_annotations_dialog(
        self,
        title: str,
        instruction: str,
        multiple: bool,
        initial_indices: list[int] | None = None,
    ) -> list[int] | None:
        if self.raw is None:
            messagebox.showwarning("No EDF", "EDF has not been loaded yet.")
            return None

        rows = self._annotation_rows()
        if not rows:
            messagebox.showwarning("No annotations", "This EDF has no annotations/event markers.")
            return None

        win = tk.Toplevel(self.root)
        win.title(title)
        win.geometry("760x560")
        win.transient(self.root)
        win.grab_set()

        ttk.Label(win, text=instruction).pack(anchor=tk.W, padx=10, pady=(10, 4))
        ttk.Label(win, text="Search annotation text, index, or onset time:", foreground="gray30").pack(anchor=tk.W, padx=10)
        search_var = tk.StringVar(value="")
        search_entry = ttk.Entry(win, textvariable=search_var)
        search_entry.pack(fill=tk.X, padx=10, pady=(0, 6))

        list_frame = ttk.Frame(win)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)

        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL)
        listbox = tk.Listbox(
            list_frame,
            selectmode=(tk.EXTENDED if multiple else tk.SINGLE),
            exportselection=False,
            yscrollcommand=scrollbar.set,
        )
        scrollbar.config(command=listbox.yview)
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        filtered_indices: list[int] = []
        result: list[int] | None = None
        initial_set = set(initial_indices or [])

        def row_text(row: tuple[int, float, float, str]) -> str:
            i, onset, duration, desc = row
            return f"#{i} | onset {onset:.6f} s | duration {duration:.6f} s | {desc}"

        def refresh_list(*_args):
            query = search_var.get().lower().strip()
            listbox.delete(0, tk.END)
            filtered_indices.clear()
            for row in rows:
                text = row_text(row)
                if not query or query in text.lower():
                    filtered_indices.append(row[0])
                    listbox.insert(tk.END, text)
            for pos, idx in enumerate(filtered_indices):
                if idx in initial_set:
                    listbox.selection_set(pos)

        def accept_selection():
            nonlocal result
            selection = listbox.curselection()
            if not selection:
                messagebox.showwarning("No annotation selected", "Please select annotation(s).", parent=win)
                return
            result = [filtered_indices[pos] for pos in selection]
            win.destroy()

        def clear_selection():
            nonlocal result
            result = []
            win.destroy()

        button_frame = ttk.Frame(win)
        button_frame.pack(fill=tk.X, padx=10, pady=(2, 10))
        ttk.Button(button_frame, text="Use selected annotation(s)", command=accept_selection).pack(side=tk.RIGHT)
        ttk.Button(button_frame, text="Clear selection", command=clear_selection).pack(side=tk.RIGHT, padx=(0, 8))
        ttk.Button(button_frame, text="Cancel", command=win.destroy).pack(side=tk.RIGHT, padx=(0, 8))

        search_var.trace_add("write", refresh_list)
        refresh_list()
        listbox.bind("<Double-Button-1>", lambda _event: accept_selection())
        search_entry.focus_set()
        self.root.wait_window(win)
        return result

    def _update_annotation_displays(self) -> None:
        if self.selected_baseline_annotation_indices:
            idx = self.selected_baseline_annotation_indices[0]
            onset, duration, desc = self._get_annotation_by_index(idx)
            self.baseline_annotation_display.set(
                f"Selected baseline: #{idx} | onset {onset:.6f} s | duration {duration:.6f} s | {desc}"
            )
        else:
            self.baseline_annotation_display.set("No baseline annotation selected")

        if self.selected_stim_annotation_indices:
            parts = []
            for idx in self.selected_stim_annotation_indices[:5]:
                onset, _duration, desc = self._get_annotation_by_index(idx)
                parts.append(f"#{idx} at {onset:.6f} s: {desc}")
            more = ""
            if len(self.selected_stim_annotation_indices) > 5:
                more = f"; ... +{len(self.selected_stim_annotation_indices) - 5} more"
            self.stim_annotation_display.set(
                f"Selected stimulation annotations ({len(self.selected_stim_annotation_indices)}): "
                + "; ".join(parts)
                + more
            )
        else:
            self.stim_annotation_display.set("No stimulation annotations selected")

    def _selected_annotation_indices(self, selection_name: str) -> list[int]:
        if selection_name == "baseline":
            return list(self.selected_baseline_annotation_indices)
        if selection_name == "stimulation":
            return list(self.selected_stim_annotation_indices)
        raise ValueError(f"Unknown annotation selection: {selection_name}")

    def _parse_float_pair(self, low_var: tk.StringVar, high_var: tk.StringVar, label: str) -> tuple[float, float] | None:
        try:
            a = float(low_var.get())
            b = float(high_var.get())
        except ValueError:
            messagebox.showerror("Invalid input", f"{label} values must be numeric.")
            return None
        if b <= a:
            messagebox.showerror("Invalid input", f"{label} end must be greater than start.")
            return None
        return a, b

    def _get_bandpass(self) -> tuple[float, float] | None:
        bp = self._parse_float_pair(self.low_hz, self.high_hz, "Bandpass frequency")
        if bp is None:
            return None
        low, high = bp
        if low < 0:
            messagebox.showerror("Invalid input", "Bandpass low frequency must be >= 0 Hz.")
            return None
        if self.raw is not None:
            nyquist = self.raw.info["sfreq"] / 2.0
            if high >= nyquist:
                messagebox.showwarning(
                    "Bandpass adjusted",
                    f"High cutoff {high:g} Hz is at/above Nyquist ({nyquist:g} Hz).\n"
                    f"Using {nyquist * 0.99:g} Hz for preview.",
                )
                high = nyquist * 0.99
        return low, high

    def _get_annotation_by_index(self, idx: int) -> tuple[float, float, str]:
        ann = self.raw.annotations[idx]
        return float(ann["onset"]), float(ann["duration"]), str(ann["description"])

    def _clip_window_to_data(self, start_s: float, stop_s: float) -> tuple[float, float]:
        sfreq = self.raw.info["sfreq"]
        data_start = 0.0
        data_stop = self.raw.n_times / sfreq
        clipped_start = max(start_s, data_start)
        clipped_stop = min(stop_s, data_stop)
        return clipped_start, clipped_stop

    def _extract_preview_data(self, onset_s: float, window: tuple[float, float], channel: str | None = None) -> tuple[np.ndarray, np.ndarray, str]:
        if self.raw is None:
            raise RuntimeError("EDF has not been loaded.")
        ch = (channel or (self.selected_channels[0] if self.selected_channels else "")).strip()
        if not ch:
            raise RuntimeError("No channel selected.")
        if ch not in self.raw.ch_names:
            raise RuntimeError(f"Selected channel not found in EDF: {ch}")

        low, high = self._get_bandpass() or (None, None)
        if low is None or high is None:
            raise RuntimeError("Invalid bandpass settings.")

        tmin_rel, tmax_rel = window
        start_s = onset_s + tmin_rel
        stop_s = onset_s + tmax_rel
        start_s, stop_s = self._clip_window_to_data(start_s, stop_s)
        if stop_s <= start_s:
            raise RuntimeError(
                f"Requested window is outside data bounds after clipping: {start_s:.3f} to {stop_s:.3f} s"
            )

        raw_segment = self.raw.copy().pick([ch]).crop(tmin=start_s, tmax=stop_s, include_tmax=False)
        raw_segment.load_data(verbose="ERROR")
        if self.notch_60_enabled.get():
            sfreq = float(raw_segment.info["sfreq"])
            if sfreq / 2.0 <= 60.0:
                raise RuntimeError(f"Sampling rate {sfreq:g} Hz is too low for a 60 Hz notch filter.")
            raw_segment.notch_filter(freqs=[60.0], picks=[ch], verbose="ERROR")
        raw_segment.filter(l_freq=low, h_freq=high, picks=[ch], verbose="ERROR")
        data = raw_segment.get_data(picks=[ch])[0]
        times = raw_segment.times + start_s - onset_s  # relative to window anchor
        window_label = f"absolute {start_s:.3f}-{stop_s:.3f} s; relative {times[0]:.3f}-{times[-1]:.3f} s"
        return times, data, window_label

    def _extract_raw_window_data(self, onset_s: float, window: tuple[float, float], channel: str) -> tuple[np.ndarray, np.ndarray, str]:
        """Extract an event window without applying the GUI bandpass filter."""
        if self.raw is None:
            raise RuntimeError("EDF has not been loaded.")
        if channel not in self.raw.ch_names:
            raise RuntimeError(f"Selected channel not found in EDF: {channel}")
        tmin_rel, tmax_rel = window
        start_s, stop_s = self._clip_window_to_data(onset_s + tmin_rel, onset_s + tmax_rel)
        if stop_s <= start_s:
            raise RuntimeError(f"Requested window is outside data bounds after clipping: {start_s:.3f} to {stop_s:.3f} s")
        raw_segment = self.raw.copy().pick([channel]).crop(tmin=start_s, tmax=stop_s, include_tmax=False)
        raw_segment.load_data(verbose="ERROR")
        # Apply optional notch to the raw channel for all downstream analyses.
        if self.notch_60_enabled.get():
            sfreq = float(raw_segment.info["sfreq"])
            if sfreq / 2.0 <= 60.0:
                raise RuntimeError(f"Sampling rate {sfreq:g} Hz is too low for a 60 Hz notch filter.")
            raw_segment.notch_filter(freqs=[60.0], picks=[channel], verbose="ERROR")
        times = raw_segment.times + start_s - onset_s
        data = raw_segment.get_data(picks=[channel])[0]
        window_label = f"absolute {start_s:.3f}-{stop_s:.3f} s; relative {times[0]:.3f}-{times[-1]:.3f} s"
        return times, data, window_label

    def _parse_rms_window_samples(self) -> int | None:
        if self.raw is None:
            messagebox.showwarning("No EDF", "EDF has not been loaded yet.")
            return None
        try:
            rms_ms = float(self.rms_window_ms.get())
        except ValueError:
            messagebox.showerror("Invalid input", "RMS smoothing window must be numeric milliseconds.")
            return None
        if rms_ms <= 0:
            messagebox.showerror("Invalid input", "RMS smoothing window must be greater than 0 ms.")
            return None
        sfreq = float(self.raw.info["sfreq"])
        return max(1, int(round((rms_ms / 1000.0) * sfreq)))

    @staticmethod
    def _moving_rms(data: np.ndarray, window_samples: int) -> np.ndarray:
        if window_samples <= 1:
            return np.sqrt(np.square(data))
        kernel = np.ones(window_samples, dtype=float) / float(window_samples)
        return np.sqrt(np.convolve(np.square(data), kernel, mode="same"))

    def _extract_analysis_values(self, onset_s: float, window: tuple[float, float], rms_window_samples: int, channel: str) -> np.ndarray:
        _times, data_volts, _window_label = self._extract_preview_data(onset_s, window, channel=channel)
        data_uv = data_volts * 1_000_000.0
        rms_uv = self._moving_rms(data_uv, rms_window_samples)
        rectified_uv = np.abs(rms_uv)
        rectified_uv = rectified_uv[np.isfinite(rectified_uv)]
        if rectified_uv.size == 0:
            raise RuntimeError("Analysis window contained no finite samples after processing.")
        return rectified_uv

    def _analysis_events(self) -> list[tuple[str, float, tuple[float, float]]] | None:
        """Return baseline/stimulation segments for the selected window modes."""
        mode=self.analysis_mode.get(); events=[]
        if mode != "stimulation":
            if self.baseline_window_mode.get()=="pair":
                if len(self.baseline_pair_indices)!=2:
                    messagebox.showwarning("No baseline pair", "Choose a start/end annotation pair in section 3."); return None
                try: onset, window, label=self._annotation_pair_window(tuple(self.baseline_pair_indices))
                except ValueError as exc: messagebox.showerror("Invalid pair",str(exc)); return None
                events.append((f"baseline: {label}",onset,window))
            else:
                baseline=self._selected_annotation_indices("baseline")
                if not baseline: messagebox.showwarning("No baseline annotation", "Select one baseline annotation in subpart 3 first."); return None
                window=self._parse_float_pair(self.baseline_tmin,self.baseline_tmax,"Baseline window")
                if window is None: return None
                onset,_,_=self._get_annotation_by_index(baseline[0]); events.append(("pre-stimulus",onset,window))
        if mode != "baseline":
            if self.stim_window_mode.get()=="pair":
                if not self.stim_pair_indices: messagebox.showwarning("No stimulation pairs", "Choose one or more annotation pairs in section 4."); return None
                for n,pair in enumerate(self.stim_pair_indices,1):
                    try: onset,window,label=self._annotation_pair_window(pair)
                    except ValueError as exc: messagebox.showerror("Invalid pair",str(exc)); return None
                    events.append((f"segment {n}: {label}",onset,window))
            else:
                stim=self._selected_annotation_indices("stimulation")
                if not stim: messagebox.showwarning("No stimulation annotations", "Select one or more stimulation annotations in subpart 4 first."); return None
                window=self._parse_float_pair(self.stim_tmin,self.stim_tmax,"Stimulation window")
                if window is None: return None
                occurrences={}
                for idx in stim:
                    onset,_,desc=self._get_annotation_by_index(idx); occurrences[desc]=occurrences.get(desc,0)+1
                    label=desc if occurrences[desc]==1 else f"{desc}_{occurrences[desc]}"
                    events.append((label,onset,window))
        return events

    def _require_two_emg_channels(self) -> list[str] | None:
        if self.raw is None:
            messagebox.showwarning("No EDF", "EDF has not been loaded yet.")
            return None
        if len(self.selected_channels) != 2:
            messagebox.showwarning("Exactly two EMG channels required", "Step 5b/5c requires exactly two anatomically opposing surface-EMG channels.")
            return None
        return list(self.selected_channels)

    def _session_p95_rms(self, channels: list[str], rms_window_samples: int) -> dict[str, float]:
        """Compute per-channel session-wide P95 RMS in µV using the current bandpass."""
        low, high = self._get_bandpass() or (None, None)
        if low is None:
            raise RuntimeError("Invalid bandpass settings.")
        key = (tuple(channels), low, high, rms_window_samples)
        if key in self._session_p95_cache:
            return self._session_p95_cache[key]
        segment = self.raw.copy().pick(channels)
        segment.load_data(verbose="ERROR")
        segment.filter(l_freq=low, h_freq=high, picks=channels, verbose="ERROR")
        values: dict[str, float] = {}
        for ch, trace in zip(channels, segment.get_data()):
            p95 = float(np.nanpercentile(self._moving_rms(trace * 1_000_000.0, rms_window_samples), 95))
            if not np.isfinite(p95) or p95 <= 0:
                raise RuntimeError(f"Session-wide P95 RMS is invalid for {ch}.")
            values[ch] = p95
        self._session_p95_cache[key] = values
        return values

    def _extract_pair(self, onset: float, window: tuple[float, float], channels: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        t1, x1, _ = self._extract_preview_data(onset, window, channels[0])
        t2, x2, _ = self._extract_preview_data(onset, window, channels[1])
        n = min(len(t1), len(t2), len(x1), len(x2))
        if n < 2:
            raise RuntimeError("Matched event window has fewer than two samples.")
        return t1[:n], x1[:n], x2[:n]

    def run_cocontraction_analysis(self) -> None:
        if Figure is None or FigureCanvasTkAgg is None:
            messagebox.showerror("Missing dependency", f"Matplotlib is required for analysis figures:\n{MPL_IMPORT_ERROR}")
            return
        channels = self._require_two_emg_channels()
        if channels is None:
            return
        events = self._analysis_events()
        if not events:
            return
        rms_n = self._parse_rms_window_samples()
        if rms_n is None:
            return
        try:
            threshold = float(self.cci_threshold_pct.get()) / 100.0
            if not 0 <= threshold <= 1:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid input", "Simultaneous-activation threshold must be between 0 and 100 percent.")
            return
        try:
            p95 = self._session_p95_rms(channels, rms_n)
        except Exception as exc:
            messagebox.showerror("Normalization error", str(exc))
            return
        results, errors = [], []
        for label, onset, window in events:
            try:
                times, x1, x2 = self._extract_pair(onset, window, channels)
                e1 = self._moving_rms(x1 * 1_000_000.0, rms_n) / p95[channels[0]]
                e2 = self._moving_rms(x2 * 1_000_000.0, rms_n) / p95[channels[1]]
                overlap = np.minimum(e1, e2)
                denom = float(np.sum(e1) + np.sum(e2))
                cci = 100.0 * (2.0 * float(np.sum(overlap)) / denom) if denom > 0 else float("nan")
                results.append((label, times, e1, e2, cci, float(np.mean(overlap)), 100.0 * float(np.mean((e1 >= threshold) & (e2 >= threshold)))))
            except Exception as exc:
                errors.append(f"{label}: {exc}")
        if not results:
            messagebox.showerror("5b analysis error", "No event windows could be processed:\n" + "\n".join(errors[:8]))
            return
        fig = Figure(figsize=(10, max(6, 2.4 * len(results) + 2)), dpi=120)
        for i, (label, times, e1, e2, _cci, _common, _pct) in enumerate(results):
            ax = fig.add_subplot(len(results) + 1, 1, i + 1)
            ax.plot(times, e1, label=channels[0], lw=0.9)
            ax.plot(times, e2, label=channels[1], lw=0.9)
            ax.fill_between(times, 0, np.minimum(e1, e2), color="purple", alpha=.20, label="common activation")
            ax.set_ylabel("P95 norm.")
            ax.set_title(label, loc="left", fontsize=9)
            ax.grid(alpha=.25)
            if i == 0: ax.legend(loc="upper right", fontsize=8)
        ax = fig.add_subplot(len(results) + 1, 1, len(results) + 1)
        labels = [r[0] for r in results]
        cci = [r[4] for r in results]; common = [r[5] for r in results]; simultaneous = [r[6] for r in results]
        pos = np.arange(len(results)); width=.25
        ax.bar(pos-width, cci, width, label="CCI (%)"); ax.bar(pos, common, width, label="Common activation"); ax.bar(pos+width, simultaneous, width, label="Simultaneous (%)")
        ax.set_xticks(pos, labels, rotation=45, ha="right"); ax.grid(axis="y", alpha=.25); ax.legend(fontsize=8)
        fig.subplots_adjust(left=.10, right=.98, bottom=.12, top=.96, hspace=.75)
        subtitle = (f"5b: {channels[0]} vs {channels[1]}; session-wide P95 RMS normalization; "
                    f"CCI = 2Σmin(E1,E2)/Σ(E1+E2) × 100; threshold {threshold * 100:g}% P95")
        self._show_figure(
            fig,
            subtitle,
            window_title="EDF EMG co-contraction analysis",
            default_png_name=self._default_analysis_png_name("EMG_CCI", analysis_type="cocontraction"),
            replace_existing=False,
        )
        if errors: messagebox.showwarning("Some 5b windows skipped", "\n".join(errors[:8]))

    def run_coherence_analysis(self) -> None:
        if coherence is None:
            messagebox.showerror("Missing dependency", f"SciPy is required for Step 5c:\n{SCIPY_IMPORT_ERROR}\n\nInstall: pip install scipy")
            return
        if Figure is None or FigureCanvasTkAgg is None:
            messagebox.showerror("Missing dependency", f"Matplotlib is required for analysis figures:\n{MPL_IMPORT_ERROR}")
            return
        channels = self._require_two_emg_channels()
        if channels is None: return
        events = self._analysis_events()
        if not events: return
        band = self._parse_float_pair(self.coherence_low_hz, self.coherence_high_hz, "Coherence summary band")
        if band is None: return
        try:
            segment_s = float(self.coherence_segment_s.get())
            if segment_s <= 0: raise ValueError
        except ValueError:
            messagebox.showerror("Invalid input", "Welch segment length must be greater than zero seconds."); return
        sfreq = float(self.raw.info["sfreq"]); nperseg = max(2, int(round(segment_s * sfreq)))
        if band[1] >= sfreq / 2:
            messagebox.showerror("Invalid input", "Coherence summary band must be below Nyquist."); return
        results, errors = [], []
        for label, onset, window in events:
            try:
                _times, x1, x2 = self._extract_pair(onset, window, channels)
                n = min(len(x1), len(x2), nperseg)
                if n < 4: raise RuntimeError("Window is too short for coherence estimation.")
                freq, coh = coherence(x1, x2, fs=sfreq, nperseg=n, detrend="constant")
                inband = (freq >= band[0]) & (freq <= band[1])
                if not np.any(inband): raise RuntimeError("No coherence frequency bins fall within the requested summary band.")
                results.append((label, freq, coh, float(np.mean(coh[inband]))))
            except Exception as exc: errors.append(f"{label}: {exc}")
        if not results:
            messagebox.showerror("5c analysis error", "No event windows could be processed:\n" + "\n".join(errors[:8])); return
        fig = Figure(figsize=(10, 7), dpi=120); ax1 = fig.add_subplot(211)
        for label, freq, coh, mean_coh in results: ax1.plot(freq, coh, lw=1, label=f"{label} (mean={mean_coh:.3f})")
        ax1.axvspan(*band, color="orange", alpha=.15, label="summary band"); ax1.set_xlim(0, max(band[1] * 1.5, 20)); ax1.set_ylim(0, 1); ax1.set_ylabel("Magnitude-squared coherence"); ax1.grid(alpha=.25); ax1.legend(fontsize=8)
        ax2 = fig.add_subplot(212); labels=[r[0] for r in results]; means=[r[3] for r in results]
        ax2.bar(np.arange(len(means)), means, color="#d9eaf7"); ax2.set_xticks(np.arange(len(means)), labels, rotation=45, ha="right"); ax2.set_ylim(0,1); ax2.set_ylabel(f"Mean coherence ({band[0]:g}-{band[1]:g} Hz)"); ax2.set_xlabel("Event window"); ax2.grid(axis="y", alpha=.25)
        fig.subplots_adjust(left=.10,right=.98,bottom=.16,top=.95,hspace=.40)
        subtitle=(f"5c: {channels[0]} vs {channels[1]}; bandpass {self.low_hz.get()}-{self.high_hz.get()} Hz; Welch {segment_s:g}-s segments; PMID 39700629-informed {band[0]:g}-{band[1]:g} Hz summary")
        self._show_figure(
            fig,
            subtitle,
            window_title="EDF intermuscular coherence analysis",
            default_png_name=self._default_analysis_png_name("EMG_coherence", analysis_type="coherence"),
            replace_existing=False,
        )
        if errors: messagebox.showwarning("Some 5c windows skipped", "\n".join(errors[:8]))

    # ------------------------- Signal analysis -------------------------
    def run_psd_analysis(self) -> None:
        """Plot Welch PSDs for all selected channels in each selected event window."""
        if welch is None:
            messagebox.showerror("Missing dependency", f"SciPy is required for Step 5d:\n{SCIPY_IMPORT_ERROR}\n\nInstall: pip install scipy")
            return
        if Figure is None or FigureCanvasTkAgg is None:
            messagebox.showerror("Missing dependency", f"Matplotlib is required for analysis figures:\n{MPL_IMPORT_ERROR}")
            return
        if not self.selected_channels:
            messagebox.showwarning("No channels selected", "Select one or more channels in subpart 1 first.")
            return
        events = self._analysis_events()
        if not events:
            return
        try:
            segment_s = float(self.psd_segment_s.get())
            if segment_s <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid input", "Welch segment length must be greater than zero seconds.")
            return
        try:
            xmin = float(self.psd_xmin_hz.get())
            xmax = float(self.psd_xmax_hz.get())
            if xmin < 0 or xmax <= xmin:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid input", "PSD x-axis limits must be numeric, with 0 ≤ minimum < maximum.")
            return
        sfreq = float(self.raw.info["sfreq"])
        if xmax > sfreq / 2.0:
            messagebox.showerror("Invalid input", f"PSD x-axis maximum must not exceed the Nyquist frequency ({sfreq / 2.0:g} Hz).")
            return
        requested_nperseg = max(2, int(round(segment_s * sfreq)))
        _low, high = self._get_bandpass() or (None, None)
        results, errors = [], []
        for label, onset, window in events:
            spectra = []
            for channel in self.selected_channels:
                try:
                    _times, data_volts, _window_label = self._extract_raw_window_data(onset, window, channel)
                    nperseg = min(len(data_volts), requested_nperseg)
                    if nperseg < 4:
                        raise RuntimeError("Window is too short for Welch PSD estimation.")
                    noverlap = int(round(0.50 * nperseg))
                    freq, power_v2_hz = welch(data_volts, fs=sfreq, nperseg=nperseg, noverlap=noverlap, detrend="constant")
                    power_db = 10.0 * np.log10(np.maximum(power_v2_hz * 1_000_000_000_000.0, np.finfo(float).tiny))
                    spectra.append((channel, freq, power_db, nperseg))
                except Exception as exc:
                    errors.append(f"{label}, {channel}: {exc}")
            if spectra:
                results.append((label, spectra))
        if not results:
            messagebox.showerror("5d analysis error", "No event windows could be processed:\n" + "\n".join(errors[:8]))
            return
        # Use identical y-limits in every panel.  Derive them from all displayed PSD
        # samples, rather than allowing Matplotlib to autoscale each event separately.
        # The lower bound is padded slightly below the most negative PSD value so no
        # trace is clipped while retaining a scale appropriate to this analysis run.
        displayed_power = np.concatenate([
            power_db[np.isfinite(power_db) & (freq >= xmin) & (freq <= xmax)]
            for _label, spectra in results
            for _channel, freq, power_db, _nperseg in spectra
        ])
        global_min = float(np.min(displayed_power))
        global_max = float(np.max(displayed_power))
        data_range = global_max - global_min
        # A constant fallback retains visible limits if a PSD is numerically flat.
        y_padding = max(0.5, 0.02 * data_range)
        common_ymin = global_min - y_padding
        common_ymax = global_max + y_padding

        fig = Figure(figsize=(10, max(4, 2.7 * len(results))), dpi=120)
        for index, (label, spectra) in enumerate(results, start=1):
            ax = fig.add_subplot(len(results), 1, index)
            for channel, freq, power_db, _nperseg in spectra:
                ax.plot(freq, power_db, label=channel, lw=1.0)
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(common_ymin, common_ymax)
            ax.set_ylabel("PSD (dB re 1 µV²/Hz)")
            ax.set_title(label, loc="left", fontsize=9)
            ax.grid(alpha=.25)
            ax.legend(loc="best", fontsize=8)
            if index == len(results):
                ax.set_xlabel("Frequency (Hz)")
        actual_segments = sorted({nperseg for _label, spectra in results for _ch, _freq, _power, nperseg in spectra})
        segment_note = ", ".join(f"{n / sfreq:g} s" for n in actual_segments)
        fig.subplots_adjust(left=.11, right=.98, bottom=.10, top=.95, hspace=.55)
        subtitle = (f"5d: Welch PSD of raw signal; 60 Hz notch {'on' if self.notch_60_enabled.get() else 'off'}; "
                    f"requested segment {segment_s:g} s; used segment(s) {segment_note}")
        self._show_figure(
            fig,
            subtitle,
            window_title="EDF Welch spectral density analysis",
            default_png_name=self._default_analysis_png_name("EMG", analysis_type="welch_psd"),
            replace_existing=False,
        )
        if errors:
            messagebox.showwarning("Some 5d traces skipped", "\n".join(errors[:8]))

    def run_signal_analysis(self) -> None:
        if Figure is None or FigureCanvasTkAgg is None:
            messagebox.showerror("Missing dependency", f"Matplotlib is required for analysis figures:\n{MPL_IMPORT_ERROR}")
            return
        if self.raw is None:
            messagebox.showwarning("No EDF", "EDF has not been loaded yet.")
            return
        if not self.selected_channels:
            messagebox.showwarning("No channel selected", "Choose one or more channels of interest first.")
            return

        mode = self.analysis_mode.get()
        events = self._analysis_events()
        if events is None:
            return
        rms_window_samples = self._parse_rms_window_samples()
        if rms_window_samples is None:
            return

        all_errors: list[str] = []
        figures_made = 0
        for channel in self.selected_channels:
            labels: list[str] = []
            datasets: list[np.ndarray] = []
            errors: list[str] = []

            for label, onset, window in events:
                try:
                    datasets.append(self._extract_analysis_values(onset, window, rms_window_samples, channel))
                    labels.append(label)
                except Exception as exc:
                    errors.append(f"{channel} {label}: {exc}")

            if not datasets:
                all_errors.extend(errors)
                continue

            fig_width = max(8, 1.4 * len(datasets) + 3)
            fig = Figure(figsize=(fig_width, 5.5), dpi=120)
            ax = fig.add_subplot(111)
            bp = ax.boxplot(
                datasets,
                tick_labels=labels,
                showfliers=True,
                patch_artist=True,
                flierprops={"marker": "o", "markersize": 2, "markerfacecolor": "none", "markeredgewidth": 0.6},
            )
            for patch in bp.get("boxes", []):
                patch.set_facecolor("#d9eaf7")
            ax.set_ylabel("Amplitude (µV)")
            ax.set_xlabel("Event")
            title = self.custom_plot_titles.get(channel, "").strip() or self.custom_plot_title.get().strip() or f"{channel} signal analysis"
            ax.set_title(title)
            ax.grid(True, axis="y", alpha=0.25)
            ax.tick_params(axis="x", labelrotation=90)
            # Reserve explicit figure margins for the vertical event labels,
            # x-axis label, and title. These margins remain valid when the
            # display is zoomed, provided the viewer does not shrink the
            # figure below its readable minimum size (enforced in _show_figure).
            fig.subplots_adjust(left=0.16, right=0.98, bottom=0.34, top=0.90)

            subtitle = (
                f"Channel: {channel}; Bandpass {self.low_hz.get()}-{self.high_hz.get()} Hz; "
                f"RMS window {self.rms_window_ms.get()} ms; values rectified and plotted in µV"
            )
            figures_made += 1
            self._show_figure(
                fig,
                subtitle,
                window_title=f"EDF signal analysis - {channel}",
                default_png_name=self._default_analysis_png_name(channel),
                replace_existing=False,
            )
            all_errors.extend(errors)

        if figures_made == 0:
            messagebox.showerror("Analysis error", "No analysis windows could be processed:\n" + "\n".join(all_errors[:8]))
            return
        if all_errors:
            messagebox.showwarning("Some analysis windows skipped", "Some windows could not be analyzed:\n" + "\n".join(all_errors[:8]))

    def _default_analysis_png_name(self, channel: str | None = None, analysis_type: str = "boxwhisker") -> str:
        """Return a descriptive default name for a manually saved analysis figure."""
        channel = (channel or (self.selected_channels[0] if self.selected_channels else "channel")).strip() or "channel"
        safe_channel = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in channel).strip("_") or "channel"
        safe_type = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in analysis_type).strip("_") or "analysis"
        return f"{Path(self.edf_path).stem}_{safe_channel}_{safe_type}.png"

    # ------------------------- Preview actions -------------------------
    def preview_baseline(self) -> None:
        if self.analysis_mode.get()=="stimulation":
            messagebox.showinfo("Ignored","Baseline settings are ignored for stimulation-only mode."); return
        events=self._analysis_events()
        if events: self._preview_events([e for e in events if e[0].startswith(("pre-stimulus","baseline:"))],"Baseline preview")

    def preview_stimulation(self) -> None:
        if self.analysis_mode.get()=="baseline":
            messagebox.showinfo("Ignored","Stimulation settings are ignored for baseline-only mode."); return
        events=self._analysis_events()
        if events: self._preview_events([e for e in events if not e[0].startswith(("pre-stimulus","baseline:"))],"Stimulation preview")

    def _preview_events(self, events: list[tuple[str,float,tuple[float,float]]], title: str) -> None:
        if not events: return
        fig=Figure(figsize=(10,max(4,2.2*len(events))),dpi=100); errors=[]; plotted=0
        for n,(label,onset,window) in enumerate(events,1):
            try: times,data,window_label=self._extract_preview_data(onset,window)
            except Exception as exc: errors.append(f"{label}: {exc}"); continue
            plotted+=1; ax=fig.add_subplot(len(events),1,n); ax.plot(times,data,lw=.7); ax.set_title(f"{label} ({window_label})",fontsize=9)
            ax.set_ylabel(self.selected_channels[0] if self.selected_channels else "",fontsize=8); ax.grid(alpha=.25)
            if n==len(events): ax.set_xlabel("Time from segment start (s)")
        if not plotted:
            messagebox.showerror("Preview error","No selected windows could be previewed:\n"+"\n".join(errors[:5])); return
        fig.tight_layout(); self._show_figure(fig,title)
        if errors: messagebox.showwarning("Some previews skipped","\n".join(errors[:5]))

    def _annotations_in_window(self, anchor_onset: float, window: tuple[float, float]) -> list[tuple[int, float, float, str]]:
        """Return annotation onsets that fall inside a preview window, in anchor-relative seconds."""
        if self.raw is None:
            return []
        start_abs = anchor_onset + window[0]
        stop_abs = anchor_onset + window[1]
        start_abs, stop_abs = self._clip_window_to_data(start_abs, stop_abs)
        rows: list[tuple[int, float, float, str]] = []
        for idx, onset, duration, desc in self._annotation_rows():
            if start_abs <= onset <= stop_abs:
                rows.append((idx, onset - anchor_onset, duration, desc))
        return rows

    def _add_annotation_markers(
        self,
        ax,
        anchor_idx: int,
        anchor_onset: float,
        window: tuple[float, float],
        *,
        show_text: bool,
    ) -> None:
        """Draw selected and non-selected annotation positions on a preview axis."""
        markers = self._annotations_in_window(anchor_onset, window)
        ymin, ymax = ax.get_ylim()
        text_y = ymax - 0.04 * (ymax - ymin)
        first_anchor = True
        first_other = True
        for idx, rel_onset, _duration, desc in markers:
            if idx == anchor_idx:
                ax.axvline(
                    rel_onset,
                    color="red",
                    linestyle="--",
                    linewidth=1.1,
                    label=("selected annotation" if first_anchor else None),
                )
                if show_text:
                    short_desc = desc if len(desc) <= 24 else desc[:21] + "..."
                    ax.text(
                        rel_onset,
                        text_y,
                        f"#{idx} {short_desc}",
                        rotation=90,
                        va="top",
                        ha="right",
                        fontsize=7,
                        color="red",
                        alpha=0.9,
                    )
                first_anchor = False
            else:
                ax.axvline(
                    rel_onset,
                    color="purple",
                    linestyle=":",
                    linewidth=0.9,
                    alpha=0.85,
                    label=("other annotation(s)" if first_other else None),
                )
                first_other = False
                if show_text:
                    short_desc = desc if len(desc) <= 24 else desc[:21] + "..."
                    ax.text(
                        rel_onset,
                        text_y,
                        f"#{idx} {short_desc}",
                        rotation=90,
                        va="top",
                        ha="right",
                        fontsize=7,
                        color="purple",
                        alpha=0.9,
                    )
        # If no annotation row was found exactly at the selected onset, still show the anchor line.
        if first_anchor:
            ax.axvline(0, color="red", linestyle="--", linewidth=1.1, label="selected annotation")

    def _preview_single(self, selected_idx: int, onset: float, window: tuple[float, float], title: str) -> None:
        try:
            times, data, window_label = self._extract_preview_data(onset, window)
        except Exception as exc:
            messagebox.showerror("Preview error", f"Could not preview window:\n{exc}")
            return

        fig = Figure(figsize=(9, 4), dpi=100)
        ax = fig.add_subplot(111)
        ax.plot(times, data, linewidth=0.8)
        self._add_annotation_markers(ax, selected_idx, onset, window, show_text=True)
        ax.set_xlim(window[0], window[1])
        ax.set_title(title)
        ax.set_xlabel("Time relative to selected annotation (s)")
        ax.set_ylabel(self.selected_channels[0] if self.selected_channels else "")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right")
        fig.tight_layout()
        self._show_figure(fig, f"{title}\n{window_label}")

    def _preview_multiple(self, indices: list[int], window: tuple[float, float], title: str) -> None:
        if Figure is None or FigureCanvasTkAgg is None:
            messagebox.showerror("Missing dependency", f"Matplotlib is required for previews:\n{MPL_IMPORT_ERROR}")
            return

        fig = Figure(figsize=(10, max(4, 1.8 * len(indices))), dpi=100)
        plotted = 0
        errors = []
        for plot_i, idx in enumerate(indices, start=1):
            onset, _duration, desc = self._get_annotation_by_index(idx)
            try:
                times, data, window_label = self._extract_preview_data(onset, window)
            except Exception as exc:
                errors.append(f"{idx}: {exc}")
                continue
            plotted += 1
            ax = fig.add_subplot(len(indices), 1, plot_i)
            ax.plot(times, data, linewidth=0.7)
            # Label every event marker, including the selected anchor event.
            self._add_annotation_markers(ax, idx, onset, window, show_text=True)
            ax.set_xlim(window[0], window[1])
            ax.set_title(f"#{idx}: {desc} ({window_label})", fontsize=9)
            ax.set_ylabel(self.selected_channels[0] if self.selected_channels else "", fontsize=8)
            ax.grid(True, alpha=0.25)
            if plot_i == len(indices):
                ax.set_xlabel("Time relative to selected annotation (s)")
        fig.suptitle(title)
        fig.tight_layout()

        if plotted == 0:
            messagebox.showerror("Preview error", "No selected windows could be previewed:\n" + "\n".join(errors[:5]))
            return
        if errors:
            messagebox.showwarning("Some previews skipped", "Some selected windows could not be previewed:\n" + "\n".join(errors[:5]))
        self._show_figure(fig, title)

    def _show_figure(
        self,
        fig: Figure,
        title: str,
        *,
        window_title: str = "EDF window preview",
        default_png_name: str | None = None,
        replace_existing: bool = True,
    ) -> None:
        if FigureCanvasTkAgg is None:
            messagebox.showerror("Missing dependency", f"Matplotlib is required for figures:\n{MPL_IMPORT_ERROR}")
            return

        if replace_existing and self.preview_window is not None and self.preview_window.winfo_exists():
            self.preview_window.destroy()

        fig_window = tk.Toplevel(self.root)
        self.preview_window = fig_window
        fig_window.title(window_title)
        fig_window.geometry("1100x700")

        ttk.Label(fig_window, text=title, wraplength=1050).pack(anchor=tk.W, padx=10, pady=(8, 2))

        # Matplotlib figures with many stacked subplots can be taller than the
        # preview window. Put the FigureCanvasTkAgg widget inside a scrollable
        # Tk canvas so the user can scroll down to every subplot.
        scroll_frame = ttk.Frame(fig_window)
        scroll_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        tk_scroll_canvas = tk.Canvas(scroll_frame, highlightthickness=0)
        y_scrollbar = ttk.Scrollbar(scroll_frame, orient=tk.VERTICAL, command=tk_scroll_canvas.yview)
        x_scrollbar = ttk.Scrollbar(scroll_frame, orient=tk.HORIZONTAL, command=tk_scroll_canvas.xview)
        tk_scroll_canvas.configure(yscrollcommand=y_scrollbar.set, xscrollcommand=x_scrollbar.set)
        tk_scroll_canvas.grid(row=0, column=0, sticky="nsew")
        y_scrollbar.grid(row=0, column=1, sticky="ns")
        x_scrollbar.grid(row=1, column=0, sticky="ew")
        scroll_frame.rowconfigure(0, weight=1)
        scroll_frame.columnconfigure(0, weight=1)

        canvas = FigureCanvasTkAgg(fig, master=tk_scroll_canvas)
        canvas.draw()
        figure_widget = canvas.get_tk_widget()
        canvas_window_id = tk_scroll_canvas.create_window((0, 0), window=figure_widget, anchor=tk.NW)

        def _update_scroll_region(_event=None) -> None:
            tk_scroll_canvas.configure(scrollregion=tk_scroll_canvas.bbox(tk.ALL))
            # Preserve the figure's natural dimensions; scrollbars expose any
            # portion that does not fit in the pop-up window.
            tk_scroll_canvas.itemconfigure(
                canvas_window_id,
                width=figure_widget.winfo_reqwidth(),
                height=figure_widget.winfo_reqheight(),
            )

        def _on_mousewheel(event) -> None:
            if event.num == 4:  # Linux scroll up
                delta = -1
            elif event.num == 5:  # Linux scroll down
                delta = 1
            else:
                delta = int(-1 * (event.delta / 120))
            tk_scroll_canvas.yview_scroll(delta, "units")

        figure_widget.bind("<Configure>", _update_scroll_region)
        tk_scroll_canvas.bind("<Configure>", _update_scroll_region)
        for widget in (tk_scroll_canvas, figure_widget):
            widget.bind("<MouseWheel>", _on_mousewheel)
            widget.bind("<Button-4>", _on_mousewheel)
            widget.bind("<Button-5>", _on_mousewheel)
        _update_scroll_region()

        button_frame = ttk.Frame(fig_window)
        button_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        def save_png() -> None:
            initialfile = default_png_name or "edf_figure.png"
            out_path = filedialog.asksaveasfilename(
                title="Save PNG figure",
                defaultextension=".png",
                initialfile=initialfile,
                filetypes=[("PNG image", "*.png"), ("All files", "*.*")],
                parent=fig_window,
            )
            if not out_path:
                return
            fig.savefig(out_path, dpi=300, bbox_inches="tight")
            self.status.set(f"Saved PNG figure: {out_path}")
            messagebox.showinfo("Saved", f"Saved PNG figure:\n{out_path}", parent=fig_window)

        original_size = tuple(fig.get_size_inches())
        # A Matplotlib figure can technically be made arbitrarily small, but
        # below this size its fixed-size title and axis-label artists no longer
        # fit within the available renderer area and are clipped. Keep enough
        # physical canvas area for those artists while still permitting useful
        # zoom-out from large multi-event analysis figures.
        min_zoom_size = (5.0, 4.0)

        def _redraw_after_resize() -> None:
            """Resize the Tk widget before redrawing to prevent stale canvas pixels."""
            # FigureCanvasTkAgg does not always update its Tk widget's dimensions
            # after a programmatic figure resize. Explicitly resizing the widget
            # clears the old backing buffer, preventing an old larger plot from
            # remaining visible behind a zoomed-out plot.
            pixel_width, pixel_height = canvas.get_width_height(physical=True)
            figure_widget.configure(width=pixel_width, height=pixel_height)
            canvas.draw()
            figure_widget.update_idletasks()
            _update_scroll_region()

        def set_zoom(factor: float) -> None:
            width, height = fig.get_size_inches()
            target_width = max(min_zoom_size[0], width * factor)
            target_height = max(min_zoom_size[1], height * factor)
            if (target_width, target_height) == (width, height):
                return
            fig.set_size_inches(target_width, target_height, forward=True)
            _redraw_after_resize()

        def reset_zoom() -> None:
            fig.set_size_inches(*original_size, forward=True)
            _redraw_after_resize()

        ttk.Button(button_frame, text="Save PNG...", command=save_png).pack(side=tk.RIGHT)
        ttk.Button(button_frame, text="Reset zoom", command=reset_zoom).pack(side=tk.LEFT)
        ttk.Button(button_frame, text="Zoom out −", command=lambda: set_zoom(0.8)).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(button_frame, text="Zoom in +", command=lambda: set_zoom(1.25)).pack(side=tk.LEFT, padx=(6, 0))
        if NavigationToolbar2Tk is not None:
            toolbar = NavigationToolbar2Tk(canvas, button_frame, pack_toolbar=False)
            toolbar.update()
            toolbar.pack(side=tk.LEFT, padx=(14, 0))

    # ------------------------- Config/save -------------------------
    def _update_enabled_sections(self) -> None:
        mode = self.analysis_mode.get()
        self._set_frame_enabled(self.baseline_frame, mode != "stimulation")
        self._set_frame_enabled(self.stim_frame, mode != "baseline")

    def _set_frame_enabled(self, frame: ttk.Frame, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        for child in frame.winfo_children():
            self._set_widget_enabled_recursive(child, state)

    def _set_widget_enabled_recursive(self, widget: tk.Widget, state: str) -> None:
        try:
            widget.configure(state=state)
        except tk.TclError:
            pass
        for child in widget.winfo_children():
            self._set_widget_enabled_recursive(child, state)

    def build_config(self) -> AnalysisConfig | None:
        bp = self._get_bandpass()
        if bp is None:
            return None
        mode = self.analysis_mode.get()

        baseline_annotation = None; baseline_window = None; baseline_pair=[]
        if mode in ("baseline", "baseline_and_stimulation"):
            if self.baseline_window_mode.get()=="pair":
                baseline_pair=[self._get_annotation_by_index(i)[2] for i in self.baseline_pair_indices]
            else:
                selection=self._selected_annotation_indices("baseline")
                if selection:
                    _o,_d,baseline_annotation=self._get_annotation_by_index(selection[0])
                    baseline_window=self._parse_float_pair(self.baseline_tmin,self.baseline_tmax,"Baseline window")
        stim_annotations=[]; stim_window=None; stim_pairs=[]
        if mode in ("stimulation", "baseline_and_stimulation"):
            if self.stim_window_mode.get()=="pair":
                stim_pairs=[[self._get_annotation_by_index(i)[2] for i in pair] for pair in self.stim_pair_indices]
            else:
                for idx in self._selected_annotation_indices("stimulation"):
                    stim_annotations.append(self._get_annotation_by_index(idx)[2])
                if stim_annotations: stim_window=self._parse_float_pair(self.stim_tmin,self.stim_tmax,"Stimulation window")

        return AnalysisConfig(
            edf_path=self.edf_path,
            channels=list(self.selected_channels),
            bandpass_low_hz=bp[0],
            bandpass_high_hz=bp[1],
            analysis_mode=mode,
            baseline_annotation=baseline_annotation,
            baseline_window_s=baseline_window,
            stimulation_annotations=stim_annotations,
            stimulation_window_s=stim_window,
            baseline_boundary_annotations=baseline_pair or None,
            stimulation_boundary_pairs=stim_pairs or None,
            baseline_window_mode=self.baseline_window_mode.get(),
            stimulation_window_mode=self.stim_window_mode.get(),
        )

    def save_config(self) -> None:
        config = self.build_config()
        if config is None:
            return
        config_dict = asdict(config)
        config_json = json.dumps(config_dict, indent=2)
        print(config_json)

        default_name = Path(self.edf_path).with_suffix(".analysis_setup.json").name
        out_path = filedialog.asksaveasfilename(
            title="Save current setup as JSON",
            defaultextension=".json",
            initialfile=default_name,
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            parent=self.root,
        )
        if out_path:
            Path(out_path).write_text(config_json + "\n", encoding="utf-8")
            self.status.set(f"Saved setup: {out_path}")
            messagebox.showinfo("Saved", f"Saved setup JSON:\n{out_path}")


def choose_edf_file() -> str | None:
    """First GUI: choose EDF file."""
    root = tk.Tk()
    root.withdraw()
    root.update()
    path = filedialog.askopenfilename(
        title="Choose EDF file",
        filetypes=[("EDF files", "*.edf *.EDF"), ("All files", "*.*")],
    )
    root.destroy()
    return path or None


def main() -> int:
    if MNE_IMPORT_ERROR is not None:
        print("ERROR: mne is required to run this GUI.", file=sys.stderr)
        print(f"Import error: {MNE_IMPORT_ERROR}", file=sys.stderr)
        print("Try: pip install mne matplotlib numpy", file=sys.stderr)
        return 1

    try:
        edf_path = choose_edf_file()
        if not edf_path:
            print("No EDF file selected. Exiting.")
            return 0

        root = tk.Tk()
        EDFAnalysisSetupGUI(root, edf_path)
        root.mainloop()
        return 0
    except Exception:
        traceback.print_exc()
        try:
            messagebox.showerror("Unexpected error", traceback.format_exc())
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
