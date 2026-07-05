#!/usr/bin/env python3
"""
Interactive EDF analysis setup GUI.

Current scope:
1. Open a file chooser to select an EDF file.
2. Open a second GUI with four setup subparts:
   - Channel + bandpass filter limits
   - Analysis type: baseline, stimulation, or both
   - Baseline annotation/window selection and preview
   - Stimulation annotation/window selection and preview
   - Final signal analysis: bandpass, RMS smoothing, rectification, and box-whisker plot

Dependencies:
    pip install mne matplotlib numpy

Recommended run from this project:
    python outputs/analyze_edf_gui.py
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
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
except ImportError as exc:  # handled in preview
    matplotlib = None
    FigureCanvasTkAgg = None
    Figure = None
    MPL_IMPORT_ERROR = exc
else:
    MPL_IMPORT_ERROR = None


@dataclass
class AnalysisConfig:
    edf_path: str
    channel: str | None
    bandpass_low_hz: float
    bandpass_high_hz: float
    analysis_mode: str
    baseline_annotation: str | None
    baseline_window_s: tuple[float, float] | None
    stimulation_annotations: list[str]
    stimulation_window_s: tuple[float, float] | None


class EDFAnalysisSetupGUI:
    """Second-stage EDF setup GUI."""

    def __init__(self, root: tk.Tk, edf_path: str):
        self.root = root
        self.edf_path = str(edf_path)
        self.raw = None
        self.preview_window: tk.Toplevel | None = None

        self.selected_channel = tk.StringVar(value="")
        self.low_hz = tk.StringVar(value="20")
        self.high_hz = tk.StringVar(value="450")
        self.analysis_mode = tk.StringVar(value="baseline_and_stimulation")
        self.baseline_tmin = tk.StringVar(value="-45")
        self.baseline_tmax = tk.StringVar(value="-1")
        self.stim_tmin = tk.StringVar(value="0")
        self.stim_tmax = tk.StringVar(value="60")
        self.rms_window_ms = tk.StringVar(value="100")
        self.custom_plot_title = tk.StringVar(value="")
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

        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill=tk.BOTH, expand=True)

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
        ttk.Label(channel_frame, textvariable=self.selected_channel, width=45).grid(
            row=0, column=1, padx=(0, 16), sticky=tk.W
        )

        ttk.Label(channel_frame, text="Bandpass low Hz:").grid(row=0, column=2, padx=(0, 4))
        ttk.Entry(channel_frame, textvariable=self.low_hz, width=8).grid(row=0, column=3, padx=(0, 12))
        ttk.Label(channel_frame, text="Bandpass high Hz:").grid(row=0, column=4, padx=(0, 4))
        ttk.Entry(channel_frame, textvariable=self.high_hz, width=8).grid(row=0, column=5)

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

        self.analysis_frame = ttk.LabelFrame(outer, text="5. Final signal analysis", padding=10)
        self.analysis_frame.pack(fill=tk.X, pady=5)
        self._build_analysis_section(self.analysis_frame)

        # Footer
        footer = ttk.Frame(outer)
        footer.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(footer, textvariable=self.status).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(footer, text="Print/Save current setup", command=self.save_config).pack(side=tk.RIGHT, padx=(8, 0))
        ttk.Button(footer, text="Close", command=self.root.destroy).pack(side=tk.RIGHT)

    def _build_baseline_section(self, parent: ttk.Frame) -> None:
        top = ttk.Frame(parent)
        top.pack(fill=tk.X)

        ttk.Button(top, text="Choose baseline annotation...", command=self.choose_baseline_annotation).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Label(top, text="Window relative to annotation, seconds:").pack(side=tk.LEFT, padx=(8, 4))
        ttk.Entry(top, textvariable=self.baseline_tmin, width=8).pack(side=tk.LEFT)
        ttk.Label(top, text="to").pack(side=tk.LEFT, padx=4)
        ttk.Entry(top, textvariable=self.baseline_tmax, width=8).pack(side=tk.LEFT)
        ttk.Button(top, text="Preview baseline window", command=self.preview_baseline).pack(side=tk.LEFT, padx=12)

        help_text = "Select one baseline annotation. Example window: -45 to -1 seconds. Ignored for stimulation-only mode."
        ttk.Label(parent, text=help_text, foreground="gray30").pack(anchor=tk.W, pady=(4, 2))
        ttk.Label(parent, textvariable=self.baseline_annotation_display, wraplength=900).pack(anchor=tk.W, pady=(4, 0), fill=tk.X)

    def _build_stimulation_section(self, parent: ttk.Frame) -> None:
        top = ttk.Frame(parent)
        top.pack(fill=tk.X)

        ttk.Button(top, text="Choose stimulation annotation(s)...", command=self.choose_stim_annotations).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Label(top, text="Window relative to annotation, seconds:").pack(side=tk.LEFT, padx=(8, 4))
        ttk.Entry(top, textvariable=self.stim_tmin, width=8).pack(side=tk.LEFT)
        ttk.Label(top, text="to").pack(side=tk.LEFT, padx=4)
        ttk.Entry(top, textvariable=self.stim_tmax, width=8).pack(side=tk.LEFT)
        ttk.Button(top, text="Preview stimulation window(s)", command=self.preview_stimulation).pack(side=tk.LEFT, padx=12)

        help_text = "Select one or more stimulation annotations. Example window: 0 to +60 seconds. Ignored for baseline-only mode."
        ttk.Label(parent, text=help_text, foreground="gray30").pack(anchor=tk.W, pady=(4, 2))
        ttk.Label(parent, textvariable=self.stim_annotation_display, wraplength=900).pack(anchor=tk.W, pady=(4, 0), fill=tk.X)

    def _build_analysis_section(self, parent: ttk.Frame) -> None:
        top = ttk.Frame(parent)
        top.pack(fill=tk.X)
        ttk.Label(top, text="RMS smoothing window, ms:").pack(side=tk.LEFT, padx=(0, 4))
        ttk.Entry(top, textvariable=self.rms_window_ms, width=8).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Label(top, text="Custom plot title:").pack(side=tk.LEFT, padx=(0, 4))
        ttk.Entry(top, textvariable=self.custom_plot_title, width=32).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Button(top, text="Run signal analysis and show box-whisker plot", command=self.run_signal_analysis).pack(side=tk.LEFT)
        help_text = (
            "Extracts selected channel windows, applies the subpart 1 bandpass, computes a moving RMS envelope, "
            "rectifies it, and plots values in microvolts. First box: subpart 3 baseline ('pre-stimulus'); "
            "subsequent boxes: subpart 4 stimulation annotations."
        )
        ttk.Label(parent, text=help_text, foreground="gray30", wraplength=900).pack(anchor=tk.W, pady=(4, 0), fill=tk.X)

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
            self.selected_channel.set(self.raw.ch_names[0])

        self._update_annotation_displays()
        self._update_enabled_sections()

    # ------------------------- Helpers -------------------------
    def choose_channel(self) -> None:
        if self.raw is None:
            messagebox.showwarning("No EDF", "EDF has not been loaded yet.")
            return

        win = tk.Toplevel(self.root)
        win.title("Choose channel of interest")
        win.geometry("520x520")
        win.transient(self.root)
        win.grab_set()

        ttk.Label(win, text="Select one channel:").pack(anchor=tk.W, padx=10, pady=(10, 4))
        search_var = tk.StringVar(value="")
        search_entry = ttk.Entry(win, textvariable=search_var)
        search_entry.pack(fill=tk.X, padx=10, pady=(0, 6))

        list_frame = ttk.Frame(win)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)

        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL)
        listbox = tk.Listbox(
            list_frame,
            selectmode=tk.SINGLE,
            exportselection=False,
            yscrollcommand=scrollbar.set,
        )
        scrollbar.config(command=listbox.yview)
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        def refresh_list(*_args):
            query = search_var.get().lower().strip()
            listbox.delete(0, tk.END)
            for ch in self.raw.ch_names:
                if not query or query in ch.lower():
                    listbox.insert(tk.END, ch)

        def accept_selection():
            selection = listbox.curselection()
            if not selection:
                messagebox.showwarning("No channel selected", "Please select a channel.", parent=win)
                return
            self.selected_channel.set(listbox.get(selection[0]))
            win.destroy()

        search_var.trace_add("write", refresh_list)
        refresh_list()
        ttk.Button(win, text="Use selected channel", command=accept_selection).pack(pady=(2, 10))
        listbox.bind("<Double-Button-1>", lambda _event: accept_selection())
        search_entry.focus_set()

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

    def _extract_preview_data(self, onset_s: float, window: tuple[float, float]) -> tuple[np.ndarray, np.ndarray, str]:
        if self.raw is None:
            raise RuntimeError("EDF has not been loaded.")
        ch = self.selected_channel.get().strip()
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
        raw_segment.filter(l_freq=low, h_freq=high, picks=[ch], verbose="ERROR")
        data = raw_segment.get_data(picks=[ch])[0]
        times = raw_segment.times + start_s - onset_s  # relative to annotation onset
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

    def _extract_analysis_values(self, onset_s: float, window: tuple[float, float], rms_window_samples: int) -> np.ndarray:
        _times, data_volts, _window_label = self._extract_preview_data(onset_s, window)
        data_uv = data_volts * 1_000_000.0
        rms_uv = self._moving_rms(data_uv, rms_window_samples)
        rectified_uv = np.abs(rms_uv)
        rectified_uv = rectified_uv[np.isfinite(rectified_uv)]
        if rectified_uv.size == 0:
            raise RuntimeError("Analysis window contained no finite samples after processing.")
        return rectified_uv

    # ------------------------- Signal analysis -------------------------
    def run_signal_analysis(self) -> None:
        if Figure is None or FigureCanvasTkAgg is None:
            messagebox.showerror("Missing dependency", f"Matplotlib is required for analysis figures:\n{MPL_IMPORT_ERROR}")
            return
        if self.raw is None:
            messagebox.showwarning("No EDF", "EDF has not been loaded yet.")
            return
        if not self.selected_channel.get().strip():
            messagebox.showwarning("No channel selected", "Choose a channel of interest first.")
            return

        mode = self.analysis_mode.get()
        baseline_selection = self._selected_annotation_indices("baseline") if mode != "stimulation" else []
        stim_selection = self._selected_annotation_indices("stimulation") if mode != "baseline" else []

        if not baseline_selection:
            messagebox.showwarning("No baseline annotation", "Select one baseline annotation in subpart 3 first.")
            return
        if mode != "baseline" and not stim_selection:
            messagebox.showwarning("No stimulation annotations", "Select one or more stimulation annotations in subpart 4 first.")
            return

        baseline_window = self._parse_float_pair(self.baseline_tmin, self.baseline_tmax, "Baseline window")
        if baseline_window is None:
            return
        stim_window = None
        if stim_selection:
            stim_window = self._parse_float_pair(self.stim_tmin, self.stim_tmax, "Stimulation window")
            if stim_window is None:
                return
        rms_window_samples = self._parse_rms_window_samples()
        if rms_window_samples is None:
            return

        labels: list[str] = []
        datasets: list[np.ndarray] = []
        errors: list[str] = []

        baseline_idx = baseline_selection[0]
        baseline_onset, _duration, _desc = self._get_annotation_by_index(baseline_idx)
        try:
            datasets.append(self._extract_analysis_values(baseline_onset, baseline_window, rms_window_samples))
            labels.append("pre-stimulus")
        except Exception as exc:
            errors.append(f"baseline #{baseline_idx}: {exc}")

        annotation_occurrences: dict[str, int] = {}
        if stim_window is not None:
            for idx in stim_selection:
                onset, _duration, desc = self._get_annotation_by_index(idx)
                try:
                    datasets.append(self._extract_analysis_values(onset, stim_window, rms_window_samples))
                    annotation_occurrences[desc] = annotation_occurrences.get(desc, 0) + 1
                    occurrence = annotation_occurrences[desc]
                    labels.append(desc if occurrence == 1 else f"{desc}_{occurrence}")
                except Exception as exc:
                    errors.append(f"stimulation #{idx}: {exc}")

        if not datasets:
            messagebox.showerror("Analysis error", "No analysis windows could be processed:\n" + "\n".join(errors[:8]))
            return

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
        ax.set_title(self.custom_plot_title.get().strip())
        ax.grid(True, axis="y", alpha=0.25)
        ax.tick_params(axis="x", labelrotation=90)
        # Use explicit margins so the embedded Tk preview canvas does not clip
        # y-axis tick numbers/label or vertical x-axis labels. Saved PNGs use
        # bbox_inches="tight", but the on-screen canvas needs reserved space.
        fig.subplots_adjust(left=0.16, right=0.98, bottom=0.34, top=0.90)

        subtitle = (
            f"Bandpass {self.low_hz.get()}-{self.high_hz.get()} Hz; "
            f"RMS window {self.rms_window_ms.get()} ms; values rectified and plotted in µV"
        )
        if errors:
            messagebox.showwarning("Some analysis windows skipped", "Some windows could not be analyzed:\n" + "\n".join(errors[:8]))
        self._show_figure(fig, subtitle, window_title="EDF signal analysis", default_png_name=self._default_analysis_png_name())

    def _default_analysis_png_name(self) -> str:
        channel = self.selected_channel.get().strip() or "channel"
        safe_channel = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in channel).strip("_") or "channel"
        return f"{Path(self.edf_path).stem}_{safe_channel}_boxwhisker.png"

    # ------------------------- Preview actions -------------------------
    def preview_baseline(self) -> None:
        if self.analysis_mode.get() == "stimulation":
            messagebox.showinfo("Ignored", "Baseline settings are ignored for stimulation-only mode.")
            return
        selection = self._selected_annotation_indices("baseline")
        if not selection:
            messagebox.showwarning("No annotation selected", "Select one baseline annotation first.")
            return
        window = self._parse_float_pair(self.baseline_tmin, self.baseline_tmax, "Baseline window")
        if window is None:
            return
        idx = selection[0]
        onset, _duration, desc = self._get_annotation_by_index(idx)
        self._preview_single(idx, onset, window, f"Baseline preview: {desc}")

    def preview_stimulation(self) -> None:
        if self.analysis_mode.get() == "baseline":
            messagebox.showinfo("Ignored", "Stimulation settings are ignored for baseline-only mode.")
            return
        selection = self._selected_annotation_indices("stimulation")
        if not selection:
            messagebox.showwarning("No annotation selected", "Select one or more stimulation annotations first.")
            return
        window = self._parse_float_pair(self.stim_tmin, self.stim_tmax, "Stimulation window")
        if window is None:
            return

        # For many selected annotations, preview the first up to 8 in stacked subplots.
        self._preview_multiple(selection[:8], window, title=f"Stimulation preview ({min(len(selection), 8)} of {len(selection)} selected)")

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
        ax.set_ylabel(self.selected_channel.get())
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
            self._add_annotation_markers(ax, idx, onset, window, show_text=False)
            ax.set_xlim(window[0], window[1])
            ax.set_title(f"#{idx}: {desc} ({window_label})", fontsize=9)
            ax.set_ylabel(self.selected_channel.get(), fontsize=8)
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
    ) -> None:
        if FigureCanvasTkAgg is None:
            messagebox.showerror("Missing dependency", f"Matplotlib is required for figures:\n{MPL_IMPORT_ERROR}")
            return

        if self.preview_window is not None and self.preview_window.winfo_exists():
            self.preview_window.destroy()

        self.preview_window = tk.Toplevel(self.root)
        self.preview_window.title(window_title)
        self.preview_window.geometry("1100x700")

        ttk.Label(self.preview_window, text=title, wraplength=1050).pack(anchor=tk.W, padx=10, pady=(8, 2))
        canvas = FigureCanvasTkAgg(fig, master=self.preview_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        button_frame = ttk.Frame(self.preview_window)
        button_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        def save_png() -> None:
            initialfile = default_png_name or "edf_figure.png"
            out_path = filedialog.asksaveasfilename(
                title="Save PNG figure",
                defaultextension=".png",
                initialfile=initialfile,
                filetypes=[("PNG image", "*.png"), ("All files", "*.*")],
                parent=self.preview_window,
            )
            if not out_path:
                return
            fig.savefig(out_path, dpi=300, bbox_inches="tight")
            self.status.set(f"Saved PNG figure: {out_path}")
            messagebox.showinfo("Saved", f"Saved PNG figure:\n{out_path}", parent=self.preview_window)

        ttk.Button(button_frame, text="Save PNG...", command=save_png).pack(side=tk.RIGHT)

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

        baseline_annotation = None
        baseline_window = None
        if mode in ("baseline", "baseline_and_stimulation"):
            baseline_selection = self._selected_annotation_indices("baseline")
            if baseline_selection:
                _onset, _duration, baseline_annotation = self._get_annotation_by_index(baseline_selection[0])
                baseline_window = self._parse_float_pair(self.baseline_tmin, self.baseline_tmax, "Baseline window")

        stim_annotations: list[str] = []
        stim_window = None
        if mode in ("stimulation", "baseline_and_stimulation"):
            stim_selection = self._selected_annotation_indices("stimulation")
            for idx in stim_selection:
                _onset, _duration, desc = self._get_annotation_by_index(idx)
                stim_annotations.append(desc)
            if stim_selection:
                stim_window = self._parse_float_pair(self.stim_tmin, self.stim_tmax, "Stimulation window")

        return AnalysisConfig(
            edf_path=self.edf_path,
            channel=self.selected_channel.get().strip() or None,
            bandpass_low_hz=bp[0],
            bandpass_high_hz=bp[1],
            analysis_mode=mode,
            baseline_annotation=baseline_annotation,
            baseline_window_s=baseline_window,
            stimulation_annotations=stim_annotations,
            stimulation_window_s=stim_window,
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
