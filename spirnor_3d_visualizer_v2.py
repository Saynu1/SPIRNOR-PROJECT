#!/usr/bin/env python3
"""
SPIRNOR 3D Interactive Visualizer v2
=====================================
A polished, dark-themed interactive visualization tool for exploring the
SPIRNOR operator phenomenon in 3D space with prime/composite filtering,
mathematical constant presets, animation, and spectral analysis.

Usage: python spirnor_3d_visualizer_v2.py

Requirements:
- numpy
- matplotlib
- numba (optional, for performance)

Controls:
- C value slider: Adjust the winding constant
- Range slider: Control how many integers to visualize
- Preset buttons: Quick access to mathematical constants
- Checkboxes: Toggle primes, composites, trails, and coloring
- Animate: Auto-sweep through C values
- FFT Analysis: Frequency domain visualization
- 2D Projection: Classic spiral view
- Mouse: Rotate/zoom the 3D plot
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider, CheckButtons, Button, RadioButtons
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe
import warnings

warnings.filterwarnings("ignore")

# ── Numba (optional) ────────────────────────────────────────────────
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def njit(func):
        return func

# ── Theme colours ───────────────────────────────────────────────────
BG_DARK   = "#0d1117"
BG_PANEL  = "#161b22"
BG_WIDGET = "#21262d"
ACCENT    = "#58a6ff"
ACCENT2   = "#bc8cff"
TEXT      = "#c9d1d9"
TEXT_DIM  = "#8b949e"
BORDER    = "#30363d"
RED       = "#ff7b72"
GREEN     = "#7ee787"
YELLOW    = "#e3b341"
CYAN      = "#79c0ff"

# ── Mathematical constants ──────────────────────────────────────────
GOLDEN_RATIO = (1 + np.sqrt(5)) / 2
CONSTANTS = {
    "φ Golden":   GOLDEN_RATIO,
    "π Pi":       np.pi,
    "√2":         np.sqrt(2),
    "e Euler":    np.e,
    "A 1.855":    1.855,
    "2π":         2 * np.pi,
    "φ²":         GOLDEN_RATIO ** 2,
    "γ E-M":      0.5772156649,
}

LIMIT = 50_000  # max integers

# ── Prime sieve ─────────────────────────────────────────────────────
@njit
def _sieve_numba(n):
    is_prime = np.ones(n + 1, dtype=np.bool_)
    is_prime[0:2] = False
    for i in range(2, int(np.sqrt(n)) + 1):
        if is_prime[i]:
            is_prime[i * i :: i] = False
    return is_prime

def _sieve_standard(n):
    is_prime = np.ones(n + 1, dtype=bool)
    is_prime[0:2] = False
    for i in range(2, int(np.sqrt(n)) + 1):
        if is_prime[i]:
            is_prime[i * i :: i] = False
    return is_prime

def generate_primes(limit):
    return _sieve_numba(limit) if NUMBA_AVAILABLE else _sieve_standard(limit)

# ── SPIRNOR coordinate generation ──────────────────────────────────
def spirnor_coords_3d(n_values, C):
    r     = np.log(n_values)
    theta = (C * n_values) % (2 * np.pi)
    phi_a = (n_values * GOLDEN_RATIO) % (2 * np.pi)
    x = r * np.sin(theta) * np.cos(phi_a)
    y = r * np.sin(theta) * np.sin(phi_a)
    z = r * np.cos(theta)
    return x, y, z


# ====================================================================
#  Main Visualizer
# ====================================================================
class SPIRNORVisualizer:
    # ── init ────────────────────────────────────────────────────────
    def __init__(self):
        self.limit        = LIMIT
        self.display_limit = LIMIT
        self.current_C    = GOLDEN_RATIO
        self.animating    = False
        self.anim_timer   = None
        self.anim_step    = 0.005
        self.anim_dir     = 1

        # Data
        print(f"Sieving primes up to {self.limit:,} …")
        self.integers    = np.arange(1, self.limit + 1)
        self.prime_mask  = generate_primes(self.limit)[1 : self.limit + 1]
        self.composite_mask = ~self.prime_mask
        self.primes      = self.integers[self.prime_mask]
        self.composites  = self.integers[self.composite_mask]
        n_p = len(self.primes)
        n_c = len(self.composites)
        print(f"  → {n_p:,} primes · {n_c:,} composites")

        # Display state
        self.show_primes      = True
        self.show_composites  = True
        self.color_by_seq     = True
        self.show_trails      = False

        # Coordinate cache
        self._recompute()
        self._build_ui()

    # ── coordinate helpers ──────────────────────────────────────────
    def _active_primes(self):
        mask = self.prime_mask.copy()
        mask[self.display_limit:] = False
        return self.integers[mask]

    def _active_composites(self):
        mask = self.composite_mask.copy()
        mask[self.display_limit:] = False
        return self.integers[mask]

    def _recompute(self):
        p = self._active_primes()
        c = self._active_composites()
        self.p_coords = spirnor_coords_3d(p, self.current_C) if len(p) else (np.array([]),) * 3
        self.c_coords = spirnor_coords_3d(c, self.current_C) if len(c) else (np.array([]),) * 3

    # ── UI construction ─────────────────────────────────────────────
    def _build_ui(self):
        matplotlib.rcParams.update({
            "figure.facecolor": BG_DARK,
            "axes.facecolor":   BG_DARK,
            "axes.edgecolor":   BORDER,
            "axes.labelcolor":  TEXT,
            "text.color":       TEXT,
            "xtick.color":      TEXT_DIM,
            "ytick.color":      TEXT_DIM,
            "grid.color":       BORDER,
            "grid.alpha":       0.25,
        })

        self.fig = plt.figure(figsize=(18, 11))
        self.fig.canvas.manager.set_window_title("SPIRNOR 3D Visualizer v2")

        # ── 3-D axes (left 60 %) ───────────────────────────────────
        self.ax3d = self.fig.add_axes([0.02, 0.05, 0.56, 0.88], projection="3d")
        self.ax3d.set_facecolor(BG_DARK)
        self.ax3d.xaxis.pane.fill = False
        self.ax3d.yaxis.pane.fill = False
        self.ax3d.zaxis.pane.fill = False
        self.ax3d.xaxis.pane.set_edgecolor(BORDER)
        self.ax3d.yaxis.pane.set_edgecolor(BORDER)
        self.ax3d.zaxis.pane.set_edgecolor(BORDER)
        self.ax3d.view_init(elev=25, azim=45)

        # ── Title ──────────────────────────────────────────────────
        self.title_text = self.fig.text(
            0.30, 0.97, "SPIRNOR 3D Visualizer",
            fontsize=16, fontweight="bold", color=ACCENT,
            ha="center", va="top",
            path_effects=[pe.withStroke(linewidth=2, foreground=BG_DARK)],
        )
        self.subtitle_text = self.fig.text(
            0.30, 0.945, f"C = {self.current_C:.6f}  ·  φ (Golden Ratio)",
            fontsize=10, color=TEXT_DIM, ha="center", va="top",
        )

        # ── RIGHT PANEL ────────────────────────────────────────────
        panel_l, panel_w = 0.62, 0.36

        # --- C-value slider ---
        self._label(panel_l, 0.92, "WINDING CONSTANT  C")
        ax_sl = self.fig.add_axes([panel_l, 0.885, panel_w, 0.025])
        ax_sl.set_facecolor(BG_WIDGET)
        self.c_slider = Slider(ax_sl, "", 0.1, 10.0,
                               valinit=self.current_C, valfmt="%.5f",
                               color=ACCENT)
        self.c_slider.valtext.set_color(TEXT)
        self.c_slider.on_changed(self._on_c_changed)

        # --- Range slider ---
        self._label(panel_l, 0.865, "INTEGER RANGE  N")
        ax_rng = self.fig.add_axes([panel_l, 0.84, panel_w, 0.025])
        ax_rng.set_facecolor(BG_WIDGET)
        self.range_slider = Slider(ax_rng, "", 500, LIMIT,
                                   valinit=LIMIT, valstep=500,
                                   valfmt="%d", color=ACCENT2)
        self.range_slider.valtext.set_color(TEXT)
        self.range_slider.on_changed(self._on_range_changed)

        # --- Preset buttons (2 rows × 4) ---
        self._label(panel_l, 0.815, "PRESETS")
        bw, bh, bpad = 0.083, 0.032, 0.006
        self.preset_btns = {}
        for i, (name, val) in enumerate(CONSTANTS.items()):
            col, row = i % 4, i // 4
            x = panel_l + col * (bw + bpad)
            y = 0.775 - row * (bh + bpad)
            bx = self.fig.add_axes([x, y, bw, bh])
            bx.set_facecolor(BG_WIDGET)
            btn = Button(bx, name, color=BG_WIDGET, hovercolor="#30363d")
            btn.label.set_fontsize(8)
            btn.label.set_color(TEXT)
            btn.on_clicked(lambda _, v=val, n=name: self._set_preset(v, n))
            self.preset_btns[name] = btn

        # --- Display toggles ---
        self._label(panel_l, 0.70, "DISPLAY OPTIONS")
        ax_chk = self.fig.add_axes([panel_l, 0.555, 0.18, 0.14])
        ax_chk.set_facecolor(BG_PANEL)
        self.checkboxes = CheckButtons(
            ax_chk,
            ["Show Primes", "Show Composites", "Color by Sequence", "Show Trails"],
            [self.show_primes, self.show_composites, self.color_by_seq, self.show_trails],
        )
        for lbl in self.checkboxes.labels:
            lbl.set_fontsize(9)
            lbl.set_color(TEXT)
        # Style check-mark boxes (attribute name varies by mpl version)
        for attr in ("rectangles", "_buttons",):
            rects = getattr(self.checkboxes, attr, None)
            if rects is not None:
                for r in rects:
                    try:
                        r.set_edgecolor(BORDER)
                        r.set_facecolor(BG_WIDGET)
                    except Exception:
                        pass
                break
        self.checkboxes.on_clicked(self._on_toggle)

        # --- Colour-map selector ---
        self._label(panel_l + 0.20, 0.70, "PRIME COLORMAP")
        ax_cmap = self.fig.add_axes([panel_l + 0.20, 0.585, 0.15, 0.11])
        ax_cmap.set_facecolor(BG_PANEL)
        self.cmap_radio = RadioButtons(
            ax_cmap, ["plasma", "inferno", "turbo", "cool"],
            active=0,
        )
        for lbl in self.cmap_radio.labels:
            lbl.set_fontsize(9)
            lbl.set_color(TEXT)
        # Style radio circles (attribute name varies by mpl version)
        for attr in ("circles", "_buttons"):
            items = getattr(self.cmap_radio, attr, None)
            if items is not None:
                for c in items:
                    try:
                        c.set_edgecolor(BORDER)
                        c.set_facecolor(BG_WIDGET)
                    except Exception:
                        pass
                break
        self.cmap_radio.on_clicked(self._on_cmap)
        self.prime_cmap = "plasma"

        # --- Action buttons ---
        self._label(panel_l, 0.525, "ACTIONS")
        btn_defs = [
            ("FFT Analysis",   self._fft_analysis),
            ("2D Projection",  self._show_2d),
            ("▶ Animate",      self._toggle_animate),
            ("Reset View",     self._reset_view),
        ]
        for i, (label, cb) in enumerate(btn_defs):
            col, row = i % 2, i // 2
            x = panel_l + col * 0.185
            y = 0.475 - row * 0.045
            bx = self.fig.add_axes([x, y, 0.175, 0.038])
            bx.set_facecolor(BG_WIDGET)
            btn = Button(bx, label, color=BG_WIDGET, hovercolor="#30363d")
            btn.label.set_fontsize(9)
            btn.label.set_color(ACCENT if "Animate" in label else TEXT)
            btn.on_clicked(cb)
            if "Animate" in label:
                self.anim_btn = btn

        # --- Stats panel ---
        self._label(panel_l, 0.38, "STATISTICS")
        self.stats_text = self.fig.text(
            panel_l + 0.005, 0.36, "", fontsize=9,
            fontfamily="monospace", color=TEXT_DIM, va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=BG_PANEL,
                      edgecolor=BORDER, alpha=0.95),
        )

        # --- Formula box ---
        self.fig.text(
            panel_l + 0.005, 0.12,
            "SPIRNOR Mapping\n"
            "─────────────────────────\n"
            "r = ln(n)\n"
            "θ = C · n   (mod 2π)\n"
            "φ = n · φ   (mod 2π)\n\n"
            "x = r sin(θ) cos(φ)\n"
            "y = r sin(θ) sin(φ)\n"
            "z = r cos(θ)",
            fontsize=9, fontfamily="monospace", color=CYAN, va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=BG_PANEL,
                      edgecolor=BORDER, alpha=0.95),
        )

        self._draw()

    # ── tiny helpers ────────────────────────────────────────────────
    def _label(self, x, y, text):
        self.fig.text(x, y, text, fontsize=8, fontweight="bold",
                      color=TEXT_DIM, va="bottom",
                      fontfamily="monospace")

    def _current_preset_name(self):
        for name, val in CONSTANTS.items():
            if abs(val - self.current_C) < 1e-6:
                return name
        return "custom"

    # ── callbacks ───────────────────────────────────────────────────
    def _on_c_changed(self, val):
        self.current_C = val
        self._recompute()
        self._draw()

    def _on_range_changed(self, val):
        self.display_limit = int(val)
        self._recompute()
        self._draw()

    def _set_preset(self, val, name):
        self.current_C = val
        self.c_slider.set_val(val)
        # draw happens via slider callback

    def _on_toggle(self, label):
        if   label == "Show Primes":      self.show_primes     = not self.show_primes
        elif label == "Show Composites":  self.show_composites = not self.show_composites
        elif label == "Color by Sequence": self.color_by_seq    = not self.color_by_seq
        elif label == "Show Trails":      self.show_trails     = not self.show_trails
        self._draw()

    def _on_cmap(self, label):
        self.prime_cmap = label
        self._draw()

    def _reset_view(self, event):
        self.ax3d.view_init(elev=25, azim=45)
        self.fig.canvas.draw_idle()

    # ── animation ───────────────────────────────────────────────────
    def _toggle_animate(self, event):
        if self.animating:
            self.animating = False
            if self.anim_timer is not None:
                self.anim_timer.stop()
                self.anim_timer = None
            self.anim_btn.label.set_text("▶ Animate")
            self.anim_btn.label.set_color(ACCENT)
        else:
            self.animating = True
            self.anim_btn.label.set_text("■ Stop")
            self.anim_btn.label.set_color(RED)
            self.anim_timer = self.fig.canvas.new_timer(interval=60)
            self.anim_timer.add_callback(self._anim_step)
            self.anim_timer.start()

    def _anim_step(self):
        if not self.animating:
            return
        new_c = self.current_C + self.anim_step * self.anim_dir
        if new_c > 10.0:
            self.anim_dir = -1
            new_c = 10.0
        elif new_c < 0.1:
            self.anim_dir = 1
            new_c = 0.1
        self.current_C = new_c
        self.c_slider.set_val(new_c)

    # ── main draw ───────────────────────────────────────────────────
    def _draw(self):
        ax = self.ax3d
        ax.clear()
        ax.set_facecolor(BG_DARK)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor(BORDER)
        ax.yaxis.pane.set_edgecolor(BORDER)
        ax.zaxis.pane.set_edgecolor(BORDER)

        n_prime = n_comp = 0
        max_r = 0.1

        # Composites (draw first so primes are on top)
        if self.show_composites and len(self.c_coords[0]) > 0:
            xc, yc, zc = self.c_coords
            n_comp = len(xc)
            max_r = max(max_r, np.max(np.abs(xc)), np.max(np.abs(yc)), np.max(np.abs(zc)))

            if self.color_by_seq:
                ax.scatter(xc, yc, zc, c=np.arange(n_comp), cmap="viridis",
                           s=0.6, alpha=0.25, rasterized=True)
            else:
                ax.scatter(xc, yc, zc, c=ACCENT2, s=0.6, alpha=0.2, rasterized=True)

            if self.show_trails and n_comp > 1:
                ax.plot(xc, yc, zc, color=ACCENT2, alpha=0.12, linewidth=0.3)

        # Primes
        if self.show_primes and len(self.p_coords[0]) > 0:
            xp, yp, zp = self.p_coords
            n_prime = len(xp)
            max_r = max(max_r, np.max(np.abs(xp)), np.max(np.abs(yp)), np.max(np.abs(zp)))

            if self.color_by_seq:
                sc = ax.scatter(xp, yp, zp, c=np.arange(n_prime),
                                cmap=self.prime_cmap, s=2.5, alpha=0.85,
                                rasterized=True)
            else:
                ax.scatter(xp, yp, zp, c=RED, s=2, alpha=0.65, rasterized=True)

            if self.show_trails and n_prime > 1:
                ax.plot(xp, yp, zp, color=RED, alpha=0.2, linewidth=0.4)

        # Limits
        ax.set_xlim(-max_r, max_r)
        ax.set_ylim(-max_r, max_r)
        ax.set_zlim(-max_r, max_r)

        ax.set_xlabel("X", fontsize=9, labelpad=6)
        ax.set_ylabel("Y", fontsize=9, labelpad=6)
        ax.set_zlabel("Z", fontsize=9, labelpad=6)
        ax.grid(True, alpha=0.15)
        ax.tick_params(labelsize=7, pad=2)

        # Build legend manually
        from matplotlib.lines import Line2D
        handles = []
        if self.show_primes and n_prime:
            handles.append(Line2D([0], [0], marker="o", color="w", markerfacecolor=RED,
                                  markersize=6, linestyle="None",
                                  label=f"Primes  ({n_prime:,})"))
        if self.show_composites and n_comp:
            handles.append(Line2D([0], [0], marker="o", color="w", markerfacecolor=ACCENT2,
                                  markersize=6, linestyle="None",
                                  label=f"Composites  ({n_comp:,})"))
        if handles:
            ax.legend(handles=handles, loc="upper left", fontsize=8,
                      facecolor=BG_PANEL, edgecolor=BORDER, labelcolor=TEXT)

        # Update subtitle
        self.subtitle_text.set_text(
            f"C = {self.current_C:.6f}  ·  {self._current_preset_name()}"
            f"  ·  N ≤ {self.display_limit:,}"
        )

        # Update stats
        self.stats_text.set_text(
            f"  Primes visible   {n_prime:>8,}\n"
            f"  Composites vis.  {n_comp:>8,}\n"
            f"  Total plotted    {n_prime + n_comp:>8,}\n"
            f"  C value          {self.current_C:>12.6f}\n"
            f"  Range            1 – {self.display_limit:,}\n"
            f"  Prime density    {100 * n_prime / max(1, n_prime + n_comp):>8.2f} %"
        )

        self.fig.canvas.draw_idle()

    # ── FFT Analysis ────────────────────────────────────────────────
    def _fft_analysis(self, event):
        try:
            if self.show_primes and len(self.p_coords[0]) > 0:
                coords, tag = self.p_coords, "Primes"
            elif self.show_composites and len(self.c_coords[0]) > 0:
                coords, tag = self.c_coords, "Composites"
            else:
                print("Nothing visible to analyse.")
                return

            x, y, z = coords
            gs = 64
            mr = max(np.max(np.abs(x)), np.max(np.abs(y)), np.max(np.abs(z)))
            rng = [[-mr, mr]] * 3
            voxels, _ = np.histogramdd((x, y, z), bins=gs, range=rng)
            fft3 = np.fft.fftshift(np.fft.fftn(voxels))
            mag = np.log1p(np.abs(fft3))

            # Dark-themed FFT figure
            fig, axes = plt.subplots(2, 2, figsize=(13, 10),
                                     facecolor=BG_DARK)
            fig.suptitle(f"SPIRNOR FFT  ·  {tag}  ·  C = {self.current_C:.4f}",
                         fontsize=13, fontweight="bold", color=ACCENT)

            c = gs // 2
            slices = [
                (mag[:, :, c],        "XY Centre Slice"),
                (mag[:, c, :],        "XZ Centre Slice"),
                (mag[c, :, :].T,      "YZ Centre Slice"),
            ]

            # 3D peaks
            ax0 = fig.add_subplot(2, 2, 1, projection="3d")
            ax0.set_facecolor(BG_DARK)
            thresh = np.percentile(mag, 99.5)
            idx = np.argwhere(mag > thresh)
            vals = mag[mag > thresh]
            if len(idx):
                sc = ax0.scatter(idx[:, 2] - c, idx[:, 1] - c, idx[:, 0] - c,
                                 c=vals, cmap="hot", s=18, alpha=0.85)
                fig.colorbar(sc, ax=ax0, shrink=0.65, label="log|FFT|")
            ax0.set_title("3D Peaks", color=TEXT, fontsize=10)
            ax0.set_xlabel("fX", fontsize=8); ax0.set_ylabel("fY", fontsize=8)
            ax0.set_zlabel("fZ", fontsize=8)

            for i, (data, title) in enumerate(slices):
                ax = fig.add_subplot(2, 2, i + 2)
                ax.set_facecolor(BG_DARK)
                im = ax.imshow(data, cmap="inferno", origin="lower")
                ax.set_title(title, color=TEXT, fontsize=10)
                fig.colorbar(im, ax=ax, shrink=0.8)
                ax.tick_params(colors=TEXT_DIM, labelsize=7)

            fig.tight_layout(rect=[0, 0, 1, 0.95])
            plt.show()
            print(f"FFT Analysis complete  –  {len(idx)} peaks above 99.5th percentile")

        except Exception as e:
            import traceback
            print(f"FFT failed: {e}")
            traceback.print_exc()

    # ── 2-D Projection ──────────────────────────────────────────────
    def _show_2d(self, event):
        try:
            n = self.integers[: self.display_limit]
            r = np.log(n)
            theta = (self.current_C * n) % (2 * np.pi)
            x2 = r * np.cos(theta)
            y2 = r * np.sin(theta)

            pm = self.prime_mask[: self.display_limit]
            cm = self.composite_mask[: self.display_limit]

            fig, ax = plt.subplots(figsize=(11, 10), facecolor=BG_DARK)
            ax.set_facecolor(BG_DARK)
            ax.set_aspect("equal")
            fig.suptitle(
                f"SPIRNOR 2D Projection  ·  C = {self.current_C:.6f}",
                fontsize=13, fontweight="bold", color=ACCENT,
            )

            if self.show_composites:
                ax.scatter(x2[cm], y2[cm], c=ACCENT2, s=0.4, alpha=0.2,
                           rasterized=True, label=f"Composites ({cm.sum():,})")

            if self.show_primes:
                if self.color_by_seq:
                    sc = ax.scatter(x2[pm], y2[pm], c=np.arange(pm.sum()),
                                    cmap=self.prime_cmap, s=2, alpha=0.8,
                                    rasterized=True, label=f"Primes ({pm.sum():,})")
                    fig.colorbar(sc, ax=ax, label="Sequence index", shrink=0.8)
                else:
                    ax.scatter(x2[pm], y2[pm], c=RED, s=1.5, alpha=0.7,
                               rasterized=True, label=f"Primes ({pm.sum():,})")

            ax.set_xlabel("X", color=TEXT)
            ax.set_ylabel("Y", color=TEXT)
            ax.legend(fontsize=9, facecolor=BG_PANEL, edgecolor=BORDER, labelcolor=TEXT)
            ax.grid(True, alpha=0.15, color=BORDER)
            ax.tick_params(colors=TEXT_DIM)
            fig.text(0.5, 0.01, "r = ln(n)   θ = C · n  (mod 2π)",
                     ha="center", fontsize=9, color=TEXT_DIM, fontfamily="monospace")
            fig.tight_layout(rect=[0, 0.02, 1, 0.96])
            plt.show()

        except Exception as e:
            import traceback
            print(f"2D Projection failed: {e}")
            traceback.print_exc()

    # ── run ─────────────────────────────────────────────────────────
    def show(self):
        plt.show()


# ====================================================================
def main():
    print("═" * 52)
    print("  SPIRNOR 3D Interactive Visualizer  v2")
    print("═" * 52)
    viz = SPIRNORVisualizer()
    print("Ready — close the window to exit.")
    viz.show()


if __name__ == "__main__":
    main()
