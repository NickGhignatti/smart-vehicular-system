"""Report figures (static PNG, light theme), drawn with matplotlib.

Color follows the sensor, never the rank: camera = aqua, radar = orange, fused = blue
(slots 3, 2, 1 of the validated reference palette). The raw (unfiltered) baseline of a
sensor uses the same hue with hatching/dots, so identity never depends on color alone.
Ground truth is drawn in primary ink.
"""

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
CRITICAL = "#d03b3b"
COLORS = {"fused": "#2a78d6", "radar": "#eb6834", "camera": "#1baf7a",
          "radar_raw": "#eb6834", "camera_raw": "#1baf7a"}
LABELS = {"camera_raw": "camera raw", "radar_raw": "radar raw", "camera": "camera KF",
          "radar": "radar KF", "fused": "fused KF"}
EVENT_LABELS = {"camera_occlusion": "camera occluded", "partial_occlusion": "partial occlusion",
                "overtake": "car overtaking", "cut_in": "cut-in", "distraction": "driver distracted"}


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.yaxis.label.set_color(INK_2)
    ax.xaxis.label.set_color(INK_2)


def _shade_events(ax, events, label=False):
    last_label_end, row = -1e9, 0
    for start, end, kind, *_ in sorted(events, key=lambda e: e[0]):
        if kind == "distraction":
            continue
        ax.axvspan(start, end, color=GRID, alpha=0.7, linewidth=0)
        if label:
            # Stagger labels that would overlap the previous one.
            row = row + 1 if start < last_label_end else 0
            ax.text(start, 1.0 + 0.06 * row, EVENT_LABELS.get(kind, kind), transform=ax.get_xaxis_transform(),
                    fontsize=7, color=INK_2, va="bottom")
            span = ax.get_xlim()[1] - ax.get_xlim()[0]
            last_label_end = start + 0.11 * span


def plot_run(recording, series_by_mode, path, ttc_on_s=2.5, title=None, filtered=("camera", "radar", "fused")):
    """Timeline of one recording: gap, closing speed, TTC and the warning of every mode."""
    first = next(iter(series_by_mode.values()))
    t = first["t"]
    events = recording.meta.events
    modes = list(series_by_mode)
    fig, axes = plt.subplots(4, 1, figsize=(11, 9.5), sharex=True,
                             gridspec_kw={"height_ratios": [3, 2, 2, 0.35 * (len(modes) + 1)]})
    fig.patch.set_facecolor(SURFACE)
    for ax in axes:
        _style(ax)

    # (a) gap
    ax = axes[0]
    _shade_events(ax, events, label=True)
    ax.scatter(t, first["camera_gap"], s=3, color=COLORS["camera"], alpha=0.35, linewidths=0, label="camera measurement")
    ax.scatter(t, first["radar_gap"], s=3, color=COLORS["radar"], alpha=0.35, linewidths=0, label="radar measurement")
    for mode in filtered:
        if mode in series_by_mode:
            ax.plot(t, series_by_mode[mode]["est_gap"], color=COLORS[mode], linewidth=2 if mode == "fused" else 1.2,
                    label=LABELS[mode])
    ax.plot(t, first["gt_gap"], color=INK, linewidth=1, linestyle="--", label="ground truth")
    ax.set_ylabel("gap [m]")
    top = np.nanmax(first["gt_gap"]) if np.isfinite(np.nanmax(first["gt_gap"])) else 50
    ax.set_ylim(0, min(60, top * 1.3 + 2))
    ax.legend(loc="upper right", fontsize=7, ncol=3, frameon=False, labelcolor=INK_2, markerscale=3)

    # (b) closing speed
    ax = axes[1]
    _shade_events(ax, events)
    for mode in filtered:
        if mode in series_by_mode:
            ax.plot(t, -series_by_mode[mode]["est_rate"], color=COLORS[mode],
                    linewidth=2 if mode == "fused" else 1.2)
    gt_closing = -first["gt_rate"]
    ax.plot(t, gt_closing, color=INK, linewidth=1, linestyle="--")
    span = np.nanmax(np.abs(gt_closing)) if np.isfinite(np.nanmax(np.abs(gt_closing))) else 5
    ax.set_ylim(-span - 2, span + 2)
    ax.axhline(0, color=AXIS, linewidth=0.8)
    ax.set_ylabel("closing speed [m/s]")

    # (c) TTC
    ax = axes[2]
    _shade_events(ax, events)
    for mode in filtered:
        if mode in series_by_mode:
            ttc = np.where(np.isfinite(series_by_mode[mode]["ttc"]), series_by_mode[mode]["ttc"], np.nan)
            ax.plot(t, ttc, color=COLORS[mode], linewidth=2 if mode == "fused" else 1.2)
    gt_ttc = np.where(np.isfinite(first["gt_ttc"]), first["gt_ttc"], np.nan)
    ax.plot(t, gt_ttc, color=INK, linewidth=1, linestyle="--")
    ax.axhline(ttc_on_s, color=CRITICAL, linewidth=0.8, linestyle=":")
    ax.set_ylim(0, 8)
    ax.set_ylabel("TTC [s]")

    # (d) warnings, one strip per mode + ground-truth critical periods
    ax = axes[3]
    rows = modes + ["truth: TTC < on"]
    dt = t[1] - t[0] if len(t) > 1 else 0.05
    for i, name in enumerate(rows):
        if name in series_by_mode:
            active = series_by_mode[name]["warning"]
            color, hatch = COLORS[name], "////" if name.endswith("_raw") else None
        else:
            active = first["gt_ttc"] < ttc_on_s
            color, hatch = CRITICAL, None
        for start in np.where(np.diff(np.concatenate([[0], active.astype(int)])) == 1)[0]:
            end = start
            while end < len(active) and active[end]:
                end += 1
            ax.broken_barh([(t[start], (end - start) * dt)], (i + 0.15, 0.7), facecolors=color,
                           edgecolor=SURFACE if hatch else color, hatch=hatch, linewidth=0)
    ax.set_yticks([i + 0.5 for i in range(len(rows))])
    ax.set_yticklabels([LABELS.get(r, r) for r in rows], fontsize=7)
    ax.set_ylim(0, len(rows))
    ax.invert_yaxis()
    ax.grid(False)
    ax.set_xlabel("time [s]")
    for start, end, kind, *_ in events:
        if kind == "distraction":
            ax.axvspan(start, end, color=GRID, alpha=0.5, linewidth=0)

    fig.suptitle(title or f"{recording.meta.scenario} (seed {recording.meta.seed}, {recording.meta.source})",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.text(0.01, 0.005, "gray bands: scripted events (warning strips: driver distracted); "
             "strips: warning on (hatched = raw baseline); red = ground-truth TTC below the warning threshold",
             fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.015, 1, 0.98))
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_summary(summary, path):
    """Bar panels across modes: false warnings / min, gap RMSE, gap noise, closing-speed RMSE."""
    panels = [("false_per_min", "false warnings per minute"), ("missed", "missed critical episodes"),
              ("gap_rmse_m", "gap RMSE [m]"), ("gap_noise_m", "gap noise [m / frame]"),
              ("rate_rmse_mps", "closing-speed RMSE [m/s]")]
    modes = [row["mode"] for row in summary]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.0 * len(panels), 3.2))
    fig.patch.set_facecolor(SURFACE)
    for ax, (key, label) in zip(axes, panels):
        _style(ax)
        values = [row[key] for row in summary]
        x = np.arange(len(modes))
        bars = ax.bar(x, [0 if (v is None or (isinstance(v, float) and math.isnan(v))) else v for v in values],
                      width=0.68, color=[COLORS[m] for m in modes], edgecolor=SURFACE, linewidth=2,
                      hatch=None)
        for bar, mode in zip(bars, modes):
            if mode.endswith("_raw"):
                bar.set_hatch("////")
        for xi, v in zip(x, values):
            text = "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else (
                f"{v:.2f}" if isinstance(v, float) else str(v))
            ax.text(xi, ax.get_ylim()[1] * 0.01 + (0 if text == "n/a" else (v or 0)), text,
                    ha="center", va="bottom", fontsize=7, color=INK_2)
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[m] for m in modes], rotation=35, ha="right", fontsize=7)
        ax.set_title(label, fontsize=9, color=INK, loc="left")
        ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_error_vs_distance(distance, errors_by_sensor, sigma_by_sensor, path, title):
    """Calibration: measurement error vs true distance, binned mean +- std, reported sigma band."""
    fig, axes = plt.subplots(1, len(errors_by_sensor), figsize=(5.5 * len(errors_by_sensor), 3.6), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    bins = np.arange(0, 55, 5.0)
    for ax, (sensor, err) in zip(axes[0], errors_by_sensor.items()):
        _style(ax)
        color = COLORS[sensor]
        ok = ~np.isnan(err) & ~np.isnan(distance)
        ax.scatter(distance[ok], err[ok], s=4, color=color, alpha=0.3, linewidths=0, label="error per frame")
        centers, means, stds, sig = [], [], [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            sel = ok & (distance >= lo) & (distance < hi)
            if sel.sum() >= 5:
                centers.append((lo + hi) / 2)
                means.append(np.mean(err[sel]))
                stds.append(np.std(err[sel]))
                s = sigma_by_sensor.get(sensor)
                sig.append(np.nanmedian(s[sel]) if s is not None else np.nan)
        centers, means, stds, sig = map(np.array, (centers, means, stds, sig))
        if len(centers):
            ax.errorbar(centers, means, yerr=stds, color=INK, fmt="o", markersize=4, linewidth=1, capsize=3,
                        label="bin mean +- std")
            if not np.all(np.isnan(sig)):
                ax.fill_between(centers, -sig, sig, color=color, alpha=0.15, linewidth=0, label="reported sigma")
        ax.axhline(0, color=AXIS, linewidth=0.8)
        ax.set_xlabel("true gap [m]")
        ax.set_ylabel("measured - true [m]")
        ax.set_title(f"{sensor}", fontsize=9, color=INK, loc="left")
        ax.legend(fontsize=7, frameon=False, labelcolor=INK_2)
    fig.suptitle(title, fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    return path
