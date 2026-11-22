"""`python -m svs evaluate`: replay recordings in every mode and produce the report material.

Outputs in --out:
- metrics.csv       one row per (recording, mode)
- summary.csv/.md   aggregated per mode (the comparison table of the report)
- summary.png       bar charts across modes
- <recording>.png   timeline of every recording (gap, closing speed, TTC, warnings)
"""

import csv
import logging
import math

from svs.data.recording import Recording
from svs.evaluation import evaluate_recording, summarize
from svs.fusion.tracker import MODES
from svs.plots import plot_run, plot_summary

log = logging.getLogger(__name__)

SUMMARY_COLUMNS = [
    ("mode", "mode"), ("availability", "availability"), ("gap_rmse_m", "gap RMSE [m]"),
    ("gap_noise_m", "gap noise [m]"), ("rate_rmse_mps", "closing-speed RMSE [m/s]"),
    ("rate_noise_mps", "closing-speed noise [m/s]"), ("warnings", "warnings"),
    ("false_warnings", "false"), ("false_per_min", "false / min"), ("critical_episodes", "critical episodes"),
    ("missed", "missed"), ("latency_s", "mean latency [s]"),
]


def _fmt(value):
    if isinstance(value, float):
        return "n/a" if math.isnan(value) else f"{value:.3f}"
    return str(value)


def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _fmt(v) for k, v in row.items()})


def markdown_table(summary):
    lines = ["| " + " | ".join(label for _, label in SUMMARY_COLUMNS) + " |",
             "|" + "---|" * len(SUMMARY_COLUMNS)]
    for row in summary:
        lines.append("| " + " | ".join(_fmt(row[key]) for key, _ in SUMMARY_COLUMNS) + " |")
    return "\n".join(lines)


def run(cfg, paths, out_dir, modes=MODES, plots=True):
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    sources = set()
    for path in paths:
        recording = Recording.load(path)
        sources.add(recording.meta.source)
        rec_rows, series = evaluate_recording(recording, cfg, modes)
        rows.extend(rec_rows)
        for row in rec_rows:
            log.info("%-24s %-10s gap RMSE %5.2f m  false %d  missed %d/%d", row["recording"], row["mode"],
                     row["gap_rmse_m"], row["false_warnings"], row["missed"], row["critical_episodes"])
        if plots:
            plot_run(recording, series, out_dir / f"{path.stem}.png", ttc_on_s=cfg.warning.ttc_on_s, label_ttc_s=cfg.warning.label_ttc_s)

    summary = summarize(rows)
    write_csv(rows, out_dir / "metrics.csv")
    write_csv(summary, out_dir / "summary.csv")
    table = markdown_table(summary)
    note = ""
    if "synthetic" in sources:
        note = "\n\n> Includes SYNTHETIC recordings: development numbers, not results.\n"
    (out_dir / "summary.md").write_text(
        f"# Evaluation summary\n\n{len(paths)} recordings: {', '.join(p.stem for p in paths)}\n\n{table}{note}",
        encoding="utf-8")
    if plots:
        plot_summary(summary, out_dir / "summary.png")
    print(table)
    log.info("Report material in %s", out_dir)
    return summary
