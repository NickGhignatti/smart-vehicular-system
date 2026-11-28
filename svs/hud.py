"""Driver warning widget drawn on the camera image (OpenCV, BGR).

- every detection in thin gray, the lead in green with its gap;
- a panel: mode, track status, gap +- sigma, closing speed, TTC, headway, controller;
- a TTC bar: green above ttc_off, amber between the thresholds, red below ttc_on;
- when the warning is active, a red banner "! BRAKE - collision risk" (icon + text, so the
  state never relies on color alone);
- ground truth in small muted text (bottom right), for the demo videos.
"""

import math

import cv2

# Status palette (BGR): good / warning / critical, plus inks.
GOOD = (12, 163, 12)
AMBER = (25, 178, 250)
CRITICAL = (59, 59, 208)
WHITE = (240, 239, 236)
MUTED = (183, 182, 175)
PANEL = (25, 26, 26)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _fmt(value, spec, unit=""):
    return "--" if value is None or not math.isfinite(value) else f"{format(value, spec)}{unit}"


def _panel(img, x, y, w, h, alpha=0.6):
    roi = img[y:y + h, x:x + w]
    overlay = roi.copy()
    overlay[:] = PANEL
    cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, dst=roi)


def _text(img, text, org, scale=0.5, color=WHITE, thickness=1):
    cv2.putText(img, text, org, FONT, scale, color, thickness, cv2.LINE_AA)


def draw_warning_hud(img, step, mode, warning_cfg, truth=None):
    """Draw on `img` in place and return it. step: svs.pipeline.StepResult."""
    h, w = img.shape[:2]
    est = step.estimate

    for det in step.detections:
        x1, y1, x2, y2 = (int(round(c)) for c in det.box)
        cv2.rectangle(img, (x1, y1), (x2, y2), MUTED, 1)
    if step.camera is not None:
        x1, y1, x2, y2 = (int(round(c)) for c in step.camera.box)
        cv2.rectangle(img, (x1, y1), (x2, y2), GOOD, 2)
        _text(img, f"{step.camera.gap_m:.1f} m", (x1, max(14, y1 - 6)), 0.45, GOOD, 1)

    # Info panel.
    _panel(img, 8, 8, 232, 132)
    status = est.status if est.valid else "no lead"
    closing = est.closing_speed_mps if est.confirmed else None
    lines = [
        (f"{mode}  |  track: {status}", MUTED),
        (f"gap      {_fmt(est.gap_m if est.confirmed else None, '5.1f', ' m')}"
         f"  +-{_fmt(est.sigma_gap_m if est.confirmed else None, '.2f')}", WHITE),
        (f"closing  {_fmt(closing, '+5.1f', ' m/s')}", WHITE),
        (f"TTC      {_fmt(step.warning.ttc_s, '5.1f', ' s')}", WHITE),
        (f"headway  {_fmt(step.warning.headway_s, '5.1f', ' s')}", WHITE),
    ]
    if step.command is not None:
        lines.append((f"ACC {step.command.reason:9s} {step.command.accel_mps2:+.1f} m/s2", WHITE))
    for i, (text, color) in enumerate(lines):
        _text(img, text, (16, 28 + 20 * i), 0.45, color)

    # TTC bar along the bottom: full = TTC >= 6 s.
    ttc = step.warning.ttc_s
    bar_w = w - 32
    fill = 1.0 if not math.isfinite(ttc) else max(0.0, min(1.0, ttc / 6.0))
    color = GOOD if ttc > warning_cfg.ttc_off_s else (AMBER if ttc > warning_cfg.ttc_on_s else CRITICAL)
    cv2.rectangle(img, (16, h - 18), (16 + bar_w, h - 8), PANEL, -1)
    cv2.rectangle(img, (16, h - 18), (16 + int(bar_w * fill), h - 8), color, -1)
    for threshold in (warning_cfg.ttc_on_s, warning_cfg.ttc_off_s):
        x = 16 + int(bar_w * threshold / 6.0)
        cv2.line(img, (x, h - 22), (x, h - 4), WHITE, 1)
    _text(img, "TTC", (16, h - 24), 0.4, WHITE)

    if step.warning.active:
        x0 = 248                       # right of the info panel
        bw = min(w - x0 - 8, 360)
        cv2.rectangle(img, (x0, 8), (x0 + bw, 48), CRITICAL, -1)
        cv2.circle(img, (x0 + 24, 28), 13, WHITE, 2)
        _text(img, "!", (x0 + 20, 35), 0.65, WHITE, 2)
        _text(img, "BRAKE - collision risk", (x0 + 48, 35), 0.65, WHITE, 2)

    if truth is not None:
        _text(img, f"truth gap {truth.gap_m:5.1f} m  TTC {_fmt(truth.ttc_s, '4.1f', ' s')}",
              (w - 260, h - 28), 0.4, MUTED)
    return img
