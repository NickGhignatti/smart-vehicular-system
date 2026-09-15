## Goal

Build a small sensor-fusion project that tracks the lead vehicle smoothly in stop-and-go traffic and triggers a stable warning or a gentle slow-down action.

### Tools
- CARLA front RGB camera, ragar and ego-vehicle speed.
- A simple costant-velocity Kalman filter to fuse distance, relative speed and object persistence over time
- A warning widget or a basic longitudinal controller

### Implementation
- [ ] Detect the lead vehicle in the camera image and extract radar range/range-rate for the same target.
- [ ] Associate the two measurements and fuse them with a Kalman filter to obtain a smoother estimate of gap and closing speed.
- [ ] Compute a simple time-to-collision indicator and use it to warn the driver or reduce speed.
- [ ] Compare camera-only, radar-only, and fused behavior in stop-and-go traffic and mild occlusion.

## Validation and Deliverables
- Show that the fused estimate is less noisy and produce fewer false warnings than the single-sensor baseline.
- Deliverables:
    - [ ] CARLA code
    - [ ] Kalman-fusion module
    - [ ] Report with distance/TTC plots
    - [ ] Demo video