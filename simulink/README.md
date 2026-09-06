# Simulink Telemedicine Workflow Model

## Purpose

This directory contains the Simulink model for the DR screening telemedicine pipeline simulation (SIH26038).

## Pipeline Overview

```
PHC Fundus Capture (50 img/day)
    → Upload via 2 Mbps link
    → AI Server (~350ms/image)
    → Triage: Referable (18%) vs Normal (82%)
    → Ophthalmologist Review Queue (<30s per case)
    → Clinical Report Generation
```

## Model Parameters

| Parameter | Value | Source |
|:---|:---|:---|
| Daily image volume | 50 images/PHC | WHO rural screening guidelines |
| Upload bandwidth | 2 Mbps | Typical rural Indian connectivity |
| AI inference time | ~350 ms | Measured on CPU (ONNX Runtime) |
| Referral rate | 18% | Calibrated from E1 threshold (0.164) |
| Auto-clear rate | 82% | Non-referable cases |
| Review time per case | <30 seconds | With Grad-CAM + clinical evidence |

## Files

- `dr_screening_pipeline.slx` — Simulink model (requires MATLAB R2024a+ with Simulink)
- `create_dr_screening_pipeline.m` — Programmatic model generator and automated CI verification script
- This model is visualized interactively in the web frontend's Simulink tab

## Running the Simulation

### Open and Simulate via MATLAB GUI:
```matlab
addpath('simulink');
open_system('dr_screening_pipeline');
simOut = sim('dr_screening_pipeline');
```

### Programmatic Re-generation and Automated Verification:
```matlab
run('simulink/create_dr_screening_pipeline.m');
```

## Requirements

- MATLAB R2024a or later
- Simulink
- Simulink Report Generator (optional, for PDF export)

