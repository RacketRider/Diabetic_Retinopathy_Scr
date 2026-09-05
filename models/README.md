# Local model checkpoints

This directory is intentionally ignored except for this file.

Expected checkpoints:

- `AG_V4_5_ReducedAug.mat` — required E1 initialization checkpoint
- `V3_E1_HighResFOV448.mat` — final E1 checkpoint and E3 initializer
- `V3_E3_RetinaWalker_best.mat` — final E3 inference checkpoint
- `V3_E3_RetinaWalker_latest.mat` — E3 resume state

Checkpoint files are large generated artifacts and must not be committed.
