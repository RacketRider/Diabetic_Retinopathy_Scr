# Local data

This directory is intentionally ignored except for this file.

Expected runtime artifact:

- `DR_V4_RESNET101_SCREENING.mat` with the fixed train/validation/test
  datastores and `classNames`.

Do not commit retinal images, labels, or regenerated splits.

V3 genuine-resolution source:

- `downloads/DR1/dr_unified_v2/dr_unified_v2/` containing the matching original
  JPGs. This is the only retained image dataset. Its own train/val/test folders
  are not authoritative; E1/E3 preserve the
  split from `DR_V4_RESNET101_SCREENING.mat` and remaps files by basename.
