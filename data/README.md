# Local data

This directory is intentionally ignored except for this file.

Expected runtime artifact:

- `DR_V4_RESNET101_SCREENING.mat` with the fixed train/validation/test
  datastores and `classNames`.

Raw downloads currently live under ignored subdirectories. Do not commit
retinal images, labels, archives, or regenerated splits.

V3 genuine-resolution source:

- `downloads/DR1/dr_unified_v2/dr_unified_v2/` containing the matching original
  JPGs. Its own train/val/test folders are not authoritative; V3 preserves the
  split from `DR_V4_RESNET101_SCREENING.mat` and remaps files by basename.
