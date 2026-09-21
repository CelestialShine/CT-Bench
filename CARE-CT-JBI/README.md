# CARE-CT supplementary code for JBI revision

This directory contains the paper-specific CARE-CT source files provided with the revised Journal of Biomedical Informatics submission. The original Python folder structure is preserved.

## Included components

- `tasks/solve.py`: case orchestration, candidate acquisition, original and recovery trajectories, ARM execution, path selection, and output persistence.
- `tasks/fit_cts_v21.py`: validation-derived family weights, classification/size calibration targets, logistic fitting, and frozen artifact construction.
- `octotools/care_ct/`: CTS, ARM, choice parsing, tri-model candidate handling, reflection admission, and trajectory selection.
- `octotools/models/`: constrained execution and reflection-output formatting.
- `README_CODE_MAP.txt`: detailed implementation map corresponding to the Supplementary Methods.
- `SHA256SUMS.txt`: checksums for the submitted code files.

## Scope

This is the exact paper-specific supplementary code bundle from the JBI revision package. It contains the files explicitly mapped in the Supplementary Methods and is intended to document the implementation used in the manuscript. It does not include model weights, datasets, credentials, or unrelated repository files.

## Citation

If you use this code, please cite the accompanying CARE-CT manuscript.
