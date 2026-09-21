# CARE-CT supplementary code for JBI revision

This directory contains the paper-specific CARE-CT source code accompanying the revised *Journal of Biomedical Informatics* submission.

## Download

Download the complete archive:

**`CARE_CT_JBI_Supplementary_Code.zip`**

The ZIP preserves the original Python directory structure and contains the 11 implementation files mapped in the Supplementary Methods, together with the code map, checksums, and README.

ZIP SHA-256: `d673ff571a54480e34014a80b3ab86c53498643c25bcee9ec68abc207b9ef931`

## Included components

- `tasks/solve.py` — case orchestration, candidate acquisition, original/recovery trajectories, ARM execution, path selection, and output persistence.
- `tasks/fit_cts_v21.py` — validation-derived family weights, calibration targets, logistic fitting, and frozen artifact construction.
- `octotools/care_ct/` — CTS, ARM, choice parsing, tri-model candidate handling, reflection admission, and trajectory selection.
- `octotools/models/` — constrained execution and reflection-output formatting.
- `README_CODE_MAP.txt` — detailed mapping between manuscript methods and implementation files.
- `SHA256SUMS.txt` — SHA-256 checksums of the 11 Python source files.

## Scope

This is the exact paper-specific supplementary code package prepared for the JBI revision. It does not include patient data, CT datasets, model weights, API credentials, or unrelated repository files.

## Citation

If you use this code, please cite the accompanying CARE-CT manuscript.
