# CARE-CT: JBI Supplementary Code

This directory contains the paper-specific supplementary code for:

**CARE-CT: Consistency-Aware Reflective Agent for Multi-Task Lesion Analysis in Computed Tomography**

Submitted to the *Journal of Biomedical Informatics (JBI)*.

## Scope

This release preserves the original Python package structure used by the manuscript and contains the code locations referenced in the Supplementary Methods Implementation Map. It is intended to support transparent review of the CARE-CT controller, Consistency Trust Score (CTS), Adaptive Reflection Module (ARM), constrained tool execution, calibration, and trajectory-selection logic.

It does **not** include datasets, model weights, API credentials, environment-specific files, or other private resources.

## Code map

| Path | Responsibility |
|---|---|
| `octotools/care_ct/tri_model.py` | Hosted-candidate schema, score normalization, argmax validation, prompt/image hashes, and provider candidate parsing |
| `octotools/care_ct/cts_v2_1.py` | Independent candidate collection, lineage-conflict handling, temperature scaling, reliability-weighted fusion, classification features, calibrator loading, and score computation |
| `octotools/care_ct/cts_v2.py` | Measurement-family features and size CTS support |
| `octotools/care_ct/choice_parser.py` | Exact structured A-D output schema and validation |
| `octotools/care_ct/arm_prompt.py` | Label-free ARM prompt, permitted audit error types, and sanitization |
| `octotools/models/formatters.py` | `ReflectionAudit` schema and strict JSON parser |
| `octotools/care_ct/selective_reflection.py` | Deterministic evidence-defect admission, stopping rules, and trajectory step limits |
| `octotools/care_ct/arm_strategies.py` | Task evidence profile, repair criteria, quality key, tie handling, and trajectory selection |
| `octotools/models/executor.py` | Constrained tool execution, argument validation, derived-image registration, and provenance enforcement |
| `tasks/fit_cts_v21.py` | Validation-derived family weights, classification/size calibration targets, logistic fitting, and frozen artifact construction |
| `tasks/solve.py` | Case orchestration, candidate acquisition, original/recovery trajectories, ARM execution, path selection, and output persistence |

## Validation

The supplementary bundle was checked so that all 11 Python files pass Python syntax compilation. The included file paths match the Supplementary Methods Implementation Map.

## Reproducibility note

This repository folder is a paper-specific code supplement rather than a complete standalone distribution of every upstream dependency. Some modules import additional components from the broader CARE-CT/OctoTools development environment. Those dependencies, datasets, model weights, and credentials are not included in this manuscript supplement.

## Citation

If you use this code, please cite the CARE-CT paper once the final bibliographic record is available.
