CARE-CT supplementary code files

This directory contains only the code locations listed in the Supplementary Methods Implementation Map. The original uploaded code archive is preserved as CARE_CT_Supplementary_Code_Only_Mentioned.zip.

Code location                                  Responsibility
----------------------------------------------------------------------------------------------------
octotools/care_ct/tri_model.py                Hosted-candidate schema, score normalization, argmax validation, prompt and image hashes, and provider candidate parsing.
octotools/care_ct/cts_v2_1.py                 Independent candidate collection, lineage conflict handling, temperature scaling, reliability-weighted fusion, classification features, calibrator loading, and score computation.
octotools/care_ct/cts_v2.py                   Measurement-family features and size CTS support.
octotools/care_ct/choice_parser.py             Exact structured A-D output schema and validation.
octotools/care_ct/arm_prompt.py                Label-free ARM prompt, permitted audit error types, and sanitization of labels, identifiers, paths, and image bytes.
octotools/models/formatters.py                 ReflectionAudit schema and strict JSON parser.
octotools/care_ct/selective_reflection.py      Deterministic evidence-defect admission, stopping rules, and trajectory step limits.
octotools/care_ct/arm_strategies.py            Task evidence profile, material-repair criteria, lexicographic quality key, tie handling, and trajectory selection.
octotools/models/executor.py                   Constrained tool execution, argument validation, derived-image registration, and provenance enforcement.
tasks/fit_cts_v21.py                           Validation-derived family weights, classification and size calibration targets, logistic fitting, and frozen artifact construction.
tasks/solve.py                                 Case orchestration, candidate acquisition, original and recovery trajectories, ARM execution, path selection, and output persistence.

Validation performed for this bundle:
- All 11 Python files passed Python syntax compilation.
- The file paths match the Supplementary Methods Implementation Map.
- No additional repository files, model weights, data, credentials, or environment files were added.
