# Exact CARE-CT JBI code archive

The exact paper-specific ZIP from the JBI revision package is stored here losslessly as eight Base64 text parts because the connected GitHub uploader accepts UTF-8 text files but not direct binary ZIP uploads.

To reconstruct the original archive:

```bash
python reconstruct_code_archive.py
```

This creates `CARE_CT_JBI_Code.zip` with SHA-256:

`cc3a2757ed3d4b32895acaaa370be7c6ca75648135f1c296d5e432731fd0b61b`

The reconstructed ZIP preserves the original folder structure (`tasks/`, `octotools/care_ct/`, and `octotools/models/`) and contains the exact paper-specific code files supplied with the revised manuscript.
