# Supplementary code archive

The CARE-CT JBI supplementary code archive is stored in four binary parts under `archive_parts/` because the connected GitHub workflow does not upload binary release assets directly.

## Reconstruct the archive

From inside the `CARE-CT-JBI` directory, run:

```bash
cat archive_parts/CARE_CT_JBI_Supplementary_Code.tar.xz.part-* > CARE_CT_JBI_Supplementary_Code.tar.xz
```

Then verify the archive:

```bash
sha256sum CARE_CT_JBI_Supplementary_Code.tar.xz
```

Expected SHA-256:

```text
8c0949bfa6eb4a3e98e6f2fffb59d732d857835d982e508582d300b862086c36
```

Extract it with:

```bash
tar -xJf CARE_CT_JBI_Supplementary_Code.tar.xz
```

The extracted bundle preserves the Python package paths referenced by the manuscript rather than flattening them.

## Part checksums

```text
6725dd0dec5d1f615e5a25e9f3b07e150059ecce7403aeca4fb5affd000be957  CARE_CT_JBI_Supplementary_Code.tar.xz.part-00
1e0c5357674c76bcf137aa49dbf89af67004e6581cefb0051c588641fc0caa8a  CARE_CT_JBI_Supplementary_Code.tar.xz.part-01
f69aa5a979b9251a4b7e230574bab682bfa0c39571a5d548b98a6269076b912e  CARE_CT_JBI_Supplementary_Code.tar.xz.part-02
d0381c0c58552149fb932f3bd144d503b69c7bcf16081b49ef6d62918e3f6ab5  CARE_CT_JBI_Supplementary_Code.tar.xz.part-03
```
