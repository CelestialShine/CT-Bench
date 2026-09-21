from pathlib import Path
import base64
import hashlib

EXPECTED_SHA256 = "cc3a2757ed3d4b32895acaaa370be7c6ca75648135f1c296d5e432731fd0b61b"

here = Path(__file__).resolve().parent
parts = sorted(here.glob("CARE_CT_JBI_Code.b64.part*"))
if len(parts) != 8:
    raise RuntimeError(f"Expected 8 archive parts, found {len(parts)}")

payload = "".join(part.read_text(encoding="utf-8").strip() for part in parts)
out = here / "CARE_CT_JBI_Code.zip"
out.write_bytes(base64.b64decode(payload))

digest = hashlib.sha256(out.read_bytes()).hexdigest()
if digest != EXPECTED_SHA256:
    raise RuntimeError(f"SHA-256 mismatch: {digest}")

print(f"Reconstructed: {out}")
print(f"SHA-256: {digest}")
