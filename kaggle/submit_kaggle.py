"""
Kaggle job: submit a .vcc produced by a vcc-run kernel (attached as a kernel source) to the Virtual Cell
Challenge. The VCC API key comes from a Kaggle secret named VCC_TOKEN (Add-ons -> Secrets in the Kaggle
editor); it is never printed or written to disk. Without the secret the job stops before submitting.
"""
import os
import subprocess
import sys
from pathlib import Path

MODEL = "__MODEL__"
NAME = "__NAME__"
DESCRIPTION = "__DESCRIPTION__"

try:
    from kaggle_secrets import UserSecretsClient
    token = UserSecretsClient().get_secret("VCC_TOKEN")
except Exception:
    token = None
if not token:
    sys.exit("No VCC_TOKEN secret attached to this kernel -- add it under Add-ons -> Secrets, then re-run. "
             "Nothing was submitted.")

hits = list(Path("/kaggle/input").rglob(f"{MODEL}_submission.vcc"))
if len(hits) != 1:
    sys.exit(f"expected exactly one {MODEL}_submission.vcc under /kaggle/input, found {hits}")
print(f"submitting {hits[0]} ({hits[0].stat().st_size / 1e6:.0f} MB)", flush=True)

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)
subprocess.run(["uv", "tool", "install", "vcc-cli"], check=True)
vcc = str(Path.home() / ".local" / "bin" / "vcc")
env = dict(os.environ, VCC_TOKEN=token)
subprocess.run([vcc, "submit", str(hits[0]), "-m", NAME, "-d", DESCRIPTION, "--wait"], check=True, env=env)
print("done", flush=True)
