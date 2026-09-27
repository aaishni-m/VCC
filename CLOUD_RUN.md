# Running VCC off the laptop

Code lives in git; data never does (see `.gitignore`). The heavy steps need roughly **20 GB RAM**
(`vcc prep` peaked around 19 GB) and **~20 GB disk** (a 14 GB `.h5ad` plus a 3 GB `.vcc`).

## Free options (check the current limits before relying on them)

| Option | Cost | Resources | Notes |
|---|---|---|---|
| **Kaggle Notebook** (CPU) | Free, no card | ~30 GB RAM, 4 CPU, ~20 GB in `/kaggle/working`, 12 h sessions | Recommended first try. Turn on Internet in notebook settings (needs phone verification). |
| **Oracle Cloud Always Free** (Ampere A1) | Free forever, card needed for sign-up | Up to 4 ARM cores / 24 GB RAM, 200 GB disk | A real VM. ARM, but numpy/anndata work. A1 capacity is often "out of stock"; retry or pick another region. |
| **Google Cloud free trial** | $300 credit for 90 days, card needed | e.g. e2-highmem-4 (32 GB) | Not charged unless you manually upgrade. **Stop the VM when you're done.** |
| GitHub Actions | Free | ~16 GB RAM, ~14 GB disk | Too small for `vcc prep` / the output. Fine for linting and the LOO scripts only. |

## GitHub Actions (recommended: no notebook clicking)

- **CI** (`.github/workflows/ci.yml`) runs on every push: lint + unit tests on synthetic data.
- **Kaggle run** (`.github/workflows/kaggle-run.yml`) is a manual button. It uploads the repo to Kaggle
  as dataset `<user>/vcc-code`, then runs `kaggle/run_kaggle.py` on Kaggle's free CPU machine. That script
  runs the model, then `vcc prep` (dry-run, or full). The `.vcc` lands in the kernel's Output tab.

One-time setup:
1. On Kaggle: **Settings -> API -> Create New Token** (an access token string).
2. On GitHub: **repo Settings -> Secrets and variables -> Actions**, add `KAGGLE_USERNAME` (`aaishnimulik`)
   and `KAGGLE_API_TOKEN`.

Locally on Windows the token lives in `C:\Users\User\.kaggle\access_token` (no `.txt`). Run the CLI as
`uvx --python 3.12 kaggle ...` (Python 3.10 only gets the old CLI), and pass `-p .` from inside the
folder -- absolute `C:/...` paths break its uploader.
3. On Kaggle: create a **private** dataset with slug `vcc-data` containing `controls/` and `vcc-perturb.csv`.
4. On Kaggle: verify your phone number so kernels can use the internet (needed to install `vcc-cli`).

Then: **Actions -> Kaggle run -> Run workflow**. Submit the resulting `.vcc` yourself.

## Kaggle workflow (manual notebook alternative)

1. Create a **private Kaggle Dataset** named `vcc-data` with `controls/` (the 3 context `.h5ad` files,
   `gene_names.csv`, `pert_counts.csv`, `manifest.json`) and `vcc-perturb.csv`.
2. In the notebook: **Add Data -> vcc-data**, and put your VCC API key in **Add-ons -> Secrets** as `VCC_TOKEN`.
3. Run these cells:

```bash
!git clone https://github.com/<you>/<repo>.git vcc && pip install -q -r vcc/requirements.txt uv
```

```python
import os
from kaggle_secrets import UserSecretsClient
os.environ["VCC_TOKEN"] = UserSecretsClient().get_secret("VCC_TOKEN")
os.environ["VCC_CONTROLS_DIR"] = "/kaggle/input/vcc-data/controls"
os.environ["VCC_PERTURB_CSV"] = "/kaggle/input/vcc-data/vcc-perturb.csv"
os.environ["VCC_OUTPUTS_DIR"] = "/tmp/outputs"   # big .h5ad goes to scratch, not the 20 GB working dir
```

```bash
!cd vcc && python models/gene_transfer_b4.py && python models/generate_b2_submission.py --model b4
!uv tool install vcc-cli   # installs to ~/.local/bin; call it by full path if `vcc` isn't found
!vcc prep /tmp/outputs/b4_submission.h5ad -g $VCC_CONTROLS_DIR/gene_names.csv --perts $VCC_CONTROLS_DIR/pert_counts.csv -o /kaggle/working/b4_submission.vcc --dry-run
```

Review the dry run, drop `--dry-run`, then `vcc submit` as described in `REPRODUCE_AND_SUBMIT.md`.
Never print the token or commit it.
