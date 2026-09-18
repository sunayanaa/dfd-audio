# =============================================================================
# Program      : 01c_p1_bootstrap_ci.py
# Version      : 1.0
# Description  : Bootstrap 95% confidence intervals for the P1
#                per-filter discrimination statistic T_m.
#
#                Copies exp1_p1_energy.npz from Google Drive (saved by the
#                updated Program 01) and bootstraps CIs on T_m^max for
#                each filterbank by resampling clips (B=500). Reports
#                CI on T_m^max and on the peak frequency filter index.
#
# INPUT (Google Drive):
#   exp1_p1_energy.npz    Per-clip energy matrices for 12-EDO and
#                         53-EDO conditions, all four filterbanks
#                         plus matched linear references.
#
# OUTPUT (Google Drive):
#   exp1_p1_ci_results.json   Bootstrap 95% CIs on T_m^max per
#                             filterbank; used to update Table I.
#
# METHOD:
#   For each bootstrap replicate, resample N clips with replacement
#   from both the 12-EDO and 53-EDO energy matrices. Recompute
#   T_m for each filter, take the max. Repeat B=500 times.
#   Report 2.5th and 97.5th percentiles as the 95% CI.
#
# GPU Required : NO
# Dependencies : numpy
# Running time : < 2 minutes
# To be run after running 01b_exp1_bootstrap_ci.py
# Change Log   :
#   v1.0  2026-09-18  Initial version
# =============================================================================

from datetime import datetime
start_time = datetime.now()
print(f"Start time: {start_time}")

import os, json
import numpy as np

from google.colab import drive
drive.mount('/content/drive')

PROJECT_DIR = "/content/drive/MyDrive/papers/dfd-audio/"
LOCAL_DIR = "/tmp/dfd_exp1c"
os.makedirs(LOCAL_DIR, exist_ok=True)
os.makedirs(PROJECT_DIR, exist_ok=True)

RANDOM_SEED  = 42
N_BOOT       = 500
FILTERBANKS  = ["mel", "bark", "erb", "cqt"]
FB_LABELS    = {"mel":"Mel","bark":"Bark","erb":"ERB","cqt":"CQT"}


def copy_from_drive(remote_name, local_path, skip_if_exists=True):
    """Copy a file from Google Drive project directory to local."""
    drive_path = os.path.join(PROJECT_DIR, remote_name)
    if not os.path.exists(drive_path):
        print(f"[Drive] {remote_name} not found on Drive.")
        return False
    if skip_if_exists and os.path.exists(local_path):
        if os.path.getsize(local_path) == os.path.getsize(drive_path):
            return True
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    tmp_path = local_path + ".partial"
    with open(drive_path, "rb") as fsrc, open(tmp_path, "wb") as fdst:
        chunk_size = 16 * 1024 * 1024
        while True:
            buf = fsrc.read(chunk_size)
            if not buf:
                break
            fdst.write(buf)
    os.replace(tmp_path, local_path)
    print(f"  [Drive] Copied from Drive: {remote_name}")
    return True


def copy_to_drive(local_path, remote_name):
    """Copy a file from local to Google Drive project directory."""
    drive_path = os.path.join(PROJECT_DIR, remote_name)
    tmp_path = drive_path + ".partial"
    with open(local_path, "rb") as fsrc, open(tmp_path, "wb") as fdst:
        chunk_size = 16 * 1024 * 1024
        while True:
            buf = fsrc.read(chunk_size)
            if not buf:
                break
            fdst.write(buf)
    os.replace(tmp_path, drive_path)
    print(f"  [Drive] Copied to Drive: {remote_name}")


# =============================================================================
# COPY FROM GOOGLE DRIVE
# =============================================================================

print("[Step 1] Copying exp1_p1_energy.npz from Google Drive...")
npz_local = os.path.join(LOCAL_DIR, "exp1_p1_energy.npz")
copy_from_drive("exp1_p1_energy.npz", npz_local)
E = np.load(npz_local)
print(f"  Keys: {list(E.files)[:4]}...")

# =============================================================================
# BOOTSTRAP T_m CI
# =============================================================================

print("\n[Step 2] Bootstrapping T_m CIs (B=500, clip resampling)...")

def tm_max_from_matrices(E_12, E_53):
    """Compute T_m^max from two (n_clips, n_filters) energy matrices."""
    N      = len(E_12)
    mu_12  = E_12.mean(axis=0)
    mu_53  = E_53.mean(axis=0)
    sig_12 = E_12.std(axis=0)
    sig_53 = E_53.std(axis=0)
    denom  = np.sqrt((sig_12**2 + sig_53**2) / N + 1e-12)
    Tm     = np.abs(mu_12 - mu_53) / denom
    peak_m = int(np.argmax(Tm))
    return float(Tm.max()), peak_m

results = {}
rng = np.random.default_rng(RANDOM_SEED)

print(f"\n  {'FB':<6} {'Tm_max':>8}  {'95% CI':>22}  {'SE':>7}  {'peak_m CI':>18}")
print("  " + "-"*70)

for fb in FILTERBANKS:
    E_12 = E[f"E_12_{fb}"]
    E_53 = E[f"E_53_{fb}"]
    N    = len(E_12)

    # Point estimate
    tm_max_obs, peak_m_obs = tm_max_from_matrices(E_12, E_53)

    # Bootstrap
    tm_boots     = []
    peak_m_boots = []

    for _ in range(N_BOOT):
        idx_12 = rng.integers(0, N, size=N)
        idx_53 = rng.integers(0, N, size=N)
        tm_b, pm_b = tm_max_from_matrices(E_12[idx_12], E_53[idx_53])
        tm_boots.append(tm_b)
        peak_m_boots.append(pm_b)

    tm_boots     = np.array(tm_boots)
    peak_m_boots = np.array(peak_m_boots)

    tm_lo  = float(np.percentile(tm_boots, 2.5))
    tm_hi  = float(np.percentile(tm_boots, 97.5))
    tm_se  = float(tm_boots.std())
    pm_lo  = int(np.percentile(peak_m_boots, 2.5))
    pm_hi  = int(np.percentile(peak_m_boots, 97.5))

    results[fb] = {
        "tm_max"     : tm_max_obs,
        "tm_ci_lo"   : tm_lo,
        "tm_ci_hi"   : tm_hi,
        "tm_se"      : tm_se,
        "peak_m"     : peak_m_obs,
        "peak_m_ci"  : [pm_lo, pm_hi],
    }

    print(f"  {FB_LABELS[fb]:<6} {tm_max_obs:>8.2f}  "
          f"[{tm_lo:>8.2f}, {tm_hi:>8.2f}]  {tm_se:>7.2f}  "
          f"[{pm_lo:>3d}, {pm_hi:>3d}]")

# =============================================================================
# SAVE AND COPY TO DRIVE
# =============================================================================

print("\n[Step 3] Saving results...")
out_path = os.path.join(LOCAL_DIR, "exp1_p1_ci_results.json")
with open(out_path, "w") as f:
    json.dump({"P1_Tm_CI": results,
               "n_boot": N_BOOT,
               "method": "clip resampling with replacement"}, f, indent=2)
copy_to_drive(out_path, "exp1_p1_ci_results.json")

# =============================================================================
# UPDATED TABLE I LATEX
# =============================================================================

print("\n" + "="*65)
print("UPDATED TABLE I LATEX")
print("="*65)
print(r"\begin{table}[t]")
print(r"\centering")
print(r"\caption{P2: Peak per-filter discrimination statistic")
print(r"  $T_m$ for each filterbank with bootstrap 95\,\% CIs")
print(r"  ($B=500$, clip resampling). A high $T_m$ indicates the")
print(r"  filterbank assigns different energy distributions to")
print(r"  12-EDO and 53-EDO partials at that frequency.}")
print(r"\label{tab:tm}")
print(r"\footnotesize")
print(r"\begin{tabular}{lcccc}")
print(r"\toprule")
print(r"Filterbank & $T_m^{\max}$ & 95\,\% CI & Peak freq.\ (Hz) & Behaviour \\")
print(r"\midrule")
for fb in FILTERBANKS:
    r2 = results[fb]
    lo = r2['tm_ci_lo']
    hi = r2['tm_ci_hi']
    tm = r2['tm_max']
    # Peak Hz: load from exp1 json for display
    print(f"{FB_LABELS[fb]:<5} & {tm:>6.1f} & "
          f"[{lo:.1f},\\ {hi:.1f}] & "
          f"--- & --- \\\\")
print(r"\bottomrule")
print(r"\end{tabular}")
print(r"\end{table}")
print("\nNote: fill peak freq (Hz) and Behaviour from Table I values.")

end_time = datetime.now()
print(f"\nEnd time:   {end_time}")
elapsed = end_time - start_time
elapsed_seconds = int(elapsed.total_seconds())
hours, remainder = divmod(elapsed_seconds, 3600)
minutes, seconds = divmod(remainder, 60)
print(f"Running time: {hours}:{minutes:02d}:{seconds:02d}")