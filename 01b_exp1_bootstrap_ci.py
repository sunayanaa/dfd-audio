# =============================================================================
# Program      : 01b_exp1_bootstrap_ci.py
# Version      : 1.0
# Description  : Bootstrap 95% confidence intervals for Experiment 1
#                synthetic validation results.
#
#                Computes CIs on DFD-bar for Experiment 1 (P3 epsilon
#                sweep) by resampling filters (B=500). Raw per-clip
#                energy matrices were not saved by Program 01, so
#                filter-resampling is used as a proxy for clip-level
#                uncertainty. P1 T_m values are reported as exact point
#                estimates from the saved JSON (clip-level CIs for T_m
#                require rerunning Program 01 with energy matrix saving).
#
#                All inputs are copied from Google Drive; output is copied
#                to Google Drive on completion.
#
# INPUT (Google Drive):
#   exp1_synthetic_results.json   DFD vectors and Tm values from Exp 1
#   exp2_energy_matrices.npz      Copied but not used in this cell;
#                                 retained for future clip-level CI extension
#
# OUTPUT (Google Drive):
#   exp1_ci_results.json          95% CIs for P3 DFD-bar per filterbank
#                                 per epsilon; exact Tm_max and peak_hz
#                                 for P1 per filterbank
#
# METHOD:
#   P3 DFD-bar CI: resample M filter indices with replacement from the
#   saved per-filter DFD vector (length M); repeat B=500 times; take
#   2.5th and 97.5th percentiles. This captures filter-to-filter
#   variance but not clip-to-clip variance.
#
#   P1 Tm CI: not computed (requires raw energy matrices). Point
#   estimates from exp1_synthetic_results.json are reported instead.
#
# NOTE:
#   Several P3 CIs cross zero (Mel/eps=0.01, CQT/eps>=0.05), reflecting
#   high filter-to-filter variance at large detuning. Only cells whose
#   CI excludes zero should be cited as significant in the paper.
#   To be run after 01_synthetic_validation.py
# GPU Required : NO
# Dependencies : numpy, scipy
# Running time : < 1 minute
#
# Change Log   :
#   v1.0  2026-09-18  Initial version
# =============================================================================

from datetime import datetime
import numpy as np
from scipy.stats import gaussian_kde
from scipy.spatial import KDTree
import json, os

from google.colab import drive
drive.mount('/content/drive')

# --- Google Drive config ---
PROJECT_DIR = "/content/drive/MyDrive/papers/dfd-audio/"
LOCAL_DIR   = "/tmp/dfd_ci"
os.makedirs(LOCAL_DIR, exist_ok=True)
os.makedirs(PROJECT_DIR, exist_ok=True)


def download(remote, local, skip_if_exists=True):
    """Copy a file from Google Drive project directory to local."""
    drive_path = os.path.join(PROJECT_DIR, remote)
    if not os.path.exists(drive_path):
        print(f"[Drive] {remote} not found on Drive.")
        return False
    if skip_if_exists and os.path.exists(local):
        if os.path.getsize(local) == os.path.getsize(drive_path):
            return True
    os.makedirs(os.path.dirname(local), exist_ok=True)
    tmp_path = local + ".partial"
    with open(drive_path, "rb") as fsrc, open(tmp_path, "wb") as fdst:
        chunk_size = 16 * 1024 * 1024
        while True:
            buf = fsrc.read(chunk_size)
            if not buf:
                break
            fdst.write(buf)
    os.replace(tmp_path, local)
    print(f"  Copied from Drive: {remote}")
    return True


def upload(local, remote):
    """Copy a file from local to Google Drive project directory."""
    drive_path = os.path.join(PROJECT_DIR, remote)
    tmp_path = drive_path + ".partial"
    with open(local, "rb") as fsrc, open(tmp_path, "wb") as fdst:
        chunk_size = 16 * 1024 * 1024
        while True:
            buf = fsrc.read(chunk_size)
            if not buf:
                break
            fdst.write(buf)
    os.replace(tmp_path, drive_path)
    print(f"  Copied to Drive: {remote}")


# --- Copy required files from Google Drive ---
print("Copying from Google Drive...")
download("exp1_synthetic_results.json",
         os.path.join(LOCAL_DIR, "exp1_synthetic_results.json"))
download("exp2_energy_matrices.npz",
         os.path.join(LOCAL_DIR, "exp2_energy_matrices.npz"))

with open(os.path.join(LOCAL_DIR, "exp1_synthetic_results.json")) as f:
    r = json.load(f)

# --- Constants ---
RANDOM_SEED = 42
K_KNN       = 5
N_KDE_SAMP  = 500
N_BOOT      = 500
FILTERBANKS = ["mel","bark","erb","cqt"]
FB_LABELS   = {"mel":"Mel","bark":"Bark","erb":"ERB","cqt":"CQT"}
EPSILONS    = [0.0, 0.01, 0.05, 0.10]

# --- KNN KL estimator ---
def knn_kl(p_samp, q_samp, k=K_KNN):
    n, m = len(p_samp), len(q_samp)
    if n < k+1 or m < k:
        return 0.0
    p = p_samp.reshape(-1,1)
    q = q_samp.reshape(-1,1)
    r_k = KDTree(p).query(p, k=k+1)[0][:,k]
    s_k = KDTree(q).query(p, k=k)[0][:,k-1]
    mask = (r_k>1e-10)&(s_k>1e-10)
    if mask.sum() == 0:
        return 0.0
    return float(max(0.0,
        np.log(s_k[mask]/r_k[mask]).mean() + np.log(m/(n-1))))

def fit_kdes(E):
    return [gaussian_kde(E[:,m], bw_method="silverman")
            if E[:,m].std()>1e-10 else None
            for m in range(E.shape[1])]

def per_filter_kl(kdes, E_q):
    return np.array([
        knn_kl(kdes[m].resample(N_KDE_SAMP, seed=RANDOM_SEED).flatten(),
               E_q[:,m]) if kdes[m] else 0.0
        for m in range(len(kdes))])

def bootstrap_dfd_ci(E_phi_A, E_lin_A, E_phi_B, E_lin_B,
                     n_boot=N_BOOT, ci=95, seed=RANDOM_SEED):
    rng  = np.random.default_rng(seed)
    nA, nB = len(E_phi_A), len(E_phi_B)
    boots = []
    for _ in range(n_boot):
        iA = rng.integers(0, nA, size=nA)
        iB = rng.integers(0, nB, size=nB)
        kl_phi = per_filter_kl(fit_kdes(E_phi_A[iA]), E_phi_B[iB])
        kl_lin = per_filter_kl(fit_kdes(E_lin_A[iA]), E_lin_B[iB])
        boots.append((kl_phi - kl_lin).mean())
    boots = np.array(boots)
    lo = float(np.percentile(boots, (100-ci)/2))
    hi = float(np.percentile(boots, 100-(100-ci)/2))
    return lo, hi, float(boots.std())

# --- Reconstruct synthetic energy matrices from exp1 JSON ---
# The JSON stores kl_phi and kl_lin vectors per pair per filterbank.
# We cannot re-derive raw energy matrices from the JSON alone.
# Instead we bootstrap over the KL vectors directly (scalar bootstrap):
# resample the per-filter DFD values across filters to get CI on DFD-bar.

def scalar_bootstrap_ci(dfd_vec, n_boot=N_BOOT, ci=95, seed=RANDOM_SEED):
    """Bootstrap CI on DFD-bar by resampling filters."""
    rng   = np.random.default_rng(seed)
    dfd   = np.array(dfd_vec)
    boots = [rng.choice(dfd, size=len(dfd), replace=True).mean()
             for _ in range(n_boot)]
    boots = np.array(boots)
    lo = float(np.percentile(boots, (100-ci)/2))
    hi = float(np.percentile(boots, 100-(100-ci)/2))
    return lo, hi, float(boots.std())

# --- P3: CIs on DFD-bar per filterbank per epsilon ---
print("\nP3 bootstrap 95% CIs on DFD-bar (filter-resampling):")
print(f"{'FB':<6} {'eps':>6}  {'DFD-bar':>8}  {'95% CI':>20}  {'SE':>7}")
print("-"*58)

p3_ci = {}
for eps in EPSILONS:
    p3_ci[str(eps)] = {}
    for fb in FILTERBANKS:
        dfd_vec = r['P3_results'][str(eps)][fb]['dfd_vec']
        dfd_bar = r['P3_results'][str(eps)][fb]['dfd_bar']
        lo, hi, se = scalar_bootstrap_ci(dfd_vec)
        p3_ci[str(eps)][fb] = {"dfd_bar":dfd_bar,"ci_lo":lo,"ci_hi":hi,"se":se}
        print(f"{FB_LABELS[fb]:<6} {eps:>6.2f}  {dfd_bar:>8.4f}  "
              f"[{lo:>7.4f}, {hi:>7.4f}]  {se:>7.4f}")

# --- P1: CIs on Tm_max per filterbank ---
print("\nP1 Tm_max values (point estimates from JSON):")
print(f"{'FB':<6}  {'Tm_max':>8}  {'peak_hz':>10}")
print("-"*30)
for fb in FILTERBANKS:
    tm  = r['P1_Tm'][fb]['Tm_max']
    hz  = r['P1_Tm'][fb]['peak_hz']
    print(f"{FB_LABELS[fb]:<6}  {tm:>8.2f}  {hz:>10.1f}")

print("\nNote: Tm_max CIs require raw clip energy matrices.")
print("These were not saved to Drive by Program 01.")
print("Reporting exact point estimates in Table I instead of approximations.")

# --- Save CI results ---
ci_results = {"P3_ci": p3_ci,
              "P1_Tm_exact": {fb: r['P1_Tm'][fb] for fb in FILTERBANKS}}
out = os.path.join(LOCAL_DIR, "exp1_ci_results.json")
with open(out, "w") as f:
    json.dump(ci_results, f, indent=2)
upload(out, "exp1_ci_results.json")
print("\nDone. exp1_ci_results.json copied to Google Drive.")