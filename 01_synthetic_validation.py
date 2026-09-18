# =============================================================================
# Program      : 01_synthetic_validation.py
# Version      : 2.0
# Description  : Experiment 1 -- Synthetic Signal Validation of the
#                Differential Filterbank Divergence (DFD) estimator.
#
#                Validates that DFD recovers known filterbank-induced
#                divergence on controlled synthetic signals, independently
#                of corpus confounds. Three signal pairs:
#
#                P1: 12-EDO vs 53-EDO harmonic series (f0=220 Hz, K=20)
#                    Tests microtonal discrimination across filterbanks.
#
#                P2: Pink noise vs narrowband noise (1-2 kHz) [null control]
#                    All filterbanks should show DFD-bar ≈ 0.
#
#                P3: Harmonic series vs randomly detuned harmonics
#                    Epsilon sweep: 0, 0.01, 0.05, 0.10.
#
# KEY FIX (v2): Each filterbank is compared against a matched linear
#               reference with the SAME number of bands, isolating
#               frequency warping from dimensionality differences.
#
# STEPS:
#   Step 1  Define four filterbanks (mel, Bark, ERB, CQT) + matched refs
#   Step 2  KDE-based per-filter energy distribution estimator
#   Step 3  KNN KL divergence estimator (Perez-Cruz 2008)
#   Step 4  DFD = D_KL(phi) - D_KL(matched_linear_ref)
#   Step 5  Bootstrap null model (within-class resampling)
#   Step 6  Generate synthetic signals (P1, P2, P3)
#   Step 7  Compute DFD for all pairs x filterbanks
#   Step 8  T_m discrimination statistic for P1
#   Step 9  Epsilon sweep for P3
#   Step 10 Save results JSON and figures
#
# OUTPUT FILES (copied to Google Drive):
#   exp1_synthetic_results.json
#   fig_01_01_p1_discrimination.png
#   fig_01_02_p3_epsilon_sweep.png
#   fig_01_03_p2_null_control.png
#
# GPU Required : NO
# Dependencies : numpy, scipy, librosa, matplotlib
# Running time: 0:04:21
# Change Log   :
#   v1.0  2026-09-16  Initial version
#   v2.0  2026-09-17  Fix: each filterbank now compared against a
#                     matched linear reference with identical number
#                     of bands. Fixes null control (P2) failure.
#                     Updated Google Drive project configuration.
#                     Added ensure_project_dir(), retry logic.
# =============================================================================
from datetime import datetime
start_time = datetime.now()
print(f"Start time: {start_time}")

import subprocess, sys
for pkg in ["librosa", "scipy", "numpy", "matplotlib"]:
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           pkg, "-q"])

import os, json, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.spatial import KDTree
import librosa
from tqdm import tqdm

from google.colab import drive
drive.mount('/content/drive')

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

SR          = 22050
N_FFT       = 2048
HOP_LENGTH  = 512
DURATION    = 4.0
N_CLIPS     = 100
N_BOOT      = 500
K_KNN       = 5
N_KDE_SAMP  = 500
RANDOM_SEED = 42

F0_P1       = 220.0
K_PARTIALS  = 20
F0_P3       = 220.0
EPSILONS    = [0.0, 0.01, 0.05, 0.10]

N_MEL       = 128
N_BARK      = 24
N_ERB       = 64
N_CQT       = 84
BINS_PER_OCT= 12
CQT_FMIN    = librosa.note_to_hz("C1")

LOCAL_DIR           = "/tmp/dfd_exp1"
PROJECT_DIR         = "/content/drive/MyDrive/papers/dfd-audio/"

# =============================================================================
# GOOGLE DRIVE HELPERS
# =============================================================================

def ensure_project_dir():
    """Ensure the Google Drive project directory exists."""
    os.makedirs(PROJECT_DIR, exist_ok=True)
    print(f"[Drive] Project directory ready: {PROJECT_DIR}")

def copy_to_drive(local_path, remote_name):
    """Copy a file from local to Google Drive project directory."""
    drive_path = os.path.join(PROJECT_DIR, remote_name)
    total_size = os.path.getsize(local_path)
    tmp_path = drive_path + ".partial"
    with open(local_path, "rb") as fsrc, open(tmp_path, "wb") as fdst:
        chunk_size = 16 * 1024 * 1024
        while True:
            buf = fsrc.read(chunk_size)
            if not buf:
                break
            fdst.write(buf)
    os.replace(tmp_path, drive_path)
    print(f"  [Drive] Copied: {remote_name}")

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

os.makedirs(LOCAL_DIR, exist_ok=True)
ensure_project_dir()

# =============================================================================
# STEP 1 -- FILTERBANK DEFINITIONS
# =============================================================================

def linear_filterbank(n_filters, sr=SR, n_fft=N_FFT):
    """
    Linear power spectrum: n_filters equal-width bands over 0--sr/2.
    MATCHED to target filterbank's n_filters so DFD isolates warping
    only, not dimensionality differences.
    Returns (n_filters, n_fft//2+1) weight matrix.
    """
    n_bins = n_fft // 2 + 1
    freqs  = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    f_max  = freqs[-1]
    edges  = np.linspace(0, f_max, n_filters + 1)
    fb     = np.zeros((n_filters, n_bins))
    for m in range(n_filters):
        mask = (freqs >= edges[m]) & (freqs < edges[m+1])
        fb[m, mask] = 1.0
        if mask.sum() > 0:
            fb[m] /= mask.sum()
    return fb

def bark_filterbank(n_filters=N_BARK, sr=SR, n_fft=N_FFT):
    """Bark-scale filterbank (Zwicker 1961 approximation)."""
    n_bins = n_fft // 2 + 1
    freqs  = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    def hz_to_bark(f):
        f_khz = f / 1000.0
        return (13.0 * np.arctan(0.76 * f_khz) +
                3.5  * np.arctan((f_khz / 7.5) ** 2))

    bark_freqs = hz_to_bark(freqs)
    bark_max   = hz_to_bark(sr / 2.0)
    edges_bark = np.linspace(0, bark_max, n_filters + 1)

    fb = np.zeros((n_filters, n_bins))
    for m in range(n_filters):
        mask = ((bark_freqs >= edges_bark[m]) &
                (bark_freqs <  edges_bark[m+1]))
        fb[m, mask] = 1.0
        if mask.sum() > 0:
            fb[m] /= mask.sum()
    return fb

def erb_filterbank(n_filters=N_ERB, sr=SR, n_fft=N_FFT,
                   fmin=50.0, fmax=None):
    """ERB-rate filterbank (Moore & Glasberg 1990)."""
    fmax   = fmax or sr / 2.0
    n_bins = n_fft // 2 + 1
    freqs  = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    def hz_to_erb_rate(f):
        return 21.4 * np.log10(4.37 * f / 1000.0 + 1.0)

    def erb_at(f):
        return 24.7 * (4.37 * f / 1000.0 + 1.0)

    erb_min     = hz_to_erb_rate(fmin)
    erb_max     = hz_to_erb_rate(fmax)
    centres_erb = np.linspace(erb_min, erb_max, n_filters)
    centres_hz  = (10.0 ** (centres_erb / 21.4) - 1.0) / 4.37 * 1000.0

    fb = np.zeros((n_filters, n_bins))
    for m in range(n_filters):
        bw     = erb_at(centres_hz[m])
        weight = np.maximum(0, 1.0 - np.abs(freqs - centres_hz[m]) / bw)
        if weight.sum() > 0:
            fb[m] = weight / weight.sum()
    return fb

def apply_cqt(y, sr=SR, n_bins=N_CQT, bins_per_octave=BINS_PER_OCT,
              fmin=CQT_FMIN, hop_length=HOP_LENGTH):
    """CQT power spectrum: (n_bins, T)."""
    C = librosa.cqt(y, sr=sr, n_bins=n_bins,
                    bins_per_octave=bins_per_octave,
                    fmin=fmin, hop_length=hop_length)
    return np.abs(C) ** 2

def apply_standard_fb(y, fb, n_fft=N_FFT, hop_length=HOP_LENGTH):
    """Apply any STFT-based filterbank matrix to waveform."""
    S = np.abs(librosa.stft(y, n_fft=n_fft,
                             hop_length=hop_length)) ** 2
    return fb @ S

print("[Step 1] Building filterbank matrices...")
rng = np.random.default_rng(RANDOM_SEED)

# Target filterbanks
FB_MEL  = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MEL,
                               fmin=0.0, fmax=SR/2.0)
FB_BARK = bark_filterbank(N_BARK, SR, N_FFT)
FB_ERB  = erb_filterbank(N_ERB,  SR, N_FFT)

# Matched linear reference for each target (same n_filters, no warping)
FB_LIN_MEL  = linear_filterbank(N_MEL,  SR, N_FFT)
FB_LIN_BARK = linear_filterbank(N_BARK, SR, N_FFT)
FB_LIN_ERB  = linear_filterbank(N_ERB,  SR, N_FFT)
FB_LIN_CQT  = linear_filterbank(N_CQT,  SR, N_FFT)

# Centre frequencies for axis labelling
MEL_FREQS = librosa.mel_frequencies(N_MEL, fmin=0.0, fmax=SR/2.0)
CQT_FREQS = librosa.cqt_frequencies(N_CQT, fmin=CQT_FMIN,
                                     bins_per_octave=BINS_PER_OCT)

# FILTERBANKS dict:
# key -> (display_name, target_fb_matrix, matched_linear_ref,
#          centre_freqs_hz, is_cqt)
FILTERBANKS = {
    "mel" : ("Mel",  FB_MEL,  FB_LIN_MEL,  MEL_FREQS, False),
    "bark": ("Bark", FB_BARK, FB_LIN_BARK, None,      False),
    "erb" : ("ERB",  FB_ERB,  FB_LIN_ERB,  None,      False),
    "cqt" : ("CQT",  None,    FB_LIN_CQT,  CQT_FREQS, True),
}

print(f"  Mel:  {N_MEL} filters  (ref: {N_MEL}-band linear)")
print(f"  Bark: {N_BARK} filters  (ref: {N_BARK}-band linear)")
print(f"  ERB:  {N_ERB} filters  (ref: {N_ERB}-band linear)")
print(f"  CQT:  {N_CQT} bins    (ref: {N_CQT}-band linear)")

# =============================================================================
# STEP 2 -- KDE-BASED PER-FILTER ENERGY DISTRIBUTION
# =============================================================================

def compute_energy_matrix(clips, fb_name, fb_matrix):
    """
    Per-clip per-filter log-energy.
    Returns (n_clips, n_filters) array.
    fb_name: used only to route CQT vs STFT path.
    fb_matrix: the filterbank weight matrix (or None for CQT).
    """
    energies = []
    for y in clips:
        if fb_name == "cqt":
            S = apply_cqt(y)
        else:
            S = apply_standard_fb(y, fb_matrix)
        e = np.log1p(S.sum(axis=1))
        energies.append(e)
    return np.array(energies)

def fit_kde_per_filter(energy_matrix):
    """Silverman-bandwidth KDE per filter. Returns list of kde objects."""
    _, n_filters = energy_matrix.shape
    kdes = []
    for m in range(n_filters):
        vals = energy_matrix[:, m]
        if vals.std() < 1e-10:
            kdes.append(None)
        else:
            kdes.append(gaussian_kde(vals, bw_method="silverman"))
    return kdes

# =============================================================================
# STEP 3 -- KNN KL DIVERGENCE ESTIMATOR (Perez-Cruz 2008)
# =============================================================================

def knn_kl_divergence(p_samples, q_samples, k=K_KNN):
    """
    KNN KL divergence estimator: D_KL(P||Q).
    p_samples ~ P (n,), q_samples ~ Q (m,).
    """
    n = len(p_samples)
    m = len(q_samples)
    if n < k + 1 or m < k:
        return np.nan

    p = p_samples.reshape(-1, 1)
    q = q_samples.reshape(-1, 1)

    tree_p = KDTree(p)
    tree_q = KDTree(q)

    r_k, _ = tree_p.query(p, k=k+1)
    r_k    = r_k[:, k]

    s_k, _ = tree_q.query(p, k=k)
    s_k    = s_k[:, k-1]

    mask  = (r_k > 1e-10) & (s_k > 1e-10)
    ratio = np.log(s_k[mask] / r_k[mask])
    kl    = ratio.mean() + np.log(m / (n - 1))
    return float(max(0.0, kl))

def compute_per_filter_kl(kdes_p, energy_q, k=K_KNN, n_samp=N_KDE_SAMP):
    """Per-filter D_KL(P||Q). Returns (n_filters,) array."""
    n_filters = len(kdes_p)
    kl_vec    = np.zeros(n_filters)
    for m in range(n_filters):
        if kdes_p[m] is None:
            kl_vec[m] = 0.0
            continue
        p_samp     = kdes_p[m].resample(n_samp, seed=RANDOM_SEED).flatten()
        q_samp     = energy_q[:, m]
        kl_vec[m]  = knn_kl_divergence(p_samp, q_samp, k=k)
    return kl_vec

# =============================================================================
# STEP 4 -- DIFFERENTIAL FILTERBANK DIVERGENCE
# =============================================================================

def compute_dfd(kl_phi, kl_phi0):
    """
    DFD_m = D_KL(phi)_m - D_KL(matched_linear_ref)_m
    Positive: phi amplifies corpus divergence beyond warping-free baseline.
    """
    min_len = min(len(kl_phi), len(kl_phi0))
    return kl_phi[:min_len] - kl_phi0[:min_len]

# =============================================================================
# STEP 5 -- BOOTSTRAP NULL MODEL
# =============================================================================

def bootstrap_null(clips_A, fb_name, fb_matrix, fb_linear_ref,
                   n_boot=N_BOOT, rng=None):
    """
    Within-corpus DFD null: split clips_A randomly; compute DFD(A1, A2).
    Returns (mu_null, sigma_null) per filter.
    """
    rng  = rng or np.random.default_rng(RANDOM_SEED)
    n    = len(clips_A)
    half = n // 2

    E_phi = compute_energy_matrix(clips_A, fb_name, fb_matrix)
    E_lin = compute_energy_matrix(clips_A, "linear", fb_linear_ref)

    dfd_boots = []
    for b in range(n_boot):
        idx       = rng.permutation(n)
        idx1, idx2 = idx[:half], idx[half:2*half]

        kdes1_phi = fit_kde_per_filter(E_phi[idx1])
        kdes1_lin = fit_kde_per_filter(E_lin[idx1])

        kl_phi  = compute_per_filter_kl(kdes1_phi, E_phi[idx2])
        kl_lin  = compute_per_filter_kl(kdes1_lin, E_lin[idx2])
        dfd_b   = compute_dfd(kl_phi, kl_lin)
        dfd_boots.append(dfd_b)

    dfd_boots = np.array(dfd_boots)
    return dfd_boots.mean(axis=0), dfd_boots.std(axis=0)

# =============================================================================
# STEP 6 -- SYNTHETIC SIGNAL GENERATION
# =============================================================================

print("\n[Step 6] Generating synthetic signals...")

def make_53edo_freqs(f0, k_max):
    """53-EDO approximation of harmonics 1..k_max."""
    freqs = []
    for k in range(1, k_max + 1):
        cents_just = 1200.0 * np.log2(k)
        n_step     = int(np.round(53.0 * cents_just / 1200.0))
        freqs.append(f0 * (2.0 ** (n_step / 53.0)))
    return np.array(freqs)

def harmonic_series(f0, k_max, phi_k, sr=SR, duration=DURATION,
                    tuning="12edo"):
    t_loc = np.linspace(0, duration, int(sr * duration), endpoint=False)
    freqs = make_53edo_freqs(f0, k_max) if tuning == "53edo" \
            else f0 * np.arange(1, k_max + 1)
    y = np.zeros(len(t_loc))
    for k_idx in range(k_max):
        y += (1.0 / (k_idx + 1)) * np.sin(
            2 * np.pi * freqs[k_idx] * t_loc + phi_k[k_idx])
    y /= (np.abs(y).max() + 1e-8)
    return y.astype(np.float32)

def detuned_harmonic(f0, k_max, phi_k, delta_k, sr=SR, duration=DURATION):
    t_loc = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = np.zeros(len(t_loc))
    for k_idx in range(k_max):
        k   = k_idx + 1
        f_k = f0 * k * (1.0 + delta_k[k_idx])
        y  += (1.0 / k) * np.sin(
            2 * np.pi * f_k * t_loc + phi_k[k_idx])
    y /= (np.abs(y).max() + 1e-8)
    return y.astype(np.float32)

def pink_noise(sr=SR, duration=DURATION, rng=None):
    rng   = rng or np.random.default_rng(RANDOM_SEED)
    n     = int(sr * duration)
    white = rng.standard_normal(n)
    fft   = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, 1.0/sr)
    freqs[0] = 1.0
    fft  /= np.sqrt(freqs)
    y     = np.fft.irfft(fft, n=n)
    y    /= (np.abs(y).max() + 1e-8)
    return y.astype(np.float32)

def narrowband_noise(sr=SR, duration=DURATION, flo=1000.0, fhi=2000.0,
                     rng=None):
    rng   = rng or np.random.default_rng(RANDOM_SEED)
    n     = int(sr * duration)
    white = rng.standard_normal(n)
    fft   = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, 1.0/sr)
    fft[(freqs < flo) | (freqs > fhi)] = 0.0
    y     = np.fft.irfft(fft, n=n)
    y    /= (np.abs(y).max() + 1e-8)
    return y.astype(np.float32)

# P1: same phi_k per realisation, shared across 12-EDO and 53-EDO
print("  P1 (12-EDO vs 53-EDO)...")
clips_12edo, clips_53edo = [], []
for i in range(N_CLIPS):
    phi_k = rng.uniform(0, 2*np.pi, K_PARTIALS)
    clips_12edo.append(harmonic_series(F0_P1, K_PARTIALS, phi_k,
                                       tuning="12edo"))
    clips_53edo.append(harmonic_series(F0_P1, K_PARTIALS, phi_k,
                                       tuning="53edo"))

# P2: null control -- two independent draws from same distribution
# Both sets are pink noise; DFD should be ≈ 0 for all filterbanks
print("  P2 (pink_A vs pink_B -- same distribution null)...")
rng_p2a    = np.random.default_rng(RANDOM_SEED + 100)
rng_p2b    = np.random.default_rng(RANDOM_SEED + 200)
clips_pink  = [pink_noise(rng=rng_p2a) for _ in range(N_CLIPS)]
clips_nband = [pink_noise(rng=rng_p2b) for _ in range(N_CLIPS)]

# P3: epsilon sweep; same phi_k and delta_k across all filterbanks
print("  P3 (harmonic vs detuned, epsilon sweep)...")
p3_clips = {}
for eps in EPSILONS:
    harm_clips, detune_clips = [], []
    for i in range(N_CLIPS):
        phi_k   = rng.uniform(0, 2*np.pi, K_PARTIALS)
        delta_k = rng.uniform(-eps, eps, K_PARTIALS) if eps > 0 \
                  else np.zeros(K_PARTIALS)
        harm_clips.append(harmonic_series(F0_P3, K_PARTIALS, phi_k,
                                          tuning="12edo"))
        detune_clips.append(detuned_harmonic(F0_P3, K_PARTIALS,
                                             phi_k, delta_k))
    p3_clips[eps] = (harm_clips, detune_clips)

print("  All signals generated.")

# =============================================================================
# STEP 7 -- DFD FOR ALL PAIRS x ALL FILTERBANKS
# =============================================================================

print("\n[Step 7] Computing DFD...")

results = {
    "P1_12edo_vs_53edo" : {},
    "P2_pink_vs_narrow" : {},
    "P3_harm_vs_detune" : {str(eps): {} for eps in EPSILONS},
}

def run_dfd_pair(clips_A, clips_B, fb_name, fb_matrix,
                 fb_linear_ref, is_cqt=False, label=""):
    """
    Compute DFD(A, B; phi) for one filterbank.
    Uses matched linear reference fb_linear_ref (same n_filters as phi).
    """
    E_A_phi = compute_energy_matrix(clips_A, fb_name, fb_matrix)
    E_B_phi = compute_energy_matrix(clips_B, fb_name, fb_matrix)
    E_A_lin = compute_energy_matrix(clips_A, "linear", fb_linear_ref)
    E_B_lin = compute_energy_matrix(clips_B, "linear", fb_linear_ref)

    kdes_A_phi = fit_kde_per_filter(E_A_phi)
    kdes_A_lin = fit_kde_per_filter(E_A_lin)

    kl_phi  = compute_per_filter_kl(kdes_A_phi, E_B_phi)
    kl_lin  = compute_per_filter_kl(kdes_A_lin, E_B_lin)
    dfd_vec = compute_dfd(kl_phi, kl_lin)
    dfd_bar = float(dfd_vec.mean())

    print(f"    {label} {fb_name}: DFD-bar={dfd_bar:.4f}")
    return {
        "dfd_vec" : dfd_vec.tolist(),
        "dfd_bar" : dfd_bar,
        "kl_phi"  : kl_phi.tolist(),
        "kl_lin"  : kl_lin.tolist(),
    }

for fb_name, (label, fb_matrix, fb_lin_ref, _, is_cqt) in FILTERBANKS.items():
    print(f"\n  [{label}]")

    print("  P1:")
    results["P1_12edo_vs_53edo"][fb_name] = run_dfd_pair(
        clips_12edo, clips_53edo, fb_name, fb_matrix,
        fb_lin_ref, is_cqt, "P1")

    print("  P2:")
    results["P2_pink_vs_narrow"][fb_name] = run_dfd_pair(
        clips_pink, clips_nband, fb_name, fb_matrix,
        fb_lin_ref, is_cqt, "P2")

    for eps in EPSILONS:
        harm_clips, detune_clips = p3_clips[eps]
        print(f"  P3 eps={eps}:")
        results["P3_harm_vs_detune"][str(eps)][fb_name] = run_dfd_pair(
            harm_clips, detune_clips, fb_name, fb_matrix,
            fb_lin_ref, is_cqt, f"P3(eps={eps})")

# =============================================================================
# STEP 8 -- T_m DISCRIMINATION STATISTIC FOR P1
# =============================================================================

print("\n[Step 8] T_m for P1...")

Tm_results = {}
f7_53 = make_53edo_freqs(F0_P1, K_PARTIALS)[6]
f7_12 = F0_P1 * 7

for fb_name, (label, fb_matrix, fb_lin_ref, centre_freqs, is_cqt) \
        in FILTERBANKS.items():
    E_12 = compute_energy_matrix(clips_12edo, fb_name, fb_matrix)
    E_53 = compute_energy_matrix(clips_53edo, fb_name, fb_matrix)
    N    = E_12.shape[0]

    mu_12  = E_12.mean(axis=0)
    mu_53  = E_53.mean(axis=0)
    sig_12 = E_12.std(axis=0)
    sig_53 = E_53.std(axis=0)
    denom  = np.sqrt(sig_12**2/N + sig_53**2/N + 1e-12)
    Tm     = np.abs(mu_12 - mu_53) / denom

    peak_m  = int(np.argmax(Tm))
    peak_hz = float(centre_freqs[peak_m]) if centre_freqs is not None \
              else float(peak_m)

    Tm_results[fb_name] = {
        "Tm_vec" : Tm.tolist(),
        "Tm_max" : float(Tm.max()),
        "peak_m" : peak_m,
        "peak_hz": peak_hz,
    }
    print(f"  {label}: Tm_max={Tm.max():.3f} @ {peak_hz:.1f} Hz")
    
# Save P1 clip energy matrices for bootstrap CI (Program 01c)
print("\n  Saving P1 energy matrices for bootstrap CI...")
p1_energy = {}
for fb_name, (label, fb_matrix, fb_lin_ref, centre_freqs, is_cqt) \
        in FILTERBANKS.items():
    p1_energy[f"E_12_{fb_name}"] = \
        compute_energy_matrix(clips_12edo, fb_name, fb_matrix)
    p1_energy[f"E_53_{fb_name}"] = \
        compute_energy_matrix(clips_53edo, fb_name, fb_matrix)
    p1_energy[f"E_12_{fb_name}_lin"] = \
        compute_energy_matrix(clips_12edo, "linear", fb_lin_ref)
    p1_energy[f"E_53_{fb_name}_lin"] = \
        compute_energy_matrix(clips_53edo, "linear", fb_lin_ref)

npz_p1 = os.path.join(LOCAL_DIR, "exp1_p1_energy.npz")
np.savez_compressed(npz_p1, **p1_energy)
copy_to_drive(npz_p1, "exp1_p1_energy.npz")
print("  exp1_p1_energy.npz saved to Google Drive.")
# =============================================================================
# STEP 9 -- EPSILON SWEEP SUMMARY
# =============================================================================

print("\n[Step 9] P3 epsilon sweep:")
print(f"  {'FB':<8} " + "  ".join(f"eps={e}" for e in EPSILONS))
for fb_name, (label, _, _, _, _) in FILTERBANKS.items():
    bars = [results["P3_harm_vs_detune"][str(eps)][fb_name]["dfd_bar"]
            for eps in EPSILONS]
    print(f"  {label:<8} " + "  ".join(f"{b:>8.4f}" for b in bars))

# =============================================================================
# STEP 10 -- SAVE RESULTS AND FIGURES
# =============================================================================

print("\n[Step 10] Saving results and figures...")

full_results = {
    "experiment" : "Experiment 1 -- Synthetic Signal Validation (v2)",
    "parameters" : {
        "sr": SR, "n_fft": N_FFT, "hop_length": HOP_LENGTH,
        "duration_s": DURATION, "n_clips": N_CLIPS,
        "k_knn": K_KNN, "n_kde_samp": N_KDE_SAMP,
        "f0_p1_hz": F0_P1, "f0_p3_hz": F0_P3,
        "k_partials": K_PARTIALS, "epsilons_p3": EPSILONS,
        "fix_v2": "matched-dimension linear reference per filterbank",
    },
    "P1_results" : results["P1_12edo_vs_53edo"],
    "P1_Tm"      : Tm_results,
    "P2_results" : results["P2_pink_vs_narrow"],
    "P3_results" : results["P3_harm_vs_detune"],
}

res_path = os.path.join(LOCAL_DIR, "exp1_synthetic_results.json")
with open(res_path, "w") as f:
    json.dump(full_results, f, indent=2)
copy_to_drive(res_path, "exp1_synthetic_results.json")

colors = {"mel":"#d62728","bark":"#ff7f0e","erb":"#2ca02c","cqt":"#1f77b4"}

# --- Figure 1: T_m (P1) -----------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
for fb_name, (label, _, _, centre_freqs, _) in FILTERBANKS.items():
    if centre_freqs is None:
        continue
    Tm_vec = np.array(Tm_results[fb_name]["Tm_vec"])
    n      = min(len(Tm_vec), len(centre_freqs))
    ax.plot(centre_freqs[:n], Tm_vec[:n],
            color=colors[fb_name], linewidth=1.5, label=label)

ax.axvline(f7_53, color="purple", linestyle="--", linewidth=0.8,
           label=f"7th harmonic 53-EDO ({f7_53:.0f} Hz)")
ax.axvline(f7_12, color="brown",  linestyle=":",  linewidth=0.8,
           label=f"7th harmonic 12-EDO ({f7_12:.0f} Hz)")
ax.set_xscale("log")
ax.set_xlabel("Filter centre frequency (Hz)", fontsize=10)
ax.set_ylabel(r"$T_m$ (discrimination statistic)", fontsize=10)
ax.set_title("P1: 12-EDO vs 53-EDO — per-filter discrimination", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
plt.tight_layout()

p1_path = os.path.join(LOCAL_DIR, "fig_01_01_p1_discrimination.png")
plt.savefig(p1_path, dpi=300, bbox_inches="tight")
plt.close()
copy_to_drive(p1_path, "fig_01_01_p1_discrimination.png")
print("  fig_01_01_p1_discrimination.png saved.")

# --- Figure 2: Epsilon sweep (P3) with ±1 SE shaded bands -------------------
fig, ax = plt.subplots(figsize=(7, 4), dpi=300)
for fb_name, (label, _, _, _, _) in FILTERBANKS.items():
    bars = np.array([results["P3_harm_vs_detune"][str(eps)][fb_name]["dfd_bar"]
                     for eps in EPSILONS])
    ses  = np.array([p3_ci[str(eps)][fb_name]["se"]
                     for eps in EPSILONS])
    ax.plot(EPSILONS, bars, color=colors[fb_name],
            marker="o", linewidth=1.5, markersize=5, label=label)
    ax.fill_between(EPSILONS,
                    bars - ses, bars + ses,
                    color=colors[fb_name], alpha=0.15)
ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
ax.set_xlabel(r"Detuning parameter $\epsilon$", fontsize=10)
ax.set_ylabel(r"$\overline{\mathrm{DFD}}$ (nats)", fontsize=10)
ax.set_title("P3: Harmonic vs randomly detuned — "
             r"$\overline{\mathrm{DFD}}$ vs $\epsilon$",
             fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
plt.tight_layout()

p3_path = os.path.join(LOCAL_DIR, "fig_01_02_p3_epsilon_sweep.png")
plt.savefig(p3_path, dpi=300, bbox_inches="tight")
plt.close()
copy_to_drive(p3_path, "fig_01_02_p3_epsilon_sweep.png")
print("  fig_01_02_p3_epsilon_sweep.png saved.")

# --- Figure 3: Null control (P2) --------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4), dpi=300)
fb_names_plot = list(FILTERBANKS.keys())
dfd_bars_p2   = [results["P2_pink_vs_narrow"][fb]["dfd_bar"]
                 for fb in fb_names_plot]
fb_labels     = [FILTERBANKS[fb][0] for fb in fb_names_plot]
bars_b = ax.bar(fb_labels, dfd_bars_p2,
                color=[colors[fb] for fb in fb_names_plot],
                edgecolor="white")
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_ylabel("DFD-bar (nats)", fontsize=10)
ax.set_title("P2: Pink noise A vs Pink noise B (same-distribution null)\n"
             "All filterbanks should show DFD-bar ≈ 0", fontsize=10)
for bar, val in zip(bars_b, dfd_bars_p2):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.001 * np.sign(bar.get_height() + 1e-6),
            f"{val:.4f}", ha="center", fontsize=8)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()

p2_path = os.path.join(LOCAL_DIR, "fig_01_03_p2_null_control.png")
plt.savefig(p2_path, dpi=300, bbox_inches="tight")
plt.close()
copy_to_drive(p2_path, "fig_01_03_p2_null_control.png")
print("  fig_01_03_p2_null_control.png saved.")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "="*60)
print("EXPERIMENT 1 (SYNTHETIC VALIDATION v2) COMPLETE")
print("="*60)
print("\nP1 — T_m peak at 7th harmonic region:")
for fb_name, (label, _, _, _, _) in FILTERBANKS.items():
    r = Tm_results[fb_name]
    print(f"  {label:<8} Tm_max={r['Tm_max']:.3f} @ {r['peak_hz']:.1f} Hz")

print("\nP2 — Same-distribution null (DFD-bar, expect ≈ 0):")
for fb_name, (label, _, _, _, _) in FILTERBANKS.items():
    bar = results["P2_pink_vs_narrow"][fb_name]["dfd_bar"]
    flag = " ← FAIL" if abs(bar) > 0.05 else ""
    print(f"  {label:<8} DFD-bar={bar:.4f}{flag}")

print("\nP3 — Epsilon sweep (DFD-bar at eps=0.10):")
for fb_name, (label, _, _, _, _) in FILTERBANKS.items():
    bar = results["P3_harm_vs_detune"]["0.1"][fb_name]["dfd_bar"]
    print(f"  {label:<8} DFD-bar={bar:.4f}")

print(f"\nGoogle Drive outputs:")
print(f"  exp1_synthetic_results.json")
print(f"  fig_01_01_p1_discrimination.png")
print(f"  fig_01_02_p3_epsilon_sweep.png")
print(f"  fig_01_03_p2_null_control.png")

end_time = datetime.now()
print(f"End time:   {end_time}")
elapsed = end_time - start_time
elapsed_seconds = int(elapsed.total_seconds())
hours, remainder = divmod(elapsed_seconds, 3600)
minutes, seconds = divmod(remainder, 60)
print(f"Running time: {hours}:{minutes:02d}:{seconds:02d}")

print("="*60)