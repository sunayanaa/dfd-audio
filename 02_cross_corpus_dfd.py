# =============================================================================
# Program      : 02_cross_corpus_dfd.py
# Version      : 1.0
# GPU Required : NO
# Description  : Experiment 2 -- Cross-Corpus DFD Characterisation.
#
#                Applies the Differential Filterbank Divergence (DFD)
#                estimator to real music corpora. Computes DFD-bar and
#                bootstrap z-scores for four filterbanks (Mel, Bark, ERB,
#                CQT) against the GTZAN Western reference, across five
#                non-Western corpora (Carnatic, Hindustani, Turkish Makam,
#                Jingju, Arab-Andalusian). FMA-Small provides a
#                Western-Western robustness baseline (noise floor).
#
#                All filterbank definitions, KDE estimator, and KNN KL
#                divergence estimator are identical to Program 01.
#
#                DFD(A, B; φ)_m = D_KL(φ)_m - D_KL(φ_0^(n))_m
#                where φ_0^(n) is a matched linear reference with the
#                same number of bands as φ.
#
#                Bootstrap null: within-GTZAN resampling (B=500 splits)
#                gives per-filter (mu_null, sigma_null). Cross-corpus
#                DFD is reported as z-score = (DFD-bar - mu_null) / sigma_null.
#
# INPUT (Google Drive):
#   datasets/GTZAN.zip                      genres/<genre>/<file>.wav
#   datasets/FMA-small.zip                  fma_small/<tid>/<file>.mp3
#   datasets/Carnatic/CMR_subset_1.0.zip    CMR_subset_1.0/audio/<file>.wav
#   datasets/Hindustani/HMR_1.0.zip         HMR_1.0/audio/<file>.wav
#   datasets/Turkish_Makam/gonul-yazar-*.zip <file>.mp3 (flat)
#   datasets/Jingju/wav_mono-1.zip          wav_left/{danAll,laosheng}/<file>.wav
#   datasets/Arab_Andalusian/arab-andalusian.zip
#                                           <file>.m4a/.webm (flat, long-form)
#
# STEPS:
#   Step 1  Mount Drive; build filterbanks
#   Step 2  Copy zips to local disk; extract
#   Step 3  Build audio file index for each corpus
#   Step 4  Compute per-corpus per-filter energy matrices (checkpoint each)
#   Step 5  Bootstrap null from GTZAN (within-corpus splits)
#   Step 6  Compute DFD-bar and z-scores for all corpora x filterbanks
#   Step 7  Save results JSON
#
# OUTPUT FILES (copied to Google Drive):
#   exp2_energy_matrices.npz      Per-corpus per-filter energy (float32)
#   exp2_checkpoint.json          Completed corpora + metadata
#   exp2_dfd_results.json         DFD-bar, z-score, peak filter per corpus/fb
#
# Dependencies : librosa, scipy, numpy, matplotlib, resampy, tqdm
# Program running time: 1:26:21
# Change Log   :
#   v1.0  2026-09-17  Initial version
# =============================================================================

!pip install resampy
from datetime import datetime
start_time = datetime.now()
print(f"Start time: {start_time}")



import subprocess, sys
for pkg in ["librosa", "scipy", "numpy", "resampy", "tqdm"]:
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           pkg, "-q"])

# Force resampy into the current session after install
import importlib, site
importlib.invalidate_caches()
try:
    import resampy
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "resampy", "--force-reinstall", "-q"])
    importlib.invalidate_caches()
    import resampy

# Force librosa to reload its audio module with resampy available
import importlib
import sys

# Remove cached failed lazy-load state
for mod in list(sys.modules.keys()):
    if 'librosa' in mod or 'resampy' in mod or 'lazy' in mod:
        sys.modules.pop(mod, None)

# Now import resampy explicitly first, then librosa
import resampy
import librosa

import os, json, zipfile, shutil, warnings
import numpy as np
from scipy.stats import gaussian_kde
from scipy.spatial import KDTree
from tqdm import tqdm
import librosa
from scipy.stats import spearmanr

from google.colab import drive
drive.mount("/content/drive")


warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

DRIVE_DIR   = "/content/drive/MyDrive/datasets"
LOCAL_DIR   = "/tmp/dfd_exp2"
SR          = 22050
N_FFT       = 2048
HOP_LENGTH  = 512
CLIP_DUR    = 30.0          # seconds per clip (truncate / pad)
ARAB_CHUNK  = 180           # seconds per chunk for long-form files
N_CLIPS_MAX = 100           # subsample each corpus to this many clips
FMA_MAX     = 100           # FMA-Small subsample
RANDOM_SEED = 42
N_BOOT      = 500           # bootstrap resamples for null model
K_KNN       = 5
N_KDE_SAMP  = 500

# Filterbank dimensions
N_MEL       = 128
N_BARK      = 24
N_ERB       = 64
N_CQT       = 84
BINS_PER_OCT= 12
CQT_FMIN    = librosa.note_to_hz("C1")

# Google Drive project directory
PROJECT_DIR = "/content/drive/MyDrive/papers/dfd-audio/"

CHECKPOINT_FILE = "exp2_checkpoint.json"

# Drive zip paths
ZIPS = {
    "gtzan"       : ("GTZAN.zip",                                  None),
    "fma"         : ("FMA-small.zip",                              None),
    "carnatic"    : ("Carnatic/CMR_subset_1.0.zip",                None),
    "hindustani"  : ("Hindustani/HMR_1.0.zip",                     None),
    "turkish_makam": ("Turkish_Makam/gonul-yazar-sari-gulum-var-benim.zip", None),
    "jingju"      : ("Jingju/wav_mono-1.zip",                      None),
    "arab_andalusian": ("Arab_Andalusian/arab-andalusian.zip",     ARAB_CHUNK),
}
# chunk_dur = None means standard CLIP_DUR; integer means long-form chunking

# =============================================================================
# GOOGLE DRIVE HELPERS
# =============================================================================

def ensure_project_dir():
    os.makedirs(PROJECT_DIR, exist_ok=True)
    print(f"[Drive] Project dir ready: {PROJECT_DIR}")


def copy_to_drive(local_path, remote_name):
    drive_path = os.path.join(PROJECT_DIR, remote_name)
    tmp_path   = drive_path + ".partial"
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
    return True

os.makedirs(LOCAL_DIR, exist_ok=True)
ensure_project_dir()

# =============================================================================
# CHECKPOINT HELPERS
# =============================================================================

def load_checkpoint():
    local_ckpt = os.path.join(LOCAL_DIR, CHECKPOINT_FILE)
    if copy_from_drive(CHECKPOINT_FILE, local_ckpt):
        with open(local_ckpt) as f:
            ck = json.load(f)
        print(f"[RESUME] Completed: {ck.get('completed', [])}")
        return ck
    return {"completed": [], "meta": {}}

def save_checkpoint(ck):
    local_ckpt = os.path.join(LOCAL_DIR, CHECKPOINT_FILE)
    with open(local_ckpt, "w") as f:
        json.dump(ck, f, indent=2)
    copy_to_drive(local_ckpt, CHECKPOINT_FILE)

# =============================================================================
# STEP 1 -- MOUNT DRIVE; BUILD FILTERBANKS
# =============================================================================

print("[Step 1] Drive mounted.")

def linear_filterbank(n_filters, sr=SR, n_fft=N_FFT):
    """n_filters equal-width bands, no warping. Matched reference."""
    n_bins = n_fft // 2 + 1
    freqs  = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    edges  = np.linspace(0, freqs[-1], n_filters + 1)
    fb     = np.zeros((n_filters, n_bins))
    for m in range(n_filters):
        mask = (freqs >= edges[m]) & (freqs < edges[m+1])
        fb[m, mask] = 1.0
        if mask.sum() > 0:
            fb[m] /= mask.sum()
    return fb

def bark_filterbank(n_filters=N_BARK, sr=SR, n_fft=N_FFT):
    n_bins = n_fft // 2 + 1
    freqs  = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    def hz_to_bark(f):
        fk = f / 1000.0
        return 13.0*np.arctan(0.76*fk) + 3.5*np.arctan((fk/7.5)**2)
    bf     = hz_to_bark(freqs)
    edges  = np.linspace(0, hz_to_bark(sr/2.0), n_filters + 1)
    fb     = np.zeros((n_filters, n_bins))
    for m in range(n_filters):
        mask = (bf >= edges[m]) & (bf < edges[m+1])
        fb[m, mask] = 1.0
        if mask.sum() > 0:
            fb[m] /= mask.sum()
    return fb

def erb_filterbank(n_filters=N_ERB, sr=SR, n_fft=N_FFT, fmin=50.0):
    n_bins = n_fft // 2 + 1
    freqs  = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    def hz_to_erb_rate(f):
        return 21.4 * np.log10(4.37 * f / 1000.0 + 1.0)
    def erb_at(f):
        return 24.7 * (4.37 * f / 1000.0 + 1.0)
    centres_erb = np.linspace(hz_to_erb_rate(fmin),
                               hz_to_erb_rate(sr/2.0), n_filters)
    centres_hz  = (10.0**(centres_erb/21.4) - 1.0) / 4.37 * 1000.0
    fb = np.zeros((n_filters, n_bins))
    for m in range(n_filters):
        w = np.maximum(0, 1.0 - np.abs(freqs - centres_hz[m]) / erb_at(centres_hz[m]))
        if w.sum() > 0:
            fb[m] = w / w.sum()
    return fb

FB_MEL       = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MEL,
                                    fmin=0.0, fmax=SR/2.0)
FB_BARK      = bark_filterbank()
FB_ERB       = erb_filterbank()
FB_LIN_MEL   = linear_filterbank(N_MEL)
FB_LIN_BARK  = linear_filterbank(N_BARK)
FB_LIN_ERB   = linear_filterbank(N_ERB)
FB_LIN_CQT   = linear_filterbank(N_CQT)

# (display_name, target_fb, matched_linear_ref, n_filters, is_cqt)
FILTERBANKS = {
    "mel" : ("Mel",  FB_MEL,  FB_LIN_MEL,  N_MEL,  False),
    "bark": ("Bark", FB_BARK, FB_LIN_BARK, N_BARK, False),
    "erb" : ("ERB",  FB_ERB,  FB_LIN_ERB,  N_ERB,  False),
    "cqt" : ("CQT",  None,    FB_LIN_CQT,  N_CQT,  True),
}
print("[Step 1] Filterbanks built.")

# =============================================================================
# STEP 2 -- COPY AND EXTRACT
# =============================================================================

def copy_and_extract(zip_rel_path, extract_dir, name):
    zip_drive  = os.path.join(DRIVE_DIR, zip_rel_path)
    local_zip  = os.path.join(LOCAL_DIR, os.path.basename(zip_rel_path))
    if not os.path.exists(local_zip):
        print(f"[Step 2] Copying {name} from Drive...")
        shutil.copy2(zip_drive, local_zip)
    if not os.path.exists(extract_dir):
        print(f"[Step 2] Extracting {name}...")
        with zipfile.ZipFile(local_zip) as zf:
            zf.extractall(extract_dir)
    else:
        print(f"[Step 2] {name}: already extracted.")

extract_dirs = {}
for key, (zip_rel, _) in ZIPS.items():
    d = os.path.join(LOCAL_DIR, key)
    copy_and_extract(zip_rel, d, key)
    extract_dirs[key] = d

print("[Step 2] All corpora extracted.")

# =============================================================================
# STEP 3 -- BUILD AUDIO FILE INDEX
# =============================================================================

def find_audio(root, exts=(".wav",".mp3",".flac",".m4a",".webm",".ogg")):
    paths = []
    for dp, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d != "__MACOSX"]
        for fn in files:
            if not fn.startswith(".") and fn.lower().endswith(exts):
                paths.append(os.path.join(dp, fn))
    return sorted(paths)

rng = np.random.default_rng(RANDOM_SEED)

corpus_files = {}
for key in ZIPS:
    files = find_audio(extract_dirs[key])
    # Subsample to N_CLIPS_MAX
    max_n = FMA_MAX if key == "fma" else N_CLIPS_MAX
    if len(files) > max_n:
        idx   = rng.choice(len(files), size=max_n, replace=False)
        files = [files[i] for i in sorted(idx)]
    corpus_files[key] = files
    print(f"[Step 3] {key}: {len(files)} files")

# =============================================================================
# STEP 4 -- PER-CORPUS PER-FILTER ENERGY MATRICES (checkpoint each corpus)
# =============================================================================

_load_errors = 0

def load_clip(path, sr=SR, duration=CLIP_DUR):
    global _load_errors
    try:
        y, _ = librosa.load(path, sr=sr, mono=True, duration=duration,
                            res_type="kaiser_fast")
        tlen = int(sr * duration)
        y = np.pad(y, (0, max(0, tlen - len(y))))[:tlen]
        return y.astype(np.float32)
    except BaseException as e:
        _load_errors += 1
        if _load_errors <= 3:
            print(f"  [LOAD ERR] {os.path.basename(path)}: {e}")
        return None

def load_chunks(path, sr=SR, chunk_dur=ARAB_CHUNK):
    """Load a long-form file and split into fixed-length chunks."""
    global _load_errors
    chunks = []
    chunk_len = int(sr * chunk_dur)
    try:
        y, _ = librosa.load(path, sr=sr, mono=True, res_type="kaiser_fast")
        n = len(y) // chunk_len
        for i in range(n):
            chunks.append(y[i*chunk_len:(i+1)*chunk_len].astype(np.float32))
        rem = len(y) - n * chunk_len
        if rem > sr * 30:
            last = np.pad(y[n*chunk_len:], (0, chunk_len - rem))
            chunks.append(last.astype(np.float32))
    except BaseException as e:
        _load_errors += 1
        if _load_errors <= 3:
            print(f"  [LOAD ERR] {os.path.basename(path)}: {e}")
    return chunks

def apply_fb(y, fb_name, fb_matrix):
    """Apply filterbank to one waveform. Returns (n_filters, T) power."""
    if fb_name == "cqt":
        C = librosa.cqt(y, sr=SR, n_bins=N_CQT,
                        bins_per_octave=BINS_PER_OCT,
                        fmin=CQT_FMIN, hop_length=HOP_LENGTH)
        return np.abs(C) ** 2
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)) ** 2
    return fb_matrix @ S

def compute_energy_matrix_corpus(files, corpus_key, chunk_dur=None):
    """
    Returns energy_matrices dict keyed by fb_name.
    Each value: (n_clips, n_filters) log-energy array.
    """
    # Collect clips
    clips = []
    if chunk_dur:
        for path in tqdm(files, desc=f"  {corpus_key} (chunks)", ncols=80):
            clips.extend(load_chunks(path, chunk_dur=chunk_dur))
    else:
        for path in tqdm(files, desc=f"  {corpus_key}", ncols=80):
            y = load_clip(path)
            if y is not None:
                clips.append(y)

    if not clips:
        print(f"  [WARN] No clips loaded for {corpus_key} -- skipping.")
        return {
            **{fb: np.zeros((1, FILTERBANKS[fb][3]), dtype=np.float32)
               for fb in FILTERBANKS},
            **{f"{fb}_lin": np.zeros((1, FILTERBANKS[fb][3]), dtype=np.float32)
               for fb in FILTERBANKS},
        }, 0
            
    n_clips = len(clips)
    print(f"  {corpus_key}: {n_clips} clips loaded")

    # Compute energy per filterbank
    mats = {}
    for fb_name, (label, fb_matrix, fb_lin, n_filt, is_cqt) in FILTERBANKS.items():
        E = np.zeros((n_clips, n_filt), dtype=np.float32)
        for i, y in enumerate(clips):
            S  = apply_fb(y, fb_name, fb_matrix)
            E[i] = np.log1p(S.sum(axis=1))
        mats[fb_name] = E

        # Also compute matched linear reference
        lin_key = f"{fb_name}_lin"
        E_lin = np.zeros((n_clips, n_filt), dtype=np.float32)
        for i, y in enumerate(clips):
            S_lin = apply_fb(y, "linear", fb_lin)
            E_lin[i] = np.log1p(S_lin.sum(axis=1))
        mats[lin_key] = E_lin

    return mats, n_clips

ck = load_checkpoint()
completed = set(ck.get("completed", []))

# Load or compute energy matrices
npz_local = os.path.join(LOCAL_DIR, "exp2_energy_matrices.npz")
energy_store = {}

if copy_from_drive("exp2_energy_matrices.npz", npz_local):
    saved = np.load(npz_local, allow_pickle=True)
    for k in saved.files:
        energy_store[k] = saved[k]
    print(f"[Step 4] Loaded energy store: {list(saved.files)[:4]}...")

print("\n[Step 4] Computing energy matrices...")
for corpus_key, (zip_rel, chunk_dur) in ZIPS.items():
    if corpus_key in completed:
        print(f"  {corpus_key}: already done, skipping.")
        continue

    print(f"\n  Processing: {corpus_key} "
          f"[{datetime.now().strftime('%H:%M:%S')}]")
    mats, n_clips = compute_energy_matrix_corpus(
        corpus_files[corpus_key], corpus_key, chunk_dur)

    for fb_key, mat in mats.items():
        energy_store[f"{corpus_key}_{fb_key}"] = mat

    # Save metadata to checkpoint
    ck["meta"][corpus_key] = {
        "n_clips"   : n_clips,
        "duration_h": round(n_clips * (chunk_dur or CLIP_DUR) / 3600, 4),
    }

    # Save npz and checkpoint
    np.savez_compressed(npz_local, **energy_store)
    copy_to_drive(npz_local, "exp2_energy_matrices.npz")

    completed.add(corpus_key)
    ck["completed"] = list(completed)
    save_checkpoint(ck)
    print(f"  {corpus_key}: done and checkpointed.")

print("\n[Step 4] All corpora processed.")

# =============================================================================
# STEP 5 -- BOOTSTRAP NULL FROM GTZAN
# =============================================================================

print("\n[Step 5] Computing bootstrap null from GTZAN...")

def knn_kl(p_samp, q_samp, k=K_KNN):
    n, m = len(p_samp), len(q_samp)
    if n < k+1 or m < k:
        return np.nan
    p = p_samp.reshape(-1, 1)
    q = q_samp.reshape(-1, 1)
    r_k = KDTree(p).query(p, k=k+1)[0][:, k]
    s_k = KDTree(q).query(p, k=k)[0][:, k-1]
    mask = (r_k > 1e-10) & (s_k > 1e-10)
    kl   = np.log(s_k[mask]/r_k[mask]).mean() + np.log(m/(n-1))
    return float(max(0.0, kl))

def per_filter_kl(kdes, E_q):
    n_filt = len(kdes)
    kl_vec = np.zeros(n_filt)
    for m in range(n_filt):
        if kdes[m] is None:
            continue
        p_samp    = kdes[m].resample(N_KDE_SAMP, seed=RANDOM_SEED).flatten()
        kl_vec[m] = knn_kl(p_samp, E_q[:, m])
    return kl_vec

def bootstrap_dfd_ci(E_phi_A, E_lin_A, E_phi_B, E_lin_B,
                     n_boot=500, ci=95, rng=None):
    """
    Bootstrap 95% CI on DFD-bar for one (corpus, filterbank) pair.
    Resamples both A (GTZAN) and B (target corpus) with replacement.
    Returns (ci_lo, ci_hi, se).
    """
    rng = rng or np.random.default_rng(RANDOM_SEED)
    n_A, n_B = len(E_phi_A), len(E_phi_B)
    dfd_boots = []
    for _ in range(n_boot):
        idx_A = rng.integers(0, n_A, size=n_A)
        idx_B = rng.integers(0, n_B, size=n_B)
        kdes_A     = fit_kdes(E_phi_A[idx_A])
        kdes_A_lin = fit_kdes(E_lin_A[idx_A])
        kl_phi = per_filter_kl(kdes_A,     E_phi_B[idx_B])
        kl_lin = per_filter_kl(kdes_A_lin, E_lin_B[idx_B])
        dfd_boots.append((kl_phi - kl_lin).mean())
    boots = np.array(dfd_boots)
    lo = float(np.percentile(boots, (100 - ci) / 2))
    hi = float(np.percentile(boots, 100 - (100 - ci) / 2))
    return lo, hi, float(boots.std())
    
def fit_kdes(E):
    kdes = []
    for m in range(E.shape[1]):
        v = E[:, m]
        kdes.append(gaussian_kde(v, bw_method="silverman")
                    if v.std() > 1e-10 else None)
    return kdes

rng_null = np.random.default_rng(RANDOM_SEED + 999)
null_stats = {}   # fb_name -> (mu_null, sigma_null) shape (n_filt,)

for fb_name, (label, fb_matrix, fb_lin, n_filt, is_cqt) in FILTERBANKS.items():
    E_phi = energy_store[f"gtzan_{fb_name}"]
    E_lin = energy_store[f"gtzan_{fb_name}_lin"]
    n     = len(E_phi)
    half  = n // 2

    dfd_boots = []
    for b in range(N_BOOT):
        idx    = rng_null.permutation(n)
        i1, i2 = idx[:half], idx[half:2*half]
        kdes1  = fit_kdes(E_phi[i1])
        kdes1l = fit_kdes(E_lin[i1])
        kl_phi = per_filter_kl(kdes1,  E_phi[i2])
        kl_lin = per_filter_kl(kdes1l, E_lin[i2])
        dfd_boots.append(kl_phi - kl_lin)

    boots       = np.array(dfd_boots)
    mu_null     = boots.mean(axis=0)
    sigma_null  = boots.std(axis=0)
    null_stats[fb_name] = (mu_null, sigma_null)
    print(f"  {label}: null DFD-bar = {mu_null.mean():.4f} "
          f"(sigma={sigma_null.mean():.4f})")

print("[Step 5] Bootstrap null computed.")

# =============================================================================
# STEP 6 -- DFD-BAR AND Z-SCORES FOR ALL CORPORA x FILTERBANKS
# =============================================================================

print("\n[Step 6] Computing DFD-bar and z-scores...")

NON_WESTERN = ["carnatic","hindustani","turkish_makam","jingju","arab_andalusian"]
ALL_CORPORA = ["fma"] + NON_WESTERN   # FMA = noise floor; GTZAN = reference

dfd_results = {}   # corpus -> fb_name -> {dfd_bar, z_score, dfd_vec, peak_hz}

FB_CENTRE_FREQS = {
    "mel" : librosa.mel_frequencies(N_MEL, fmin=0.0, fmax=SR/2.0),
    "bark": None,
    "erb" : None,
    "cqt" : librosa.cqt_frequencies(N_CQT, fmin=CQT_FMIN,
                                     bins_per_octave=BINS_PER_OCT),
}

for corpus_key in ALL_CORPORA:
    dfd_results[corpus_key] = {}
    for fb_name, (label, fb_matrix, fb_lin, n_filt, is_cqt) \
            in FILTERBANKS.items():

        E_phi_A = energy_store[f"gtzan_{fb_name}"]
        E_lin_A = energy_store[f"gtzan_{fb_name}_lin"]
        E_phi_B = energy_store[f"{corpus_key}_{fb_name}"]
        E_lin_B = energy_store[f"{corpus_key}_{fb_name}_lin"]

        kdes_A     = fit_kdes(E_phi_A)
        kdes_A_lin = fit_kdes(E_lin_A)

        kl_phi  = per_filter_kl(kdes_A,     E_phi_B)
        kl_lin  = per_filter_kl(kdes_A_lin, E_lin_B)
        dfd_vec = kl_phi - kl_lin
        dfd_bar = float(dfd_vec.mean())

        mu_null, sigma_null = null_stats[fb_name]
        # z-score of DFD-bar against null distribution mean
        null_bar_mean = float(mu_null.mean())
        null_bar_std  = float(sigma_null.mean())
        z_score = (dfd_bar - null_bar_mean) / (null_bar_std + 1e-12)

        # Peak bias filter
        peak_m  = int(np.argmax(np.abs(dfd_vec)))
        cfreqs  = FB_CENTRE_FREQS.get(fb_name)
        peak_hz = float(cfreqs[peak_m]) if cfreqs is not None else float(peak_m)

        ci_lo, ci_hi, se = bootstrap_dfd_ci(
            E_phi_A, E_lin_A, E_phi_B, E_lin_B)

        dfd_results[corpus_key][fb_name] = {
            "dfd_bar" : dfd_bar,
            "z_score" : round(z_score, 3),
            "ci_95_lo": round(ci_lo, 4),
            "ci_95_hi": round(ci_hi, 4),
            "se"      : round(se, 4),
            "dfd_vec" : dfd_vec.tolist(),
            "peak_m"  : peak_m,
            "peak_hz" : round(peak_hz, 1),
        }
        print(f"  {corpus_key} / {label}: "
              f"DFD-bar={dfd_bar:.4f} "
              f"[{ci_lo:.4f}, {ci_hi:.4f}]  "
              f"z={z_score:.2f}  peak={peak_hz:.0f} Hz")

# =============================================================================
# STEP 7 -- SAVE RESULTS JSON
# =============================================================================

print("\n[Step 7] Saving results...")

results = {
    "experiment"  : "Experiment 2 -- Cross-Corpus DFD Characterisation",
    "parameters"  : {
        "sr": SR, "n_fft": N_FFT, "hop_length": HOP_LENGTH,
        "clip_dur_s": CLIP_DUR, "arab_chunk_s": ARAB_CHUNK,
        "n_clips_max": N_CLIPS_MAX, "n_boot": N_BOOT,
        "k_knn": K_KNN, "n_kde_samp": N_KDE_SAMP,
        "random_seed": RANDOM_SEED,
    },
    "corpus_meta" : ck["meta"],
    "null_stats"  : {
        fb: {"mu_bar": float(mu.mean()), "sigma_bar": float(sig.mean())}
        for fb, (mu, sig) in null_stats.items()
    },
    "dfd_results" : dfd_results,
}

res_path = os.path.join(LOCAL_DIR, "exp2_dfd_results.json")
with open(res_path, "w") as f:
    json.dump(results, f, indent=2)
copy_to_drive(res_path, "exp2_dfd_results.json")
print("[Step 7] Results saved.")

# =============================================================================
# SUMMARY TABLE
# =============================================================================

print("\n" + "="*70)
print("EXPERIMENT 2 COMPLETE — DFD-bar (z-score) summary")
print("="*70)
header = f"  {'Corpus':<20}" + "".join(
    f"{FILTERBANKS[fb][0]:>12}" for fb in FILTERBANKS)
print(header)
print("  " + "-"*68)
for corpus_key in ALL_CORPORA:
    row = f"  {corpus_key:<20}"
    for fb_name in FILTERBANKS:
        r = dfd_results[corpus_key][fb_name]
        row += f"  {r['dfd_bar']:>5.3f}({r['z_score']:>5.1f})"
    print(row)
print("\n  Format: DFD-bar (z-score). z>2.0 = significant amplification.")
print(f"\n  Google Drive outputs:")
print(f"    exp2_energy_matrices.npz")
print(f"    exp2_checkpoint.json")
print(f"    exp2_dfd_results.json")


# Load energy store
energy = np.load("/tmp/dfd_exp2/exp2_energy_matrices.npz")
rng = np.random.default_rng(42)

# Subsample GTZAN to 65 clips (matching Jingju size)
E_gtzan_mel = energy["gtzan_mel"]
idx = rng.choice(len(E_gtzan_mel), size=65, replace=False)

# Recompute Jingju/CQT DFD-bar with subsampled GTZAN
# (reuse fit_kdes and per_filter_kl from Program 02)
E_gtzan_cqt_sub = energy["gtzan_cqt"][idx]
E_gtzan_lin_sub = energy["gtzan_cqt_lin"][idx]
E_jingju_cqt    = energy["jingju_cqt"]
E_jingju_lin    = energy["jingju_cqt_lin"]

kdes_sub     = fit_kdes(E_gtzan_cqt_sub)
kdes_sub_lin = fit_kdes(E_gtzan_lin_sub)
kl_phi = per_filter_kl(kdes_sub,     E_jingju_cqt)
kl_lin = per_filter_kl(kdes_sub_lin, E_jingju_lin)
dfd_bar_sub = (kl_phi - kl_lin).mean()
print(f"Jingju/CQT DFD-bar (GTZAN subsampled to 65): {dfd_bar_sub:.4f}")
print(f"Jingju/CQT DFD-bar (full GTZAN 100):          0.6084")

print("="*70)

end_time = datetime.now()
print(f"End time:   {end_time}")
elapsed = end_time - start_time
elapsed_seconds = int(elapsed.total_seconds())
hours, remainder = divmod(elapsed_seconds, 3600)
minutes, seconds = divmod(remainder, 60)
print(f"Running time: {hours}:{minutes:02d}:{seconds:02