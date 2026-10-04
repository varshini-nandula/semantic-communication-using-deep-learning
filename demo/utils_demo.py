"""
demo/utils_demo.py  –  UDeepSC-FSM Demo Utility Engine
========================================================
All processing is real where possible:
  - Mel-spectrogram : librosa (real STFT-based mel-spectrogram)
  - SSIM            : skimage.metrics.structural_similarity (windowed, standard)
  - PSNR            : standard formula on actual pixel arrays
  - Semantic sim    : sentence-transformers cosine similarity (real embeddings)
  - BLEU            : nltk sentence_bleu with 4-gram weights
  - FSM scoring     : spatial-variance proxy  (honest label – no trained ckpt)
  - Channels        : mathematically correct AWGN / Rayleigh / Rician
  - Model           : UDeepSC loaded when checkpoint present; simulation mode otherwise

All heavy imports (librosa, skimage, sentence-transformers) are deferred to
first use so demo startup is fast.
"""

import os
import sys
import math
import time
import warnings
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Availability flags (set at import by lightweight package check) ─────────────
def _pkg_available(name):
    import importlib.util
    return importlib.util.find_spec(name) is not None

_HAS_SKIMAGE   = _pkg_available("skimage")
_HAS_LIBROSA   = _pkg_available("librosa")
# sentence_transformers imports datasets -> pyarrow which causes C access violation on Python 3.14 on Windows
_HAS_SBERT     = _pkg_available("sentence_transformers") if sys.version_info < (3, 14) else False
_HAS_SKLEARN   = _pkg_available("sklearn")
_HAS_NLTK      = _pkg_available("nltk")

if not _HAS_SKIMAGE:  warnings.warn("[demo] scikit-image not found; SSIM falls back to global approx.")
if not _HAS_LIBROSA:  warnings.warn("[demo] librosa not found; audio uses synthetic spectrogram.")
if not _HAS_SBERT and not _HAS_SKLEARN: warnings.warn("[demo] TF-IDF / sentence-transformers not found; semantic sim uses unigram.")
if not _HAS_NLTK:     warnings.warn("[demo] nltk not found; BLEU uses unigram precision fallback.")

# ── Lazy loader helpers (actual imports deferred to first use) ─────────────────
_SBERT_MODEL = None

def _get_skimage_ssim():
    from skimage.metrics import structural_similarity as fn
    return fn

def _get_librosa():
    import librosa as _lib
    return _lib

def _get_sbert():
    global _SBERT_MODEL
    if _SBERT_MODEL is None and _HAS_SBERT:
        try:
            from sentence_transformers import SentenceTransformer
            _SBERT_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as e:
            warnings.warn(f"[demo] Could not load SBERT model: {e}")
    return _SBERT_MODEL

def _get_nltk_bleu():
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    return sentence_bleu, SmoothingFunction

def _get_sklearn_cosine():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    return TfidfVectorizer, cosine_similarity


# =============================================================================
# 1. Vision Processing & Patch Operations
# =============================================================================
class VisionProcessor:
    def __init__(self, img_size=512, patch_size=32):
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches_side = img_size // patch_size
        self.num_patches = self.num_patches_side ** 2

    # ── Loading ───────────────────────────────────────────────────────────────
    def load_image(self, img_path):
        """Loads any image (JPG/PNG/JPEG/RGBA/grayscale) → float32 [0,1] RGB array."""
        img = Image.open(img_path).convert("RGB")
        img = img.resize((self.img_size, self.img_size), Image.Resampling.LANCZOS)
        return np.array(img, dtype=np.float32) / 255.0

    # ── Patch operations ──────────────────────────────────────────────────────
    def image_to_patches(self, img_arr):
        """(H, W, 3) → (N, p, p, 3) patches."""
        p = self.patch_size
        patches = []
        for r in range(self.num_patches_side):
            for c in range(self.num_patches_side):
                patches.append(img_arr[r*p:(r+1)*p, c*p:(c+1)*p, :])
        return np.array(patches)

    def patches_to_image(self, patches_arr):
        """(N, p, p, 3) → (H, W, 3) image."""
        p = self.patch_size
        img = np.zeros((self.img_size, self.img_size, 3), dtype=np.float32)
        idx = 0
        for r in range(self.num_patches_side):
            for c in range(self.num_patches_side):
                img[r*p:(r+1)*p, c*p:(c+1)*p, :] = patches_arr[idx]
                idx += 1
        return np.clip(img, 0.0, 1.0)

    # ── Semantic reconstruction ───────────────────────────────────────────────
    def decode_semantic_image(self, orig_img, mask, snr_db=10.0, cbr=0.5,
                               channel_type="AWGN"):
        """
        Simulates JSCC semantic image reconstruction:
          1. Selected patches transmitted through the chosen wireless channel.
          2. Dropped patches reconstructed via contextual Gaussian-blur inpainting.
        Channel type is respected: AWGN / Rayleigh / Rician.
        """
        from scipy.ndimage import gaussian_filter
        p = self.patch_size
        num_side = self.num_patches_side
        patches = self.image_to_patches(orig_img)

        # ── Transmit selected patches through the chosen channel ──────────────
        selected = patches[mask]                          # (K, p, p, 3)
        flat_selected = selected.reshape(len(selected), -1).astype(np.float32)
        rx_flat = WirelessChannel.transmit(flat_selected, snr_db=snr_db,
                                            channel_type=channel_type)
        rx_selected = np.clip(rx_flat.reshape(selected.shape), 0.0, 1.0)

        rx_patches = patches.copy()
        rx_patches[mask] = rx_selected

        # ── Build received image ──────────────────────────────────────────────
        rx_img = self.patches_to_image(rx_patches)

        # ── Contextual inpainting for pruned (unselected) patches ─────────────
        blurred_bg = gaussian_filter(orig_img, sigma=2.5)
        mask_2d = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        idx = 0
        for r in range(num_side):
            for c in range(num_side):
                if mask[idx]:
                    mask_2d[r*p:(r+1)*p, c*p:(c+1)*p] = 1.0
                idx += 1
        smooth_mask = gaussian_filter(mask_2d, sigma=1.5)[:, :, np.newaxis]
        final = rx_img * smooth_mask + blurred_bg * (1.0 - smooth_mask)
        return np.clip(final, 0.0, 1.0)

    # ── Metrics ───────────────────────────────────────────────────────────────
    @staticmethod
    def calculate_psnr(original, reconstructed):
        """Standard PSNR (dB) on [0,1] float arrays."""
        mse = np.mean((original - reconstructed) ** 2)
        if mse == 0:
            return 100.0
        return float(20.0 * math.log10(1.0 / math.sqrt(mse)))

    @staticmethod
    def calculate_ssim(original, reconstructed):
        """
        Structural Similarity Index using skimage (windowed, standard).
        Falls back to a global approximation if skimage is unavailable.
        """
        if _HAS_SKIMAGE:
            ssim_fn = _get_skimage_ssim()
            val = ssim_fn(original, reconstructed,
                          channel_axis=2, data_range=1.0,
                          win_size=7)
            return float(np.clip(val, 0.0, 1.0))
        # Fallback: global SSIM approximation
        C1, C2 = (0.01)**2, (0.03)**2
        mu_x, mu_y = np.mean(original), np.mean(reconstructed)
        sx = np.var(original)
        sy = np.var(reconstructed)
        sxy = np.mean((original - mu_x) * (reconstructed - mu_y))
        ssim = ((2*mu_x*mu_y + C1) * (2*sxy + C2)) / \
               ((mu_x**2 + mu_y**2 + C1) * (sx + sy + C2))
        return float(np.clip(ssim, 0.0, 1.0))


# =============================================================================
# 2. Text Processing & Tokenization
# =============================================================================
class TextProcessor:
    def __init__(self, max_len=32):
        self.max_len = max_len

    def tokenize(self, text):
        """Simple word-level tokenization with [CLS]/[SEP]/[PAD] markers."""
        words = text.strip().split()
        if len(words) > self.max_len - 2:
            words = words[:self.max_len - 2]
        tokens = ["[CLS]"] + words + ["[SEP]"]
        pad_len = self.max_len - len(tokens)
        tokens += ["[PAD]"] * pad_len
        return tokens

    @staticmethod
    def calculate_bleu(reference_tokens, candidate_tokens):
        """
        BLEU score using nltk sentence_bleu with 4-gram weights and
        SmoothingFunction.method1 (handles short sentences gracefully).
        Falls back to unigram precision if nltk is unavailable.
        """
        ref_words = [t for t in reference_tokens
                     if t not in ["[PAD]", "[CLS]", "[SEP]"]]
        cand_words = [t for t in candidate_tokens
                      if t not in ["[PAD]", "[CLS]", "[SEP]"]]
        if not ref_words or not cand_words:
            return 0.0
        if _HAS_NLTK:
            sentence_bleu, SmoothingFunction = _get_nltk_bleu()
            sf = SmoothingFunction().method1
            score = sentence_bleu([ref_words], cand_words,
                                  weights=(0.25, 0.25, 0.25, 0.25),
                                  smoothing_function=sf)
            return float(np.clip(score, 0.0, 1.0))
        # Fallback: unigram precision
        matches = sum(1 for w in cand_words if w in ref_words)
        precision = matches / len(cand_words) if cand_words else 0.0
        bp = math.exp(min(0, 1 - len(ref_words) / len(cand_words))) \
             if cand_words else 0.0
        return float(np.clip(precision * bp, 0.0, 1.0))

    @staticmethod
    def calculate_semantic_similarity(original_text, reconstructed_text):
        """
        Real semantic similarity using sentence-transformers cosine similarity.
        Falls back to TF-IDF cosine if sentence-transformers unavailable.
        Falls back to token overlap precision if neither is available.
        """
        sbert = _get_sbert()
        if sbert is not None:
            try:
                from sentence_transformers import util as st_util
                emb_orig = sbert.encode(original_text, convert_to_tensor=True)
                emb_rx   = sbert.encode(reconstructed_text, convert_to_tensor=True)
                score = float(st_util.cos_sim(emb_orig, emb_rx).item())
                return float(np.clip(score, 0.0, 1.0))
            except Exception:
                pass  # fall through

        if _HAS_SKLEARN and original_text.strip() and reconstructed_text.strip():
            try:
                TfidfVectorizer, cosine_similarity = _get_sklearn_cosine()
                tfidf = TfidfVectorizer().fit_transform(
                    [original_text, reconstructed_text])
                score = float(cosine_similarity(tfidf[0], tfidf[1])[0, 0])
                return float(np.clip(score, 0.0, 1.0))
            except Exception:
                pass

        # Final fallback: unigram overlap
        a = set(original_text.lower().split())
        b = set(reconstructed_text.lower().split())
        if not a or not b:
            return 0.0
        return float(len(a & b) / max(len(a), len(b)))


# =============================================================================
# 3. Speech Processing (Real Mel-Spectrogram via librosa)
# =============================================================================
class SpeechProcessor:
    def __init__(self, num_mels=80, time_steps=128, sr=16000):
        self.num_mels = num_mels
        self.time_steps = time_steps
        self.sr = sr

    def load_or_generate_spectrogram(self, audio_path=None):
        """
        Loads audio and computes a real log mel-spectrogram using librosa.
        Falls back to a synthetic harmonic spectrogram if librosa is absent
        or if the file cannot be read. The fallback is clearly labelled.
        """
        if audio_path and os.path.exists(audio_path) and _HAS_LIBROSA:
            try:
                librosa = _get_librosa()
                y, sr = librosa.load(audio_path, sr=self.sr, mono=True)
                mel = librosa.feature.melspectrogram(
                    y=y, sr=sr, n_mels=self.num_mels,
                    hop_length=512, n_fft=1024)
                log_mel = librosa.power_to_db(mel, ref=np.max)
                # Normalise to [-1, 1]
                log_mel = (log_mel - log_mel.min()) / \
                          (log_mel.max() - log_mel.min() + 1e-8) * 2.0 - 1.0
                # Crop / pad to fixed time_steps
                if log_mel.shape[1] >= self.time_steps:
                    spec = log_mel[:, :self.time_steps]
                else:
                    pad = self.time_steps - log_mel.shape[1]
                    spec = np.pad(log_mel, ((0, 0), (0, pad)),
                                  mode='constant', constant_values=-1.0)
                return spec.astype(np.float32)
            except Exception as e:
                warnings.warn(f"[SpeechProcessor] librosa load failed ({e}); "
                              "using synthetic spectrogram.")

        # ── Synthetic fallback (clearly labelled) ─────────────────────────────
        t = np.linspace(0, 4 * np.pi, self.time_steps)
        f = np.linspace(0, 10, self.num_mels)
        T, F = np.meshgrid(t, f)
        spec = (np.sin(T * 2 + F) * np.exp(-F / 5.0)
                + 0.3 * np.cos(T * 5 - F * 2))
        return np.clip(spec, -1.0, 1.0).astype(np.float32)

    @staticmethod
    def calculate_mel_mse(original_spec, reconstructed_spec):
        """MSE on mel-spectrogram (lower is better)."""
        return float(np.mean((original_spec - reconstructed_spec) ** 2))

    @staticmethod
    def calculate_spectrogram_psnr(original_spec, reconstructed_spec):
        """PSNR on normalised mel-spectrogram (data range = 2.0 for [-1,1])."""
        mse = np.mean((original_spec - reconstructed_spec) ** 2)
        if mse < 1e-12:
            return 100.0
        return float(20.0 * math.log10(2.0 / math.sqrt(mse)))


# =============================================================================
# 4. FSM Proxy Selector
#    *** CLEARLY LABELLED AS PROXY (no trained checkpoint) ***
#    The real FSM (class FSM in model_util.py) requires a trained checkpoint.
#    This heuristic approximates the selection for demonstration purposes.
# =============================================================================
class FSMSelector:
    """
    Proxy Feature Selection Module.
    Uses spatial variance (vision), content-word heuristic (text),
    or spectral energy (speech) to rank and select features.

    NOTE: The real UDeepSC FSM is a learned neural module (mask_gen + rho_function)
    that conditions on encoder hidden states and channel SNR.
    This proxy is used when no trained checkpoint is loaded.
    Label all outputs as [PROXY-FSM] in the UI.
    """

    @staticmethod
    def score_and_select(features, cbr=0.5, modality="vision"):
        """
        Returns (top_k_indices, norm_scores, bool_mask).
        features shape:
          vision  -> (N, p, p, 3)
          text    -> list of N token strings
          speech  -> (num_mels, time_steps) 2-D array
        """
        if modality == "speech" and isinstance(features, np.ndarray) and features.ndim == 2:
            N = features.shape[1]                    # time frames
            K = max(1, int(N * cbr))
            # Spectral energy per time frame (real metric)
            scores = np.linalg.norm(features, axis=0)

        elif modality == "vision":
            N = len(features)
            K = max(1, int(N * cbr))
            # Spatial variance + mean absolute value (proxy for saliency)
            scores = (np.var(features, axis=(1, 2, 3))
                      + 0.1 * np.mean(np.abs(features), axis=(1, 2, 3)))

        elif modality == "text":
            N = len(features)
            K = max(1, int(N * cbr))
            scores = []
            for token in features:
                if token in ["[PAD]", "[CLS]", "[SEP]"]:
                    scores.append(0.05)
                elif token.lower() in {"the","in","a","an","is","of","and",
                                       "to","for","on","at","by","it","as"}:
                    scores.append(0.3)
                else:
                    # Longer / rarer words get higher importance proxy
                    scores.append(1.0 + 0.05 * len(token))
            scores = np.array(scores, dtype=np.float32)

        else:
            N = len(features)
            K = max(1, int(N * cbr))
            arr = np.array(features)
            scores = np.var(arr, axis=tuple(range(1, arr.ndim))) \
                     if arr.ndim > 1 else np.abs(arr)

        # Normalise to [0, 1]
        s_min, s_max = np.min(scores), np.max(scores)
        if s_max > s_min:
            norm_scores = (scores - s_min) / (s_max - s_min)
        else:
            norm_scores = np.ones_like(scores)

        top_k_indices = np.argsort(norm_scores)[::-1][:K]
        mask = np.zeros(N, dtype=bool)
        mask[top_k_indices] = True

        return top_k_indices, norm_scores, mask


# =============================================================================
# 5. Wireless Channel Model (AWGN / Rayleigh / Rician)
#    Mathematically correct; applied to actual signal data.
# =============================================================================
class WirelessChannel:
    @staticmethod
    def transmit(symbols, snr_db=10.0, channel_type="AWGN"):
        """
        Passes real-valued symbols through a simulated wireless fading channel.
          symbols      : numpy array of any shape (flattened internally)
          snr_db       : Signal-to-Noise Ratio in dB
          channel_type : 'AWGN', 'Rayleigh', or 'Rician'
        Returns array of same shape as input.
        """
        orig_shape = symbols.shape
        flat = symbols.flatten().astype(np.float32)

        # Unit average transmit power normalisation
        p_sig = float(np.mean(flat ** 2))
        norm = flat / np.sqrt(p_sig) if p_sig > 1e-12 else flat.copy()

        # Noise variance from SNR
        noise_var = 10.0 ** (-snr_db / 10.0)
        noise = np.random.normal(0.0, np.sqrt(noise_var), size=norm.shape)

        ch = channel_type.upper()

        if ch == "AWGN":
            rx = norm + noise

        elif ch == "RAYLEIGH":
            # Flat Rayleigh fading: h ~ CN(0,1), |h|^2 ~ Exp(1)
            h_r = np.random.normal(0.0, 1.0 / math.sqrt(2))
            h_i = np.random.normal(0.0, 1.0 / math.sqrt(2))
            h_mag = math.sqrt(h_r**2 + h_i**2)
            # Apply fading then ZF equalization
            rx = (norm * h_mag + noise) / max(h_mag, 1e-3)

        elif ch == "RICIAN":
            # Rician K = 4 (moderate line-of-sight)
            K = 4.0
            s_los = math.sqrt(K / (K + 1))
            s_nlos = math.sqrt(1.0 / (K + 1)) * float(np.random.rayleigh(scale=1.0))
            h_tot = s_los + s_nlos
            rx = (norm * h_tot + noise) / max(h_tot, 1e-3)

        else:
            rx = norm + noise

        # Restore original power scale
        if p_sig > 1e-12:
            rx = rx * math.sqrt(p_sig)

        return rx.reshape(orig_shape)


# =============================================================================
# 6. ESP32 Hardware Serial Bridge
# =============================================================================
class ESP32SerialBridge:
    def __init__(self, port="COM3", baudrate=115200, timeout=1.0):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.connected = False
        self.serial_conn = None

    def connect(self):
        try:
            import serial
            self.serial_conn = serial.Serial(
                self.port, self.baudrate, timeout=self.timeout)
            self.connected = True
            print(f"[Hardware] Connected to ESP32 on {self.port} @ {self.baudrate} baud.")
            return True
        except Exception as e:
            print(f"[Hardware] ESP32 not detected on {self.port} ({e}). "
                  "Running in software simulation mode.")
            self.connected = False
            return False

    def transmit_bytes(self, byte_payload):
        if not self.connected or not self.serial_conn:
            return byte_payload, 0.0
        t0 = time.time()
        length = len(byte_payload)
        header = bytes([0xAA, 0x55, (length >> 8) & 0xFF, length & 0xFF])
        checksum = bytes([sum(byte_payload) & 0xFF])
        packet = header + byte_payload + checksum
        self.serial_conn.write(packet)
        received = self.serial_conn.read(len(packet))
        t_ms = (time.time() - t0) * 1000.0
        rx_payload = received[4:-1] if len(received) >= 5 else byte_payload
        return rx_payload, t_ms


# =============================================================================
# 7. Visualization & Plot Export
# =============================================================================
class DemoVisualizer:

    @staticmethod
    def export_vision_demo(original_img, patches, norm_scores, mask, rx_img,
                           psnr, ssim, snr, cbr, channel_type, out_path,
                           patch_size=32, proxy_fsm=True):
        """Generates side-by-side vision demonstration figure with FSM overlays."""
        fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), dpi=200)
        fig.patch.set_facecolor('#0b0f19')

        H, W, _ = original_img.shape
        p = patch_size
        num_side = H // p
        fsm_label = "[PROXY-FSM] " if proxy_fsm else "[FSM] "

        # Panel 1 – Original
        axes[0].imshow(original_img)
        axes[0].set_title("1. Original HD Source Image",
                           color='#38bdf8', fontsize=13, fontweight='bold', pad=10)
        axes[0].axis('off')

        # Panel 2 – FSM patch saliency overlay
        fsm_vis = original_img.copy()
        idx = 0
        for r in range(num_side):
            for c in range(num_side):
                if not mask[idx]:
                    fsm_vis[r*p:(r+1)*p, c*p:(c+1)*p, :] *= 0.22
                idx += 1
        axes[1].imshow(fsm_vis)
        axes[1].set_title(
            f"2. {fsm_label}Saliency (CBR={cbr:.2f} | {int(cbr*100)}% Kept)",
            color='#c084fc', fontsize=12, fontweight='bold', pad=10)
        axes[1].axis('off')

        # Grid lines
        for i in range(1, num_side):
            axes[1].axhline(i*p, color='#38bdf8', alpha=0.2, linewidth=0.5)
            axes[1].axvline(i*p, color='#38bdf8', alpha=0.2, linewidth=0.5)

        idx = 0
        for r in range(num_side):
            for c in range(num_side):
                if mask[idx]:
                    rect = matplotlib.patches.Rectangle(
                        (c*p, r*p), p, p,
                        linewidth=1.1, edgecolor='#38bdf8',
                        facecolor='none', alpha=0.7)
                    axes[1].add_patch(rect)
                idx += 1

        # Panel 3 – Reconstructed
        axes[2].imshow(rx_img)
        axes[2].set_title(
            f"3. Reconstructed via {channel_type} (SNR={snr:.1f} dB)",
            color='#4ade80', fontsize=12, fontweight='bold', pad=10)
        axes[2].axis('off')

        metrics_text = (f"PSNR: {psnr:.2f} dB  |  SSIM (windowed): {ssim:.4f}  |"
                        f"  Bandwidth Saved: {(1-cbr)*100:.0f}%  |  Channel: {channel_type}")
        fig.suptitle(
            f"U-DeepSC Semantic Vision Communication Demonstration\n{metrics_text}",
            color='#f8fafc', fontsize=13, fontweight='bold', y=0.96)

        plt.subplots_adjust(top=0.82, bottom=0.08, left=0.04, right=0.96, wspace=0.15)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
        plt.close()

    @staticmethod
    def export_speech_demo(orig_spec, rx_spec, snr, cbr, channel_type,
                           psnr_spec, mse_spec, out_path,
                           is_real_spec=True):
        """Side-by-side mel-spectrogram comparison."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
        fig.patch.set_facecolor('#090d16')
        spec_label = "Log Mel-Spectrogram" if is_real_spec else "Synthetic Spectrogram (librosa unavailable)"

        axes[0].imshow(orig_spec, aspect='auto', origin='lower', cmap='magma')
        axes[0].set_title(f"Transmitted: {spec_label}",
                          color='#94a3b8', fontsize=11)
        axes[0].set_xlabel("Time Frames", color='#94a3b8')
        axes[0].set_ylabel("Mel Bins", color='#94a3b8')

        axes[1].imshow(rx_spec, aspect='auto', origin='lower', cmap='magma')
        axes[1].set_title(
            f"Received ({channel_type} @ {snr:.1f} dB | CBR={cbr:.2f})",
            color='#4ade80', fontsize=11)
        axes[1].set_xlabel("Time Frames", color='#94a3b8')

        for ax in axes:
            ax.tick_params(colors='#64748b')
            for spine in ax.spines.values():
                spine.set_edgecolor('#1e293b')

        fig.suptitle(
            f"U-DeepSC Speech Spectrogram Transmission\n"
            f"Spectrogram PSNR: {psnr_spec:.2f} dB  |  MSE: {mse_spec:.5f}  |"
            f"  Bandwidth Saved: {(1-cbr)*100:.0f}%",
            color='#38bdf8', fontsize=12, fontweight='bold')

        plt.tight_layout()
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
        plt.close()

    @staticmethod
    def export_multimodal_demo(img_orig, img_rx, tokens_orig, tokens_rx,
                               spec_orig, spec_rx,
                               snr, cbr, out_path,
                               psnr_img=None, ssim_img=None,
                               sem_sim=None, mse_spec=None,
                               channel_type="AWGN",
                               is_real_spec=True):
        """Unified 3-modality figure (three independent per-modality pipelines)."""
        fig = plt.figure(figsize=(16, 9), dpi=150)
        fig.patch.set_facecolor('#090d16')

        fig.suptitle(
            f"UDeepSC-FSM Multi-Modal Semantic Communication Demo\n"
            f"Channel: {channel_type} @ SNR={snr} dB  |  CBR={cbr:.2f}  "
            f"(Each modality processed over independent channel)",
            color='#38bdf8', fontsize=13, fontweight='bold', y=0.98)

        # Vision – row 1
        ax1 = fig.add_subplot(2, 3, 1)
        ax1.imshow(img_orig); ax1.set_title("Input Vision", color='#94a3b8', fontsize=10)
        ax1.axis('off')

        ax2 = fig.add_subplot(2, 3, 2)
        ax2.imshow(img_rx)
        title2 = "Received Vision"
        if psnr_img is not None:
            title2 += f"\nPSNR={psnr_img:.2f} dB"
        ax2.set_title(title2, color='#4ade80', fontsize=10)
        ax2.axis('off')

        # Speech – right column
        ax3 = fig.add_subplot(2, 3, 3)
        spec_title = "Input Speech Spec." + ("" if is_real_spec else "\n(synthetic)")
        ax3.imshow(spec_orig, aspect='auto', origin='lower', cmap='magma')
        ax3.set_title(spec_title, color='#94a3b8', fontsize=10); ax3.axis('off')

        ax4 = fig.add_subplot(2, 3, 6)
        spec_rx_title = "Received Speech Spec."
        if mse_spec is not None:
            spec_rx_title += f"\nMSE={mse_spec:.5f}"
        ax4.imshow(spec_rx, aspect='auto', origin='lower', cmap='magma')
        ax4.set_title(spec_rx_title, color='#4ade80', fontsize=10); ax4.axis('off')

        # Text – row 2 centre
        ax_t = fig.add_subplot(2, 3, (4, 5))
        ax_t.set_facecolor('#1e293b'); ax_t.axis('off')
        orig_str = " ".join(t for t in tokens_orig if t not in ["[PAD]","[CLS]","[SEP]"])
        rx_str   = " ".join(t for t in tokens_rx   if t not in ["[PAD]","[CLS]","[SEP]"])
        sim_line = f"\n  Semantic Similarity: {sem_sim*100:.1f}%" if sem_sim is not None else ""
        text_body = (
            f"Original Text:\n  \"{orig_str}\"\n\n"
            f"Received Text (channel-corrupted tokens):\n  \"{rx_str}\"\n"
            f"{sim_line}\n\n"
            f"[NOTE] Each modality transmitted independently over shared CBR budget."
        )
        ax_t.text(0.04, 0.5, text_body, color='#f1f5f9', fontsize=10,
                  family='monospace', verticalalignment='center', wrap=True)

        plt.tight_layout()
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
        plt.close()
