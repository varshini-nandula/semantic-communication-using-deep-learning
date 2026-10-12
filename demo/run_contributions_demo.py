"""
demo/run_contributions_demo.py  –  Our Extended Contributions Pipeline Demo
=============================================================================
Team: Batch No. 12 (Explainable Multimodal Semantic Communication)

This demo demonstrates our EXTENDED PIPELINE based on the work done so far:
  1. Full Multimodal Reconstruction as the Primary Task:
       - Image Reconstruction ('imgr') with PSNR and windowed SSIM
       - Text Reconstruction ('textr') with BLEU-4 and Semantic Cosine Similarity
  2. Our Novel Speech Reconstruction Architecture ('sper'):
       - Built from scratch in `SPTEncoder_FSM` (model_util.py)
       - 80-bin Log Mel-Spectrogram extraction from LibriSpeech FLAC via librosa
       - Transformer-based patch embedding (Linear 80 -> embed_dim)
       - Dynamic Speech FSM (FSM_Dict['sper']) token pruning
       - Dedicated speech reconstruction head (head['sper'] -> Linear(embed_dim, 80))
       - Evaluation using Mel-MSE and Mel-PSNR
  3. Unified Multimodal Simultaneous Reconstruction:
       - Concurrently transmits Vision + Text + Speech under shared CBR budget
  4. Wireless Channel Simulation & ESP32 Hardware Link:
       - AWGN, Rayleigh, and Rician fading models
       - USB Serial bridge to ESP32 DevKit running ESP-NOW 2.4 GHz RF
"""

import os
import sys
import argparse
import time
from pathlib import Path
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from demo.utils_demo import (
    VisionProcessor, TextProcessor, SpeechProcessor,
    FSMSelector, WirelessChannel, ESP32SerialBridge, DemoVisualizer,
    NeuralReconstructionEngine, load_udeepsc_model,
    _HAS_LIBROSA, _HAS_SBERT, _HAS_SKIMAGE
)

DEMO_DIR = ROOT_DIR / "demo"
ASSETS_DIR = DEMO_DIR / "test_assets"
OUTPUTS_DIR = DEMO_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

def _try_load_model():
    """Attempts to load a trained/smoketest UDeepSC model."""
    return load_udeepsc_model()


# =============================================================================
# 1. Vision Reconstruction Demo ('imgr')
# =============================================================================
def run_vision_recon_demo(image_path=None, snr_db=10.0, cbr=0.5, channel_type="AWGN", hardware_bridge=None):
    print("\n" + "=" * 70)
    print(" [OUR PIPELINE - VISION] Image Reconstruction & FSM Patch Selection")
    print("=" * 70)

    model, loaded = _try_load_model()
    print(f"[*] Neural Model State   : {'[TRAINED/SMOKETEST CHECKPOINT LOADED]' if loaded else '[INITIALIZED BACKBONE]'}")

    vp = VisionProcessor(img_size=512, patch_size=32)

    if not image_path or not os.path.exists(str(image_path)):
        image_path = ASSETS_DIR / "sample_dog_hd.jpg"
    print(f"[*] Image Source         : {image_path}")
    orig_img = vp.load_image(str(image_path))

    patches = vp.image_to_patches(orig_img)
    total_patches = len(patches)
    print(f"[*] Patch Grid           : {total_patches} patches ({vp.num_patches_side}x{vp.num_patches_side} @ {vp.patch_size}x{vp.patch_size}px)")

    # FSM selection
    _, norm_scores, mask = FSMSelector.score_and_select(patches, cbr=cbr, modality="vision")
    kept_count = int(np.sum(mask))
    print(f"[*] FSM Selection        : Retained {kept_count}/{total_patches} patches (CBR={cbr:.2f} | {(1-cbr)*100:.0f}% bandwidth saved)")

    # Optional Hardware Transmission
    if hardware_bridge and hardware_bridge.connected:
        dummy_bytes = bytes([int(v * 255) for v in norm_scores[mask][:64]])
        rx_bytes, t_ms = hardware_bridge.transmit_bytes(dummy_bytes)
        print(f"[*] ESP-NOW RF Transfer  : Transmitted via ESP32 in {t_ms:.1f} ms")

    # Wireless channel transmission & Neural Reconstruction
    print(f"[*] Channel Transmission : {kept_count} patch latents over {channel_type} @ SNR={snr_db:.1f} dB")
    recon_pipeline = "Neural U-DeepSC ViT + Decoder ('imgr')"
    if model is not None:
        try:
            rx_img, psnr, ssim = NeuralReconstructionEngine.reconstruct_image(
                orig_img, snr_db=snr_db, channel_type=channel_type, cbr=cbr, model=model
            )
        except Exception as e:
            warnings.warn(f"[VisionDemo] Neural reconstruction error: {e}. Falling back to JSCC simulation.")
            rx_img = vp.decode_semantic_image(orig_img, mask, snr_db=snr_db, cbr=cbr, channel_type=channel_type)
            psnr = VisionProcessor.calculate_psnr(orig_img, rx_img)
            ssim = VisionProcessor.calculate_ssim(orig_img, rx_img)
            recon_pipeline = "Simulated JSCC Inpainting"
    else:
        rx_img = vp.decode_semantic_image(orig_img, mask, snr_db=snr_db, cbr=cbr, channel_type=channel_type)
        psnr = VisionProcessor.calculate_psnr(orig_img, rx_img)
        ssim = VisionProcessor.calculate_ssim(orig_img, rx_img)
        recon_pipeline = "Simulated JSCC Inpainting"

    ssim_method = "skimage windowed" if _HAS_SKIMAGE else "global approx"

    print("\n" + "-" * 50)
    print(" [METRICS] VISION RECONSTRUCTION EVALUATION")
    print("-" * 50)
    print(f"  * Pipeline Execution   : {recon_pipeline}")
    print(f"  * PSNR                 : {psnr:.2f} dB")
    print(f"  * SSIM ({ssim_method}): {ssim:.4f}")
    print(f"  * Patches Transmitted  : {kept_count} / {total_patches}")
    print(f"  * Bandwidth Reduction  : {(1-cbr)*100:.1f}%")
    print(f"  * Channel Model        : {channel_type}")
    print("-" * 50)

    out_file = OUTPUTS_DIR / f"contributions_vision_{channel_type}_snr{int(snr_db)}dB_cbr{int(cbr*100)}.png"
    DemoVisualizer.export_vision_demo(
        orig_img, patches, norm_scores, mask, rx_img,
        psnr, ssim, snr_db, cbr, channel_type, out_file,
        patch_size=vp.patch_size, proxy_fsm=True
    )
    print(f"[OK] Visual saved : {out_file}\n")


# =============================================================================
# 2. Text Reconstruction Demo ('textr')
# =============================================================================
def run_text_recon_demo(custom_text=None, snr_db=10.0, cbr=0.5, channel_type="AWGN", hardware_bridge=None):
    print("\n" + "=" * 70)
    print(" [OUR PIPELINE - TEXT] Text Semantic Reconstruction & Token FSM")
    print("=" * 70)

    model, loaded = _try_load_model()
    print(f"[*] Neural Model State   : {'[TRAINED/SMOKETEST CHECKPOINT LOADED]' if loaded else '[INITIALIZED BACKBONE]'}")

    tp = TextProcessor(max_len=32)

    if not custom_text or not custom_text.strip():
        custom_text = "Semantic communication preserves deep meaning over noisy fading wireless channels."
    print(f"[*] Input Sentence       : \"{custom_text}\"")

    tokens = tp.tokenize(custom_text)
    active_tokens = [t for t in tokens if t not in ["[PAD]","[CLS]","[SEP]"]]
    print(f"[*] Tokenization         : {len(active_tokens)} content tokens (total seq len={len(tokens)})")

    # FSM scoring & selection
    _, norm_scores, mask = FSMSelector.score_and_select(tokens, cbr=cbr, modality="text")
    kept_tokens = [tokens[i] for i in range(len(tokens)) if mask[i] and tokens[i] not in ["[PAD]"]]
    pruned_tokens = [tokens[i] for i in range(len(tokens)) if not mask[i] and tokens[i] not in ["[PAD]","[CLS]","[SEP]"]]
    print(f"[*] FSM Kept Tokens      : {', '.join(kept_tokens)}")
    print(f"[*] FSM Pruned Tokens    : {', '.join(pruned_tokens) if pruned_tokens else 'None'}")

    recon_pipeline = "Neural BERT + Decoder ('textr')"
    if model is not None:
        try:
            rx_sentence, bleu, sem_sim = NeuralReconstructionEngine.reconstruct_text(
                custom_text, snr_db=snr_db, channel_type=channel_type, cbr=cbr, model=model
            )
        except Exception as e:
            warnings.warn(f"[TextDemo] Neural reconstruction error: {e}. Falling back to simulation.")
            rx_tokens = [t if np.random.rand() > 0.1 else t + "~" for t in tokens]
            rx_sentence = " ".join([t.rstrip("~") for t in rx_tokens if t not in ["[PAD]","[CLS]","[SEP]"]])
            bleu = TextProcessor.calculate_bleu(tokens, rx_tokens)
            sem_sim = TextProcessor.calculate_semantic_similarity(custom_text, rx_sentence)
            recon_pipeline = "Simulated Token Error Fallback"
    else:
        rx_tokens = [t if np.random.rand() > 0.1 else t + "~" for t in tokens]
        rx_sentence = " ".join([t.rstrip("~") for t in rx_tokens if t not in ["[PAD]","[CLS]","[SEP]"]])
        bleu = TextProcessor.calculate_bleu(tokens, rx_tokens)
        sem_sim = TextProcessor.calculate_semantic_similarity(custom_text, rx_sentence)
        recon_pipeline = "Simulated Token Error Fallback"

    sim_method = "sentence-transformers cosine" if _HAS_SBERT else "TF-IDF cosine fallback"

    print("\n" + "-" * 50)
    print(" [METRICS] TEXT RECONSTRUCTION EVALUATION")
    print("-" * 50)
    print(f"  * Pipeline Execution   : {recon_pipeline}")
    safe_sentence = "".join(ch for ch in rx_sentence if 32 <= ord(ch) < 127)
    print(f"  * Received Sentence    : \"{safe_sentence}\"")
    print(f"  * BLEU-4 Score         : {bleu:.4f}  [nltk BLEU-4]")
    print(f"  * Semantic Similarity  : {sem_sim*100:.1f}%  [{sim_method}]")
    print(f"  * Bandwidth Saved      : {(1-cbr)*100:.0f}%  (CBR={cbr:.2f})")
    print(f"  * Channel Model        : {channel_type}")
    print("-" * 50 + "\n")


# =============================================================================
# 3. Novel Speech Mel-Spectrogram Reconstruction Demo ('sper')
# =============================================================================
def run_speech_recon_demo(audio_path=None, snr_db=10.0, cbr=0.5, channel_type="AWGN"):
    print("\n" + "=" * 70)
    print(" [OUR PIPELINE - SPEECH] Novel Speech Mel-Spectrogram Reconstruction ('sper')")
    print(" Architecture: SPTEncoder_FSM (80 mel-bins) -> FSM Dict -> Head['sper']")
    print("=" * 70)

    model, loaded = _try_load_model()
    print(f"[*] Neural Model State   : {'[TRAINED/SMOKETEST CHECKPOINT LOADED]' if loaded else '[INITIALIZED BACKBONE]'}")

    sp = SpeechProcessor(num_mels=80, time_steps=128, sr=16000)

    if not audio_path or not os.path.exists(str(audio_path)):
        audio_path = ASSETS_DIR / "sample_speech.flac"
    print(f"[*] Audio Source         : {audio_path}")
    spec_source = "real librosa mel-spectrogram (STFT)" if _HAS_LIBROSA else "synthetic fallback"
    print(f"[*] Extraction Tool      : {spec_source}")

    # 1. 80-bin Mel-spectrogram
    orig_spec = sp.load_or_generate_spectrogram(str(audio_path))
    print(f"[*] Mel-Spectrogram      : shape {orig_spec.shape} (80 mel bins x {orig_spec.shape[1]} time frames)")
    print(f"[*] Novel Adapter        : nn.Linear(80, embed_dim) in SPTEncoder_FSM")

    # 2. FSM time-frame selection
    _, norm_scores, mask = FSMSelector.score_and_select(orig_spec, cbr=cbr, modality="speech")
    kept_frames = int(np.sum(mask))
    print(f"[*] Speech FSM Selection : Retained {kept_frames}/{orig_spec.shape[1]} time frames (CBR={cbr:.2f})")

    # 3. Channel transmission & Neural Speech Reconstruction
    recon_pipeline = "Neural SPTEncoder_FSM + Decoder ('sper')"
    if model is not None:
        try:
            rx_spec, mse_spec, psnr_spec = NeuralReconstructionEngine.reconstruct_speech(
                orig_spec, snr_db=snr_db, channel_type=channel_type, cbr=cbr, model=model
            )
        except Exception as e:
            warnings.warn(f"[SpeechDemo] Neural reconstruction error: {e}. Falling back to simulation.")
            selected_spec = orig_spec[:, mask]
            rx_selected = WirelessChannel.transmit(selected_spec, snr_db=snr_db, channel_type=channel_type)
            rx_spec = orig_spec.copy()
            rx_spec[:, mask] = rx_selected
            kept_idxs = np.where(mask)[0]
            for fi in np.where(~mask)[0]:
                nearest = kept_idxs[np.argmin(np.abs(kept_idxs - fi))]
                rx_spec[:, fi] = rx_spec[:, nearest] * 0.85
            mse_spec = SpeechProcessor.calculate_mel_mse(orig_spec, rx_spec)
            psnr_spec = SpeechProcessor.calculate_spectrogram_psnr(orig_spec, rx_spec)
            recon_pipeline = "Simulated Inpainting Fallback"
    else:
        selected_spec = orig_spec[:, mask]
        rx_selected = WirelessChannel.transmit(selected_spec, snr_db=snr_db, channel_type=channel_type)
        rx_spec = orig_spec.copy()
        rx_spec[:, mask] = rx_selected
        kept_idxs = np.where(mask)[0]
        for fi in np.where(~mask)[0]:
            nearest = kept_idxs[np.argmin(np.abs(kept_idxs - fi))]
            rx_spec[:, fi] = rx_spec[:, nearest] * 0.85
        mse_spec = SpeechProcessor.calculate_mel_mse(orig_spec, rx_spec)
        psnr_spec = SpeechProcessor.calculate_spectrogram_psnr(orig_spec, rx_spec)
        recon_pipeline = "Simulated Inpainting Fallback"

    print("\n" + "-" * 50)
    print(" [METRICS] NOVEL SPEECH RECONSTRUCTION EVALUATION")
    print("-" * 50)
    print(f"  * Pipeline Execution   : {recon_pipeline}")
    print(f"  * Spectrogram Mel-MSE  : {mse_spec:.5f}")
    print(f"  * Spectrogram PSNR     : {psnr_spec:.2f} dB")
    print(f"  * Frames Transmitted   : {kept_frames}/{orig_spec.shape[1]}")
    print(f"  * Bandwidth Saved      : {(1-cbr)*100:.1f}%  (CBR={cbr:.2f})")
    print(f"  * Channel Model        : {channel_type}")
    print("  * Novelty Highlight    : Base U-DeepSC paper had ZERO speech reconstruction!")
    print("-" * 50)

    out_file = OUTPUTS_DIR / f"contributions_speech_{channel_type}_snr{int(snr_db)}dB_cbr{int(cbr*100)}.png"
    DemoVisualizer.export_speech_demo(
        orig_spec, rx_spec, snr_db, cbr, channel_type,
        psnr_spec, mse_spec, out_file, is_real_spec=_HAS_LIBROSA
    )
    print(f"[OK] Spectrogram plot saved : {out_file}\n")


# =============================================================================
# 4. Unified Multimodal Simultaneous Reconstruction Demo
# =============================================================================
def run_multimodal_recon_demo(image_path=None, custom_text=None, audio_path=None,
                              snr_db=10.0, cbr=0.5, channel_type="AWGN"):
    print("\n" + "=" * 75)
    print(" [OUR PIPELINE - MULTIMODAL] Simultaneous 3-Modality Reconstruction")
    print(" Image ('imgr') + Text ('textr') + Speech ('sper') over Shared Bandwidth")
    print("=" * 75)

    from demo.run_demo import run_multimodal_simultaneous_demo
    run_multimodal_simultaneous_demo(
        image_path=image_path, custom_text=custom_text, audio_path=audio_path,
        snr_db=snr_db, cbr=cbr, channel_type=channel_type
    )


# =============================================================================
# Interactive Menu for Our Extended Contributions Pipeline
# =============================================================================
def contributions_interactive_menu():
    _try_load_model()
    snr = 10.0
    cbr = 0.5
    channel_type = "AWGN"
    bridge = ESP32SerialBridge()

    while True:
        print("\n" + "=" * 70)
        print("  *** OUR EXTENDED CONTRIBUTIONS PIPELINE DEMONSTRATION ***")
        print("      (Multimodal Semantic Reconstruction & Novel Speech Pipeline)")
        print("=" * 70)
        print(f"  Channel Settings: SNR = {snr:.1f} dB | CBR = {cbr:.2f} | Model = {channel_type} | "
              f"HW = {'Connected' if bridge.connected else 'Simulated'}")
        print("-" * 70)
        print("  1. Vision Reconstruction  ('imgr' with ViT + FSM + PSNR/SSIM)")
        print("  2. Text Reconstruction    ('textr' with BERT + FSM + BLEU-4/Similarity)")
        print("  3. Speech Reconstruction  ('sper' with novel SPTEncoder_FSM 80-bin mel)")
        print("  4. Joint Multimodal Recon (Vision + Text + Speech simultaneously)")
        print("  5. Connect ESP32 Hardware (Serial USB RF link via ESP-NOW)")
        print("  6. Adjust Channel / SNR / CBR Settings")
        print("  7. Return to Master Launcher")
        print("=" * 70)

        choice = input(" Select option (1-7): ").strip()
        if choice == "1":
            run_vision_recon_demo(snr_db=snr, cbr=cbr, channel_type=channel_type, hardware_bridge=bridge)
        elif choice == "2":
            run_text_recon_demo(snr_db=snr, cbr=cbr, channel_type=channel_type, hardware_bridge=bridge)
        elif choice == "3":
            run_speech_recon_demo(snr_db=snr, cbr=cbr, channel_type=channel_type)
        elif choice == "4":
            run_multimodal_recon_demo(snr_db=snr, cbr=cbr, channel_type=channel_type)
        elif choice == "5":
            port = input(" Enter ESP32 COM port (default COM3): ").strip() or "COM3"
            bridge.port = port
            bridge.connect()
        elif choice == "6":
            try:
                snr_in = input(f" Enter SNR in dB [-10 to +25] (current={snr}): ").strip()
                if snr_in: snr = float(snr_in)
                cbr_in = input(f" Enter CBR [0.1 to 1.0] (current={cbr}): ").strip()
                if cbr_in: cbr = float(cbr_in)
                ch_in = input(f" Channel (AWGN / Rayleigh / Rician) (current={channel_type}): ").strip().upper()
                if ch_in in ["AWGN", "RAYLEIGH", "RICIAN"]: channel_type = ch_in
            except ValueError:
                print(" Invalid input, keeping current settings.")
        elif choice == "7":
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Our Extended Contributions Pipeline Runner")
    parser.add_argument("--mode", type=str, choices=["vision", "text", "speech", "multimodal", "menu"], default="menu")
    parser.add_argument("--snr", type=float, default=10.0)
    parser.add_argument("--cbr", type=float, default=0.5)
    parser.add_argument("--channel", type=str, default="AWGN", choices=["AWGN", "Rayleigh", "Rician"])
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--audio", type=str, default=None)
    args = parser.parse_args()

    if args.mode == "vision":
        run_vision_recon_demo(image_path=args.image, snr_db=args.snr, cbr=args.cbr, channel_type=args.channel)
    elif args.mode == "text":
        run_text_recon_demo(custom_text=args.text, snr_db=args.snr, cbr=args.cbr, channel_type=args.channel)
    elif args.mode == "speech":
        run_speech_recon_demo(audio_path=args.audio, snr_db=args.snr, cbr=args.cbr, channel_type=args.channel)
    elif args.mode == "multimodal":
        run_multimodal_recon_demo(image_path=args.image, custom_text=args.text, audio_path=args.audio,
                                  snr_db=args.snr, cbr=args.cbr, channel_type=args.channel)
    else:
        contributions_interactive_menu()
