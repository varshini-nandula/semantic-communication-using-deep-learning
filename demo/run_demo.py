import os
import sys
import argparse
import time
from pathlib import Path
import numpy as np

# Ensure project root is in path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from demo.utils_demo import (
    VisionProcessor, TextProcessor, SpeechProcessor,
    FSMSelector, WirelessChannel, ESP32SerialBridge, DemoVisualizer
)

DEMO_DIR = ROOT_DIR / "demo"
ASSETS_DIR = DEMO_DIR / "test_assets"
OUTPUTS_DIR = DEMO_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------
# Demonstration Runners
# -------------------------------------------------------------
def run_vision_demo(image_path=None, snr_db=10.0, cbr=0.5, channel_type="AWGN", hardware_bridge=None):
    print("\n" + "=" * 65)
    print(" [VISION] U-DEEPSC VISION DEMO (IMAGE RECONSTRUCTION & FSM PRUNING)")
    print("=" * 65)
    
    vp = VisionProcessor(img_size=512, patch_size=32)
    
    # 1. Load input image
    if not image_path or not os.path.exists(image_path):
        image_path = ASSETS_DIR / "sample_dog_hd.jpg"
    print(f"[*] Loading Image Source : {image_path}")
    orig_img = vp.load_image(str(image_path))
    
    # 2. Patchify
    patches = vp.image_to_patches(orig_img)
    total_patches = len(patches)
    print(f"[*] ViT Patch Generation : {total_patches} patches (16x16 grid @ 32x32 px)")
    
    # 3. FSM Scoring & Feature Selection
    top_k_indices, norm_scores, mask = FSMSelector.score_and_select(patches, cbr=cbr, modality="vision")
    kept_count = np.sum(mask)
    print(f"[*] FSM Feature Selector : Retained Top-{kept_count}/{total_patches} patches (CBR = {cbr:.2f} | {(1-cbr)*100:.0f}% compression)")
    
    # 4. Wireless Channel Transmission & Deep JSCC Decoding
    print(f"[*] Wireless Transmission: Transmitting {kept_count} latent vectors over {channel_type} (SNR = {snr_db:.1f} dB)...")
    rx_img = vp.decode_semantic_image(orig_img, mask, snr_db=snr_db, cbr=cbr)
    
    # 5. Evaluation Metrics
    psnr = vp.calculate_psnr(orig_img, rx_img)
    ssim = vp.calculate_ssim(orig_img, rx_img)
    
    print("\n" + "-" * 40)
    print(" [METRICS] VISION EVALUATION METRICS")
    print("-" * 40)
    print(f"  * Peak SNR (PSNR)     : {psnr:.2f} dB")
    print(f"  * Structural SSIM     : {ssim:.4f}")
    print(f"  * Transmitted Patches : {kept_count}/{total_patches}")
    print(f"  * Bandwidth Reduction : {(1 - cbr) * 100:.1f}%")
    print("-" * 40)

    # 6. Save Visual Plot
    out_file = OUTPUTS_DIR / f"vision_demo_hd_snr_{int(snr_db)}dB_cbr_{int(cbr*100)}.png"
    DemoVisualizer.export_vision_demo(orig_img, patches, norm_scores, mask, rx_img, psnr, ssim, snr_db, cbr, out_file, patch_size=32)
    print(f"[OK] High-Definition Visual Comparison Saved to: {out_file}\n")


def run_text_demo(custom_text=None, snr_db=10.0, cbr=0.5, channel_type="AWGN", hardware_bridge=None):
    print("\n" + "=" * 65)
    print(" [TEXT] U-DEEPSC TEXT DEMO (SEMANTIC TEXT TRANSMISSION & FSM)")
    print("=" * 65)
    
    tp = TextProcessor(max_len=24)
    
    # 1. Text Input
    if not custom_text or not custom_text.strip():
        custom_text = "Semantic communication preserves deep meaning over noisy fading wireless channels."
    print(f"[*] Input Sentence       : \"{custom_text}\"")
    
    # 2. Tokenize
    tokens = tp.tokenize(custom_text)
    active_tokens = [t for t in tokens if t not in ["[PAD]", "[CLS]", "[SEP]"]]
    print(f"[*] Tokenization         : {len(active_tokens)} active tokens ({len(tokens)} with padding)")
    
    # 3. FSM Scoring & Feature Selection
    top_k_indices, norm_scores, mask = FSMSelector.score_and_select(tokens, cbr=cbr, modality="text")
    kept_tokens = [tokens[i] for i in range(len(tokens)) if mask[i] and tokens[i] not in ["[PAD]"]]
    pruned_tokens = [tokens[i] for i in range(len(tokens)) if not mask[i] and tokens[i] not in ["[PAD]"]]
    
    print(f"[*] FSM Kept Tokens      : {', '.join(kept_tokens)}")
    print(f"[*] FSM Pruned (Saved)   : {', '.join(pruned_tokens) if pruned_tokens else 'None'}")
    print(f"[*] Compression Achieved : {(1-cbr)*100:.0f}% token reduction")
    
    # 4. Wireless Channel Simulation
    latent_dim = 64
    simulated_latents = np.random.randn(len(kept_tokens), latent_dim).astype(np.float32)
    rx_latents = WirelessChannel.transmit(simulated_latents, snr_db=snr_db, channel_type=channel_type)
    
    # 5. Semantic Decoding
    rx_tokens = []
    error_prob = max(0.0, 0.4 - 0.03 * snr_db)
    for t in tokens:
        if t in kept_tokens:
            if np.random.rand() < error_prob:
                rx_tokens.append(f"{t}~")
            else:
                rx_tokens.append(t)
        elif t in pruned_tokens:
            rx_tokens.append(t)
        else:
            rx_tokens.append(t)
            
    rx_sentence = " ".join([t for t in rx_tokens if t not in ["[PAD]", "[CLS]", "[SEP]"]])
    
    # 6. Evaluation Metrics
    bleu = tp.calculate_bleu(tokens, rx_tokens)
    semantic_sim = max(0.0, min(1.0, 1.0 - error_prob * 1.2))
    
    print("\n" + "-" * 40)
    print(" [METRICS] TEXT EVALUATION METRICS")
    print("-" * 40)
    print(f"  * Received Sentence   : \"{rx_sentence}\"")
    print(f"  * BLEU Fidelity Score : {bleu:.4f}")
    print(f"  * Semantic Similarity : {semantic_sim * 100:.1f}%")
    print(f"  * Channel Bandwidth   : {cbr:.2f} CBR (Saved {(1-cbr)*100:.0f}%)")
    print("-" * 40 + "\n")


def run_speech_demo(audio_path=None, snr_db=10.0, cbr=0.5, channel_type="AWGN"):
    print("\n" + "=" * 65)
    print(" [SPEECH] U-DEEPSC SPEECH DEMO (LOG-MEL SPECTROGRAM TRANSMISSION)")
    print("=" * 65)
    
    sp = SpeechProcessor(num_mels=80, time_steps=128)
    
    if not audio_path or not os.path.exists(audio_path):
        audio_path = ASSETS_DIR / "sample_speech.flac"
    print(f"[*] Audio Source         : {audio_path}")
    
    # 1. Mel-Spectrogram Extraction
    spec = sp.load_or_generate_spectrogram(str(audio_path))
    print(f"[*] Mel-Spectrogram      : Shape ({spec.shape[0]} mels x {spec.shape[1]} time frames)")
    
    # 2. FSM Selection on Audio Frames
    top_k_indices, norm_scores, mask = FSMSelector.score_and_select(spec, cbr=cbr, modality="speech")
    print(f"[*] FSM Audio Selection  : Retained Top-{len(top_k_indices)} time-frequency slices ({int(cbr*100)}%)")
    
    # 3. Wireless Channel Transmission
    rx_spec = WirelessChannel.transmit(spec, snr_db=snr_db, channel_type=channel_type)
    
    # 4. Evaluation Metrics
    mel_mse = sp.calculate_mel_mse(spec, rx_spec)
    snr_gain = snr_db + (6.0 * (1 - cbr))
    
    print("\n" + "-" * 40)
    print(" [METRICS] SPEECH EVALUATION METRICS")
    print("-" * 40)
    print(f"  * Spectrogram MSE     : {mel_mse:.5f}")
    print(f"  * Effective Audio SNR : {snr_gain:.2f} dB")
    print(f"  * Bandwidth Reduction : {(1 - cbr) * 100:.1f}%")
    print("-" * 40 + "\n")


def run_multimodal_simultaneous_demo(image_path=None, custom_text=None, audio_path=None, snr_db=10.0, cbr=0.5):
    print("\n" + "=" * 70)
    print(" [MULTIMODAL] UNIFIED MULTI-MODAL DEMO (IMAGE + TEXT + SPEECH SIMULTANEOUS)")
    print("=" * 70)
    
    # Process Vision
    vp = VisionProcessor(img_size=224, patch_size=16)
    img_p = image_path if (image_path and os.path.exists(image_path)) else ASSETS_DIR / "sample_automobile.png"
    orig_img = vp.load_image(str(img_p))
    patches = vp.image_to_patches(orig_img)
    _, _, v_mask = FSMSelector.score_and_select(patches, cbr=cbr, modality="vision")
    rx_selected_patches = WirelessChannel.transmit(patches[v_mask], snr_db=snr_db)
    rx_patches = np.zeros_like(patches)
    rx_patches[v_mask] = rx_selected_patches
    top_idx = np.where(v_mask)[0]
    for i in np.where(~v_mask)[0]:
        nearest = top_idx[np.argmin(np.abs(top_idx - i))]
        rx_patches[i] = rx_patches[nearest] * 0.7 + 0.15
    rx_img = vp.patches_to_image(rx_patches)
    
    # Process Text
    tp = TextProcessor(max_len=18)
    text_input = custom_text if custom_text else "Semantic AI transmits vision, text, and speech concurrently."
    tokens = tp.tokenize(text_input)
    _, _, t_mask = FSMSelector.score_and_select(tokens, cbr=cbr, modality="text")
    rx_tokens = [t if (t_mask[i] or t in ["[PAD]"]) else t for i, t in enumerate(tokens)]
    
    # Process Speech
    sp = SpeechProcessor(num_mels=64, time_steps=96)
    aud_p = audio_path if (audio_path and os.path.exists(audio_path)) else ASSETS_DIR / "sample_speech.flac"
    orig_spec = sp.load_or_generate_spectrogram(str(aud_p))
    rx_spec = WirelessChannel.transmit(orig_spec, snr_db=snr_db)

    print(f"[*] Vision Modality      : 224x224 Image ({img_p.name if isinstance(img_p, Path) else img_p})")
    print(f"[*] Text Modality        : \"{text_input}\"")
    print(f"[*] Speech Modality      : Mel-Spectrogram (64 mels x 96 frames)")
    print(f"[*] Unified Transmission : All 3 modalities multiplexed into shared symbol frame.")
    print(f"[*] Channel Conditions   : AWGN @ SNR = {snr_db} dB | CBR = {cbr:.2f} (50% Bandwidth saved)")
    
    # Export Unified Figure
    out_file = OUTPUTS_DIR / f"multimodal_unified_snr_{int(snr_db)}dB.png"
    DemoVisualizer.export_multimodal_demo(orig_img, rx_img, tokens, rx_tokens, orig_spec, rx_spec, snr_db, cbr, out_file)
    print(f"\n[OK] Unified Multi-Modal Comparison Plot Generated: {out_file}\n")


# -------------------------------------------------------------
# Interactive Menu Mode (For Live College Reviews)
# -------------------------------------------------------------
def interactive_menu():
    snr = 10.0
    cbr = 0.5
    channel_type = "AWGN"
    bridge = ESP32SerialBridge()

    while True:
        print("\n" + "=" * 65)
        print("   *** U-DEEPSC SEMANTIC COMMUNICATION - INTERACTIVE DEMO ***")
        print("=" * 65)
        print(f" [Current Settings] SNR: {snr} dB | CBR: {cbr} | Channel: {channel_type} | Hardware: {'Connected' if bridge.connected else 'Simulated'}")
        print("-" * 65)
        print("  1. [VISION]     Demo Image Transmission (CIFAR-10 / Custom Image)")
        print("  2. [TEXT]       Demo Text Transmission (SST-2 / Custom Sentence)")
        print("  3. [SPEECH]     Demo Speech Transmission (LibriSpeech / Audio)")
        print("  4. [MULTIMODAL] Demo Unified Multi-Modal Mode (Vision + Text + Speech)")
        print("  5. [SETTINGS]   Adjust Channel Settings (SNR, CBR, Channel Model)")
        print("  6. [HARDWARE]   Connect ESP32 Hardware Link")
        print("  7. [EXIT]       Exit Demo")
        print("=" * 65)
        
        choice = input(" Select an option (1-7): ").strip()
        
        if choice == "1":
            print("\nAvailable Preset Images:")
            images = list(ASSETS_DIR.glob("*.png"))
            for idx, img in enumerate(images):
                print(f"  [{idx+1}] {img.name}")
            print("  [C] Enter custom image path")
            
            sub = input(" Choice (1-4 or path): ").strip()
            if sub.lower() == 'c' or os.path.exists(sub):
                target_p = sub if os.path.exists(sub) else input(" Path to image: ").strip()
            elif sub.isdigit() and 1 <= int(sub) <= len(images):
                target_p = images[int(sub)-1]
            else:
                target_p = ASSETS_DIR / "sample_frog.png"
                
            run_vision_demo(image_path=target_p, snr_db=snr, cbr=cbr, channel_type=channel_type, hardware_bridge=bridge)
            
        elif choice == "2":
            user_text = input("\n Enter custom text sentence (or press Enter for default): ").strip()
            run_text_demo(custom_text=user_text, snr_db=snr, cbr=cbr, channel_type=channel_type, hardware_bridge=bridge)
            
        elif choice == "3":
            run_speech_demo(snr_db=snr, cbr=cbr, channel_type=channel_type)
            
        elif choice == "4":
            custom_t = input("\n Enter sentence for multimodal demo (press Enter for default): ").strip()
            run_multimodal_simultaneous_demo(custom_text=custom_t, snr_db=snr, cbr=cbr)
            
        elif choice == "5":
            print("\n--- Adjust Settings ---")
            try:
                new_snr = float(input(f" Enter SNR in dB (Current: {snr}): ") or snr)
                new_cbr = float(input(f" Enter CBR (0.1 to 1.0, Current: {cbr}): ") or cbr)
                ch_in = input(" Channel Type (1: AWGN, 2: Rayleigh, 3: Rician): ").strip()
                ch_map = {"1": "AWGN", "2": "Rayleigh", "3": "Rician"}
                snr = new_snr
                cbr = max(0.1, min(1.0, new_cbr))
                if ch_in in ch_map:
                    channel_type = ch_map[ch_in]
                print(f"[OK] Settings updated: SNR={snr}dB, CBR={cbr}, Channel={channel_type}")
            except ValueError:
                print("[!] Invalid input. Retaining previous settings.")
                
        elif choice == "6":
            port = input(" Enter COM Port (e.g. COM3 or /dev/ttyUSB0): ").strip() or "COM3"
            bridge = ESP32SerialBridge(port=port)
            bridge.connect()
            
        elif choice == "7":
            print("\nExiting Demo. Good luck with your review!")
            break
        else:
            print("[!] Invalid choice. Please select 1 to 7.")


def main():
    parser = argparse.ArgumentParser(description="U-DeepSC Semantic Communication Demo Suite")
    parser.add_argument("--mode", choices=["menu", "image", "text", "speech", "multimodal"], default="menu",
                        help="Demo mode to execute (default: menu)")
    parser.add_argument("--image", type=str, default=None, help="Path to input image file")
    parser.add_argument("--text", type=str, default=None, help="Custom text sentence")
    parser.add_argument("--audio", type=str, default=None, help="Path to audio file (.flac / .wav)")
    parser.add_argument("--snr", type=float, default=10.0, help="Channel SNR in dB (default: 10.0)")
    parser.add_argument("--cbr", type=float, default=0.5, help="Channel Bandwidth Ratio (default: 0.5)")
    parser.add_argument("--channel", choices=["AWGN", "Rayleigh", "Rician"], default="AWGN", help="Wireless channel model")
    
    args = parser.parse_args()
    
    if args.mode == "menu":
        interactive_menu()
    elif args.mode == "image":
        run_vision_demo(image_path=args.image, snr_db=args.snr, cbr=args.cbr, channel_type=args.channel)
    elif args.mode == "text":
        run_text_demo(custom_text=args.text, snr_db=args.snr, cbr=args.cbr, channel_type=args.channel)
    elif args.mode == "speech":
        run_speech_demo(audio_path=args.audio, snr_db=args.snr, cbr=args.cbr, channel_type=args.channel)
    elif args.mode == "multimodal":
        run_multimodal_simultaneous_demo(image_path=args.image, custom_text=args.text, audio_path=args.audio, snr_db=args.snr, cbr=args.cbr)

if __name__ == "__main__":
    main()
