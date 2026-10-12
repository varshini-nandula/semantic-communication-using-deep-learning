"""
demo/run_udeepsc_demo.py  –  Original U-DeepSC Baseline Pipeline Demo
======================================================================
Base Paper: Zhang et al., "A Unified Multi-Task Semantic Communication
System for Multimodal Data", IEEE Trans. Commun., July 2024.

This demo demonstrates the original U-DeepSC pipeline:
  1. Image Classification ('imgc' on CIFAR-10) via ViT + Black-box FSM
  2. Text Classification ('textc' on SST-2) via BERT + Black-box FSM
  3. Speech Sentiment Regression ('msa' on 74-dim CMU-MOSEI features)
  4. Unified Multi-Task Mode across all three classification tasks
  5. Wireless transmission over AWGN / Rayleigh / Rician channels

Key Architectural Note:
  The base paper ONLY used speech for sentiment classification ('msa'),
  never speech reconstruction. Its FSM is a black-box MLP + Gumbel-Softmax.
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
    VisionProcessor, TextProcessor, BaselineClassifier,
    WirelessChannel, ESP32SerialBridge, DemoVisualizer,
    load_udeepsc_model
)

DEMO_DIR = ROOT_DIR / "demo"
ASSETS_DIR = DEMO_DIR / "test_assets"
OUTPUTS_DIR = DEMO_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Checkpoint Detection for Smoketest Models ─────────────────────────────
def _try_load_baseline_model():
    """Attempts to load a trained/smoketest UDeepSC model."""
    return load_udeepsc_model()


# =============================================================================
# 1. Image Classification Demo ('imgc')
# =============================================================================
def run_baseline_image_demo(image_path=None, snr_db=10.0, cbr=0.5, channel_type="AWGN"):
    print("\n" + "=" * 70)
    print(" [BASELINE VISION] Original U-DeepSC Image Classification ('imgc')")
    print("=" * 70)

    model, loaded = _try_load_baseline_model()
    print(f"[*] Backbone State       : {'[SMOKETEST CHECKPOINT LOADED]' if loaded else '[SIMULATION MODE]'}")

    if not image_path or not os.path.exists(str(image_path)):
        image_path = ASSETS_DIR / "sample_automobile.png"
    print(f"[*] Input Image Source   : {image_path}")

    # Feature selection ratio
    print(f"[*] Black-Box FSM CBR    : {cbr:.2f} (dropping {(1-cbr)*100:.0f}% feature tokens via MLP)")
    print(f"[*] Channel Model        : {channel_type} @ SNR={snr_db:.1f} dB")

    pred_class, conf, probs = BaselineClassifier.infer_image_class(
        str(image_path), snr_db=snr_db, channel_type=channel_type, model=model
    )

    print("\n" + "-" * 50)
    print(" [RESULTS] U-DEEPSC VISION CLASSIFICATION")
    print("-" * 50)
    print(f"  * Predicted Class      : {pred_class.upper()}")
    print(f"  * Confidence Score     : {conf*100:.2f}%")
    print(f"  * Target Dataset       : CIFAR-10 (10 classes)")
    print(f"  * Base Paper Mechanism : ViT + MLP FSM + VectorQuantizer -> Linear Head")
    print("-" * 50)

    out_file = OUTPUTS_DIR / f"baseline_imgc_{channel_type}_snr{int(snr_db)}dB.png"
    # Export demo visual
    import matplotlib.pyplot as plt
    from PIL import Image
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=160)
    fig.patch.set_facecolor('#090d16')
    ax1.imshow(Image.open(image_path).convert("RGB"))
    ax1.set_title(f"Input Image: {Path(image_path).name}", color='#38bdf8', fontsize=11)
    ax1.axis('off')

    ax2.set_facecolor('#111827')
    y_pos = np.arange(len(BaselineClassifier.CIFAR10_CLASSES))
    ax2.barh(y_pos, probs, color='#38bdf8', edgecolor='#1e293b')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(BaselineClassifier.CIFAR10_CLASSES, color='#f1f5f9', fontsize=9)
    ax2.set_xlabel("Probability", color='#94a3b8')
    ax2.set_title(f"Predicted: {pred_class.upper()} ({conf*100:.1f}%)\n{channel_type} @ {snr_db:.1f}dB | CBR={cbr:.2f}",
                  color='#4ade80', fontsize=11, fontweight='bold')
    ax2.tick_params(colors='#94a3b8')
    for spine in ax2.spines.values():
        spine.set_edgecolor('#1e293b')

    plt.tight_layout()
    plt.savefig(out_file, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"[OK] Classification plot saved : {out_file}\n")


# =============================================================================
# 2. Text Sentiment Classification Demo ('textc')
# =============================================================================
def run_baseline_text_demo(custom_text=None, snr_db=10.0, cbr=0.5, channel_type="AWGN"):
    print("\n" + "=" * 70)
    print(" [BASELINE TEXT] Original U-DeepSC Sentiment Classification ('textc')")
    print("=" * 70)

    model, loaded = _try_load_baseline_model()
    print(f"[*] Backbone State       : {'[SMOKETEST CHECKPOINT LOADED]' if loaded else '[SIMULATION MODE]'}")

    if not custom_text or not custom_text.strip():
        custom_text = "The deep learning model preserves meaning accurately over harsh wireless fading channels."
    print(f"[*] Input Sentence       : \"{custom_text}\"")
    print(f"[*] Black-Box FSM CBR    : {cbr:.2f} (pruned by learned MLP Gumbel-Softmax)")
    print(f"[*] Channel Model        : {channel_type} @ SNR={snr_db:.1f} dB")

    pred, conf, probs = BaselineClassifier.infer_text_sentiment(
        custom_text, snr_db=snr_db, channel_type=channel_type, model=model
    )

    print("\n" + "-" * 50)
    print(" [RESULTS] U-DEEPSC TEXT CLASSIFICATION")
    print("-" * 50)
    print(f"  * Sentiment Prediction : {pred.upper()}")
    print(f"  * Confidence Score     : {conf*100:.2f}%")
    print(f"  * Target Dataset       : SST-2 (Binary Sentiment)")
    print(f"  * Base Paper Mechanism : BERT + MLP FSM + VectorQuantizer -> Linear Head")
    print("-" * 50 + "\n")


# =============================================================================
# 3. Speech Sentiment Regression Demo ('msa') [Original Paper's Speech Task]
# =============================================================================
def run_baseline_speech_demo(snr_db=10.0, channel_type="AWGN"):
    print("\n" + "=" * 70)
    print(" [BASELINE SPEECH] Original U-DeepSC Acoustic Sentiment Task ('msa')")
    print("=" * 70)
    print(" [*] Base Paper Context   : The original paper ONLY supported CMU-MOSEI")
    print("                            74-dim pre-extracted acoustic features for 'msa'")
    print("                            (Multimodal Sentiment Analysis), NOT reconstruction!")
    print(f"[*] Feature Dimension    : 74 acoustic dimensions -> Linear(74, 128)")
    print(f"[*] Channel Transmission : {channel_type} @ SNR={snr_db:.1f} dB")

    model, loaded = _try_load_baseline_model()
    score = BaselineClassifier.infer_acoustic_sentiment(snr_db=snr_db, channel_type=channel_type, model=model)

    print("\n" + "-" * 50)
    print(" [RESULTS] U-DEEPSC ACOUSTIC SENTIMENT REGRESSION")
    print("-" * 50)
    print(f"  * Sentiment Score      : {score:+.2f}  (range: -3.0 to +3.0)")
    print(f"  * Interpretation       : {'Positive Sentiment' if score > 0 else 'Negative Sentiment'}")
    print(f"  * Base Paper Limit     : NO speech reconstruction existed in U-DeepSC!")
    print("-" * 50 + "\n")


# =============================================================================
# 4. Multi-Task Unified Classification Mode
# =============================================================================
def run_baseline_multitask_demo(image_path=None, custom_text=None, snr_db=10.0, cbr=0.5, channel_type="AWGN"):
    print("\n" + "=" * 70)
    print(" [BASELINE MULTI-TASK] Original U-DeepSC Unified Pipeline")
    print(" Image ('imgc') + Text ('textc') + Speech Sentiment ('msa')")
    print("=" * 70)

    model, loaded = _try_load_baseline_model()
    print(f"[*] Backbone State       : {'[CHECKPOINT LOADED]' if loaded else '[INITIALIZED MODEL]'}")

    if not image_path or not os.path.exists(str(image_path)):
        image_path = ASSETS_DIR / "sample_automobile.png"
    if not custom_text:
        custom_text = "Semantic communication preserves deep meaning over noisy wireless channels."

    pred_class, img_conf, probs = BaselineClassifier.infer_image_class(str(image_path), snr_db, channel_type, model=model)
    text_pred, text_conf, _ = BaselineClassifier.infer_text_sentiment(custom_text, snr_db, channel_type, model=model)
    speech_score = BaselineClassifier.infer_acoustic_sentiment(snr_db=snr_db, channel_type=channel_type, model=model)

    print(f"\n[*] Vision ('imgc')      : Image '{Path(image_path).name}' -> {pred_class.upper()} ({img_conf*100:.1f}%)")
    print(f"[*] Text ('textc')        : \"{custom_text}\" -> {text_pred.upper()} ({text_conf*100:.1f}%)")
    print(f"[*] Speech ('msa')        : 74-dim CMU-MOSEI features -> Score: {speech_score:+.2f}")
    print(f"[*] Wireless Channel     : {channel_type} @ SNR={snr_db:.1f} dB | CBR={cbr:.2f}")

    out_file = OUTPUTS_DIR / f"baseline_multitask_{channel_type}_snr{int(snr_db)}dB.png"
    DemoVisualizer.export_baseline_classification_demo(
        str(image_path), pred_class, img_conf, probs,
        custom_text, text_pred, text_conf,
        speech_score, snr_db, cbr, channel_type, out_file
    )
    print(f"\n[OK] Baseline multi-task visualization saved : {out_file}\n")


# =============================================================================
# Interactive Menu for Baseline Pipeline
# =============================================================================
def baseline_interactive_menu():
    snr = 10.0
    cbr = 0.5
    channel_type = "AWGN"

    while True:
        print("\n" + "=" * 70)
        print("  *** ORIGINAL U-DEEPSC BASELINE PIPELINE DEMONSTRATION ***")
        print("      (IEEE Trans. Commun. 2024 - Multi-Task Classification)")
        print("=" * 70)
        print(f"  Channel Settings: SNR = {snr:.1f} dB | CBR = {cbr:.2f} | Model = {channel_type}")
        print("-" * 70)
        print("  1. Image Classification ('imgc' on CIFAR-10 via ViT + FSM)")
        print("  2. Text Classification  ('textc' on SST-2 via BERT + FSM)")
        print("  3. Speech Sentiment     ('msa' on CMU-MOSEI 74-dim features)")
        print("  4. Joint Multi-Task     (Vision + Text + Speech simultaneously)")
        print("  5. Adjust Channel / SNR / CBR Settings")
        print("  6. Return to Master Launcher")
        print("=" * 70)

        choice = input(" Select option (1-6): ").strip()
        if choice == "1":
            run_baseline_image_demo(snr_db=snr, cbr=cbr, channel_type=channel_type)
        elif choice == "2":
            run_baseline_text_demo(snr_db=snr, cbr=cbr, channel_type=channel_type)
        elif choice == "3":
            run_baseline_speech_demo(snr_db=snr, channel_type=channel_type)
        elif choice == "4":
            run_baseline_multitask_demo(snr_db=snr, cbr=cbr, channel_type=channel_type)
        elif choice == "5":
            try:
                snr_in = input(f" Enter SNR in dB [-10 to +25] (current={snr}): ").strip()
                if snr_in: snr = float(snr_in)
                cbr_in = input(f" Enter CBR [0.1 to 1.0] (current={cbr}): ").strip()
                if cbr_in: cbr = float(cbr_in)
                ch_in = input(f" Channel (AWGN / Rayleigh / Rician) (current={channel_type}): ").strip().upper()
                if ch_in in ["AWGN", "RAYLEIGH", "RICIAN"]: channel_type = ch_in
            except ValueError:
                print(" Invalid input, keeping current settings.")
        elif choice == "6":
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Original U-DeepSC Baseline Demo Runner")
    parser.add_argument("--mode", type=str, choices=["image", "text", "speech", "multitask", "menu"], default="menu")
    parser.add_argument("--snr", type=float, default=10.0)
    parser.add_argument("--cbr", type=float, default=0.5)
    parser.add_argument("--channel", type=str, default="AWGN", choices=["AWGN", "Rayleigh", "Rician"])
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--text", type=str, default=None)
    args = parser.parse_args()

    if args.mode == "image":
        run_baseline_image_demo(image_path=args.image, snr_db=args.snr, cbr=args.cbr, channel_type=args.channel)
    elif args.mode == "text":
        run_baseline_text_demo(custom_text=args.text, snr_db=args.snr, cbr=args.cbr, channel_type=args.channel)
    elif args.mode == "speech":
        run_baseline_speech_demo(snr_db=args.snr, channel_type=args.channel)
    elif args.mode == "multitask":
        run_baseline_multitask_demo(image_path=args.image, custom_text=args.text, snr_db=args.snr, cbr=args.cbr, channel_type=args.channel)
    else:
        baseline_interactive_menu()
