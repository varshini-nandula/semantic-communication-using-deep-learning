# 🌟 Semantic Communication Demonstration Suite

This directory contains the demonstration suite for both the **Existing U-DeepSC Baseline Pipeline** (base paper reproduction) and **Our Extended Contributions Pipeline** (work built so far). It is designed specifically for **college project reviews, external viva evaluations, and hardware integration**.

---

## 📁 Directory Layout

```
demo/
├── run_demo.py               # Master Portal (launches both pipelines or compares them)
├── run_udeepsc_demo.py       # [PIPELINE 1] Existing U-DeepSC Baseline (Classification & Black-box FSM)
├── run_contributions_demo.py # [PIPELINE 2] Our Extended Pipeline (Multimodal Recon & Novel Speech sper)
├── utils_demo.py             # Shared utilities: processors, metrics, channel, and ESP32 bridge
├── test_assets/              # Sample images (HD & CIFAR), audio (LibriSpeech FLAC), and text prompts
│   ├── sample_frog.png
│   ├── sample_truck.png
│   ├── sample_automobile.png
│   ├── sample_dog_hd.jpg
│   ├── sample_speech.flac
│   └── sample_texts.txt
├── outputs/                  # High-resolution generated plots, heatmaps, and metric charts
└── README.md                 # This guide
```

---

## 📊 Pipeline Comparison: Baseline vs. Our Contributions

| Dimension | [Pipeline 1] Existing U-DeepSC (Base Paper) | [Pipeline 2] Our Extended Contributions (Built So Far) |
|---|---|---|
| **Primary Focus** | Multi-Task Classification & Sentiment | **Full Multimodal Reconstruction** (`imgr`, `textr`, `sper`) |
| **Image Modality** | ViT → 10-Class Classification (`imgc`) | ViT → Full Pixel Reconstruction (PSNR, windowed SSIM) |
| **Text Modality** | BERT → Binary Sentiment (`textc`) | BERT → Full Token Reconstruction (BLEU-4, Cosine Sim) |
| **Speech Modality** | CMU-MOSEI 74-dim Sentiment (`msa`) | **LibriSpeech 80-bin Mel-Spectrogram** (`sper`) |
| **Speech Architecture**| `SPTEncoder` (Linear 74→128, **NO Reconstruction**) | **`SPTEncoder_FSM` + `head['sper']` (Engineered by us)** |
| **Speech FSM** | **NONE** (Zero FSM for speech in base paper) | **Dedicated `FSM_Dict['sper']`** pruning time frames |
| **Feature Selection** | Black-Box MLP + Gumbel-Softmax | Black-Box FSM (Prepares base for XAI swap) |
| **Corruption Recovery**| None (Corrupted tokens permanently lost) | Channel-tested, ready for Selective Retransmission |
| **Hardware** | Simulated only | ESP32 DevKit 2.4 GHz RF Link via ESP-NOW |

---

## 🚀 How to Run

### 1. Master Portal (Interactive Menu)
Launch the unified master terminal dashboard:
```bash
python demo/run_demo.py
```
This opens the master portal where you can pick:
- `[1]` Existing U-DeepSC Baseline Pipeline
- `[2]` Our Extended Contributions Pipeline
- `[3]` View Architectural Comparison Summary

---

### 2. [Pipeline 1] Existing U-DeepSC Baseline Demo
Dedicated runner for the base paper's multi-task classification setup:
```bash
# Interactive menu:
python demo/run_udeepsc_demo.py

# Direct CLI execution:
python demo/run_udeepsc_demo.py --mode image --snr 10 --cbr 0.5
python demo/run_udeepsc_demo.py --mode text --snr 10 --cbr 0.5
python demo/run_udeepsc_demo.py --mode speech --snr 10
python demo/run_udeepsc_demo.py --mode multitask --snr 12 --cbr 0.5
```

---

### 3. [Pipeline 2] Our Extended Contributions Pipeline Demo
Dedicated runner for our unified multimodal reconstruction architecture:
```bash
# Interactive menu:
python demo/run_contributions_demo.py

# Direct CLI execution:
# 🖼️ Image Reconstruction (PSNR & SSIM)
python demo/run_contributions_demo.py --mode vision --snr 10 --cbr 0.5

# 📝 Text Reconstruction (BLEU-4 & Semantic Similarity)
python demo/run_contributions_demo.py --mode text --snr 8 --cbr 0.5

# 🎙️ Novel Speech Mel-Spectrogram Reconstruction (SPTEncoder_FSM + Mel-MSE/PSNR)
python demo/run_contributions_demo.py --mode speech --snr 10 --cbr 0.5

# 🌐 Unified 3-Modality Simultaneous Reconstruction
python demo/run_contributions_demo.py --mode multimodal --snr 12 --cbr 0.5
```

---

## 🔌 ESP32 Hardware Integration Setup

To demonstrate physical wireless transmission:
1. Connect the transmitter ESP32 to your laptop via USB.
2. In `run_contributions_demo.py`, select option `[5] Connect ESP32 Hardware` and enter the COM port (e.g., `COM3` on Windows).
3. The demo transmits the quantized semantic bitstream to ESP32 #1, broadcasts it over **ESP-NOW 2.4 GHz RF**, receives it on ESP32 #2, and feeds it into the decoder!

---

## 🎓 College Review & Viva Cheat-Sheet

| What Examiners Ask | How to Answer with This Demo Suite |
|---|---|
| *"What is the difference between the base paper and your work so far?"* | Run `python demo/run_demo.py --pipeline compare`. Explain that the base paper only did classification and had **zero speech reconstruction**. We built the 80-bin mel-spectrogram reconstruction pipeline with `SPTEncoder_FSM` and unified reconstruction across image, text, and speech. |
| *"Show me the original paper's classification baseline."* | Run `python demo/run_udeepsc_demo.py --mode multitask`. Shows image classification on CIFAR-10, text sentiment on SST-2, and acoustic sentiment on CMU-MOSEI features. |
| *"Show me your novel speech reconstruction."* | Run `python demo/run_contributions_demo.py --mode speech`. Shows real LibriSpeech audio transformed into an 80-bin log mel-spectrogram, time-frame FSM pruning, wireless fading, and spectrogram reconstruction with Mel-MSE/PSNR. |
| *"How does channel noise affect semantic transmission?"* | Adjust SNR from $+20\text{ dB}$ down to $-5\text{ dB}$ in either demo. Show how semantic communication avoids catastrophic cliff-effects and preserves core meaning even under low SNR. |
