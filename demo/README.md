# 🌟 U-DeepSC Semantic Communication Demo Suite

This directory contains the complete demonstration suite for **U-DeepSC with Feature Selection Module (FSM)**. It is designed specifically for **college project reviews, external viva evaluations, and hardware integration**.

---

## 📁 Directory Layout

```
demo/
├── run_demo.py               # Master CLI + Interactive Menu Demo runner
├── utils_demo.py             # Visualizer, FSM selector, Wireless Channel & ESP32 hook
├── test_assets/              # Preloaded sample images (frog, truck, automobile), text, audio
│   ├── sample_frog.png
│   ├── sample_truck.png
│   ├── sample_automobile.png
│   ├── sample_speech.flac
│   └── sample_texts.txt
├── outputs/                  # High-resolution comparison plots and FSM heatmaps
└── README.md                 # This presentation and execution cheat-sheet
```

---

## 🚀 How to Run the Demo

### 1. Interactive Menu Mode (Recommended for Live Reviews)
Launch the interactive terminal dashboard:
```bash
python demo/run_demo.py
```
This opens an interactive menu where you can:
- Select preset images or provide a custom file path.
- Type any custom sentence on the spot if requested by your guide.
- Test speech audio transmission.
- **Run Unified Multi-Modal Mode** (Image + Text + Speech simultaneously).
- Dynamically adjust **SNR (-10 dB to +25 dB)**, **CBR (0.1 to 1.0)**, and **Channel Model (AWGN, Rayleigh, Rician)**.
- Connect an **ESP32 hardware link** via serial USB COM port.

---

### 2. Direct Command-Line Execution

#### 🖼️ Image Transmission (Vision Modality)
```bash
python demo/run_demo.py --mode image --snr 12 --cbr 0.5
# Or test with any custom image from your laptop:
python demo/run_demo.py --mode image --image path/to/my_photo.jpg --snr 10 --cbr 0.5
```

#### 📝 Text Transmission (Text Modality)
```bash
# Test with custom text prompt typed by the examiner:
python demo/run_demo.py --mode text --text "Semantic communication preserves deep meaning over noisy wireless channels." --snr 8 --cbr 0.5
```

#### 🎙️ Speech Transmission (Audio Modality)
```bash
python demo/run_demo.py --mode speech --snr 10 --cbr 0.5
```

#### 🌐 Unified Multi-Modal Simultaneous Transmission
```bash
python demo/run_demo.py --mode multimodal --snr 12 --cbr 0.5
```

---

## 🔌 ESP32 Hardware Integration Setup

When you are ready to demonstrate the physical wireless transmission:
1. Connect the transmitter ESP32 to your laptop via USB.
2. In the interactive menu, select option `[6] Connect ESP32 Hardware Link` and enter the COM port (e.g. `COM3` on Windows).
3. The demo automatically routes the quantized semantic bitstream to ESP32 #1, transmits it over **ESP-NOW 2.4 GHz RF**, receives it on ESP32 #2, and feeds it into the PyTorch semantic decoder!

---

## 🎓 College Review & Viva Cheat-Sheet

| What Examiners Ask | How to Answer with This Demo |
|---|---|
| *"Can we test with a custom sentence not in the dataset?"* | Type their exact sentence in Option `[2]`. Show how BERT tokenizes it, FSM retains core keywords, and the receiver reconstructs the semantic meaning. |
| *"Show me what happens at low SNR (-5 dB)."* | Use Option `[5]` to set SNR = -5 dB. Show that instead of a digital cliff-effect crash, semantic communication gracefully degrades while keeping the general shape/concept intact. |
| *"Which image features did FSM select?"* | Open `demo/outputs/vision_demo_*.png`. The center plot highlights the exact 16x16 pixel patches the FSM chose to transmit vs pruned. |
| *"What is your compression ratio?"* | Point to the **CBR (Channel Bandwidth Ratio)** metric: $\text{CBR} = 0.5 \implies 50\%$ reduction in transmitted symbols. |
