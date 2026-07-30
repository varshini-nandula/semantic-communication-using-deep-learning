# Semantic Communication Using Deep Learning: A Comprehensive Research Survey

> **Prepared for:** Final-year engineering major project (Research-Oriented, AIML Career Track)  
> **Date:** July 2026  
> **Scope:** Foundational theory → Literature survey (2019–2026) → Research gaps → Benchmarks → Feasible project ideas → Publication venues

---

## Table of Contents

1. [Foundational Understanding](#1-foundational-understanding)
2. [Existing Works Survey (2019–2026)](#2-existing-works-survey-20192026)
3. [Research Gaps and Open Problems](#3-research-gaps-and-open-problems)
4. [Benchmark Datasets and Tools](#4-benchmark-datasets-and-tools)
5. [Trend and Gap Analysis for Novel Contribution](#5-trend-and-gap-analysis-for-novel-contribution)
6. [Publication Venues](#6-publication-venues)
7. [References](#7-references)

---

## 1. Foundational Understanding

### 1.1 The Paradigm Shift: From Shannon to Semantics

Claude Shannon's landmark 1948 paper, *"A Mathematical Theory of Communication,"* established the foundational framework for modern communication systems. Shannon's model decomposes a communication system into three levels:

| Level | Focus | Goal |
|-------|-------|------|
| **Level A — Technical** | Accurate transmission of symbols | Minimize bit/symbol error rate |
| **Level B — Semantic** | How precisely symbols convey meaning | Maximize meaning fidelity |
| **Level C — Effectiveness** | How effectively received meaning affects behavior | Maximize task success |

Shannon deliberately scoped his theory to **Level A only**, stating: *"The semantic aspects of communication are irrelevant to the engineering problem."* For over seven decades, this assumption drove the design of every wireless standard from 2G to 5G — all optimized for **bit-level fidelity** measured by Bit Error Rate (BER) and throughput.

**Semantic communication** represents a fundamental paradigm shift by operating at **Level B and Level C**. Instead of transmitting raw bits, semantic communication systems:

- **Extract the meaning** (semantic features) of the source data at the transmitter
- **Transmit only the meaning-bearing representations** through the wireless channel
- **Reconstruct or regenerate the source** at the receiver using the received semantic features

This shift is motivated by the observation that in many applications (natural language, images, speech, video), **not all bits are equally important**. A human listener doesn't need every phoneme perfectly reproduced to understand a sentence — they need the *meaning* preserved.

### 1.2 Traditional Separate Source-Channel Coding (SSCC)

Shannon's separation theorem states that, under infinite block length assumptions, **source coding** (compression) and **channel coding** (error protection) can be designed separately without loss of optimality. This gives rise to the classical communication pipeline:

```
Source → Source Encoder → Channel Encoder → Modulation → Channel → Demodulation → Channel Decoder → Source Decoder → Sink
         (JPEG, H.265)    (LDPC, Turbo)     (QAM, OFDM)            (QAM, OFDM)    (LDPC, Turbo)    (JPEG, H.265)
```

**Limitations of SSCC in practice:**

| Limitation | Description |
|-----------|-------------|
| **Cliff Effect** | Performance degrades catastrophically when channel SNR falls below the designed threshold. No graceful degradation. |
| **Suboptimality at finite block lengths** | Shannon's separation theorem only holds asymptotically. Real-world finite-length codes suffer from a gap between source coding and channel coding optimization. |
| **Redundancy overhead** | Source coding aggressively removes redundancy, then channel coding adds it back — a fundamentally wasteful two-step process. |
| **Task agnosticism** | The pipeline doesn't know *why* data is being transmitted. A surveillance image and a birthday photo get identical treatment, even though task requirements differ drastically. |

### 1.3 Joint Source-Channel Coding (JSCC)

**Joint Source-Channel Coding (JSCC)** abandons the separation principle and instead jointly optimizes compression and error protection in a single, unified mapping:

```
Source → [Joint Encoder] → Channel Symbols → Channel → Received Symbols → [Joint Decoder] → Reconstructed Source
```

Classical (pre-deep-learning) JSCC methods existed but were limited in scope due to the difficulty of designing optimal joint codes analytically. **Deep learning** made JSCC practical and powerful by:

1. **Learning the joint mapping end-to-end** from data via gradient descent
2. **Adapting to any differentiable channel model** (AWGN, Rayleigh, etc.)
3. **Scaling to high-dimensional, complex source distributions** (images, speech, text)

### 1.4 The Autoencoder Framework for Communication

The key insight enabling deep JSCC is the **autoencoder interpretation** of a communication system:

```
                    ┌─────────────────────────────────────────────────────┐
                    │              End-to-End Autoencoder                  │
                    │                                                      │
  Source ──→  [ Neural Encoder ]  ──→  [ Channel (non-trainable) ]  ──→  [ Neural Decoder ]  ──→  Reconstruction
              (semantic + channel)       (AWGN / Rayleigh / etc.)         (channel + semantic)
                    │                                                      │
                    └──────────── Trained end-to-end via SGD ─────────────┘
```

- **Encoder** (parameterized by θ_E): Maps source data x to channel symbols z ∈ ℂ^k, where k is the number of channel uses (bandwidth). This simultaneously performs *semantic feature extraction* and *channel-adaptive encoding*.
- **Channel Layer**: A non-trainable, stochastic layer that simulates the physical channel. During training, it adds noise, fading, etc. During backpropagation, gradients flow through it (using the reparameterization trick for stochastic channels).
- **Decoder** (parameterized by θ_D): Maps received noisy symbols ẑ back to reconstructed source x̂.

The encoder and decoder are jointly trained to minimize a task-relevant loss:

```
ℒ = 𝔼_{x, channel}[ d(x, x̂) ]
```

where `d(·,·)` is a distortion metric appropriate to the modality:
- **Images:** MSE, 1 − SSIM, LPIPS (perceptual loss)
- **Text:** Cross-entropy, 1 − BLEU, 1 − sentence similarity
- **Speech:** MSE on spectrograms, PESQ, STOI

### 1.5 How Deep Semantic Communication Differs from Traditional Systems

| Aspect | Traditional (SSCC) | Deep Semantic Communication (JSCC) |
|--------|-------------------|-------------------------------------|
| **Design philosophy** | Modular, handcrafted blocks | End-to-end learned, data-driven |
| **Optimization** | Each block optimized independently | Joint optimization of entire pipeline |
| **What is transmitted** | Compressed bits + redundancy bits | Learned semantic representations (continuous-valued symbols) |
| **Channel adaptation** | Requires explicit CSI feedback and mode switching | Implicitly learned; graceful degradation |
| **Performance at low SNR** | Cliff effect (sudden failure) | Graceful degradation (smooth quality decline) |
| **Task awareness** | None — all data treated equally | Task-oriented encoding possible |
| **Evaluation metric** | BER, BLER, throughput | Semantic metrics (BLEU, SSIM, task accuracy) |

---

## 2. Existing Works Survey (2019–2026)

### 2.1 Text-Based Semantic Communication

#### 2.1.1 Seminal: DeepSC (Xie, Qin et al., 2021)

| Attribute | Details |
|-----------|---------|
| **Paper** | H. Xie, Z. Qin, G. Y. Li, B.-H. Juang, "Deep Learning Enabled Semantic Communication Systems," *IEEE Trans. Signal Processing*, vol. 69, pp. 2663–2675, 2021 |
| **Architecture** | **Transformer-based** encoder-decoder. Semantic encoder uses self-attention to extract meaning from input sentences. Channel encoder/decoder are dense layers that map semantic features to/from channel symbols. |
| **Channel Model** | AWGN, Rayleigh fading |
| **Metrics** | BLEU score, Sentence Similarity (cosine similarity of sentence embeddings) |
| **Dataset** | European Parliament Proceedings Parallel Corpus (Europarl) |
| **Key Contribution** | First end-to-end DL-based semantic communication system for text. Outperforms traditional SSCC at low SNR regimes. Introduces sentence-level similarity as evaluation metric. Supports transfer learning for channel adaptation. |
| **Citations** | 1500+ (as of 2026) — the most influential paper in semantic communication |

#### 2.1.2 Lightweight DeepSC (L-DeepSC)

| Attribute | Details |
|-----------|---------|
| **Paper** | H. Xie, Z. Qin et al., "Lite Distributed Semantic Communication Systems," *IEEE JSAC*, 2023 |
| **Architecture** | **DeLighT Transformer** — a deep-and-light-weight variant using block-wise scaling and group linear transformations to reduce parameters by ~3× vs. standard Transformer |
| **Channel Model** | AWGN, Rayleigh fading |
| **Metrics** | BLEU, Sentence Similarity |
| **Dataset** | Europarl |
| **Key Contribution** | Addresses computational overhead of standard Transformers for edge deployment. Achieves comparable semantic performance with significantly fewer parameters and FLOPs. |

#### 2.1.3 Robust DeepSC (R-DeepSC)

| Attribute | Details |
|-----------|---------|
| **Paper** | Various authors, 2022–2024 |
| **Architecture** | Transformer + adversarial training module + calibrated self-attention |
| **Channel Model** | AWGN, Rayleigh fading, **semantic noise** (adversarial perturbations) |
| **Metrics** | BLEU, Sentence Similarity, adversarial robustness |
| **Dataset** | Europarl |
| **Key Contribution** | Introduces robustness against "semantic noise" — adversarial perturbations that alter meaning rather than just signal quality. Uses adversarial training to harden the semantic encoder. |

#### 2.1.4 Knowledge-Enhanced Semantic Communication

| Attribute | Details |
|-----------|---------|
| **Paper** | Multiple works, 2023–2025 |
| **Architecture** | Transformer + external knowledge graph / shared knowledge base |
| **Channel Model** | AWGN, Rayleigh fading |
| **Metrics** | BLEU, Sentence Similarity, knowledge recall |
| **Dataset** | Europarl, custom domain-specific corpora |
| **Key Contribution** | Exploits shared background knowledge between transmitter and receiver to further compress transmitted information. If both sides know "Paris is the capital of France," the transmitter can send a more compressed representation. |

---

### 2.2 Image-Based Semantic Communication

#### 2.2.1 Seminal: DeepJSCC (Bourtsoulatze, Kurka, Gündüz, 2019)

| Attribute | Details |
|-----------|---------|
| **Paper** | E. Bourtsoulatze, D. B. Kurka, D. Gündüz, "Deep Joint Source-Channel Coding for Wireless Image Transmission," *IEEE Trans. Cognitive Comm. & Networking (TCCN)*, vol. 5, no. 3, pp. 567–579, Sep. 2019 |
| **Architecture** | **CNN-based autoencoder** — encoder uses convolutional layers to map images to complex-valued channel symbols; decoder uses transposed convolutions for reconstruction |
| **Channel Model** | AWGN, slow Rayleigh fading |
| **Metrics** | PSNR, SSIM, MS-SSIM |
| **Dataset** | CIFAR-10, ImageNet (Kodak for evaluation) |
| **Key Contribution** | First CNN-based JSCC for image transmission. Demonstrates graceful degradation (no cliff effect). Outperforms JPEG/JPEG2000 + LDPC at low SNR. Seminal paper establishing the DeepJSCC paradigm. |
| **Citations** | 900+ (as of 2026) |

#### 2.2.2 DeepJSCC-f (With Channel Output Feedback)

| Attribute | Details |
|-----------|---------|
| **Paper** | D. B. Kurka, D. Gündüz, "DeepJSCC-f: Deep Joint Source-Channel Coding of Images with Feedback," *IEEE JSAC*, 2020 |
| **Architecture** | CNN autoencoder + feedback link from receiver to transmitter |
| **Channel Model** | AWGN with feedback |
| **Metrics** | PSNR, SSIM |
| **Dataset** | CIFAR-10, Kodak |
| **Key Contribution** | Adds a feedback mechanism allowing the transmitter to refine its encoding based on receiver's decoding quality, progressively improving image quality over multiple transmission rounds. |

#### 2.2.3 SwinJSCC (Transformer-Based Image JSCC)

| Attribute | Details |
|-----------|---------|
| **Paper** | Various authors, 2023–2024 |
| **Architecture** | **Swin Transformer** encoder-decoder with channel-adaptive attention |
| **Channel Model** | AWGN, Rayleigh fading, varying SNR |
| **Metrics** | PSNR, SSIM, LPIPS |
| **Dataset** | CIFAR-10, DIV2K, Kodak |
| **Key Contribution** | Replaces CNN backbone with Vision Transformer (Swin Transformer), achieving better semantic understanding of image structure. Incorporates SNR-adaptive modules for single-model multi-SNR operation. |

#### 2.2.4 Diffusion-Aided Semantic Communication for Images (2024–2025)

| Attribute | Details |
|-----------|---------|
| **Paper** | Multiple works, 2024–2026 |
| **Architecture** | **JSCC encoder + Diffusion Model decoder** (e.g., Stable Diffusion, Latent Diffusion) |
| **Channel Model** | AWGN, Rayleigh fading |
| **Metrics** | PSNR, SSIM, LPIPS, **FID** (Fréchet Inception Distance), perceptual quality |
| **Dataset** | CIFAR-10, ImageNet, DIV2K, CelebA |
| **Key Contribution** | Uses diffusion models at the receiver to *regenerate* high-quality images from heavily compressed semantic features. Achieves extreme compression ratios (<1% of original data). Shifts focus from pixel-level fidelity (PSNR) to perceptual quality (FID, LPIPS). Represents the frontier of generative semantic communication. |

#### 2.2.5 GAN-Based Image Semantic Communication

| Attribute | Details |
|-----------|---------|
| **Paper** | Various authors, 2022–2024 |
| **Architecture** | **JSCC encoder + GAN decoder** (generator + discriminator) |
| **Channel Model** | AWGN, Rayleigh fading |
| **Metrics** | PSNR, SSIM, FID, LPIPS |
| **Dataset** | CelebA, CIFAR-10 |
| **Key Contribution** | Uses adversarial training to improve perceptual quality of reconstructed images. Produces sharper, more realistic outputs than MSE-trained models, though may sacrifice pixel-level metrics (PSNR) for perceptual metrics (FID). |

---

### 2.3 Speech/Audio-Based Semantic Communication

#### 2.3.1 DeepSC-S (Semantic Communication for Speech)

| Attribute | Details |
|-----------|---------|
| **Paper** | Z. Weng, Z. Qin et al., "Semantic Communication Systems for Speech Transmission," *IEEE JSAC*, 2021 |
| **Architecture** | **CNN-based** encoder for speech feature extraction + dense layers for channel coding |
| **Channel Model** | AWGN, Rayleigh fading |
| **Metrics** | PESQ, SNR of reconstructed speech, MSE on spectrograms |
| **Dataset** | TIMIT speech corpus |
| **Key Contribution** | First deep learning-based semantic communication for speech signals. Transmits semantic features of speech rather than waveform samples. Outperforms traditional codecs at low SNR. |

#### 2.3.2 DeepSC-ST (Speech Recognition + Synthesis)

| Attribute | Details |
|-----------|---------|
| **Paper** | H. Xie, Z. Qin et al., "Deep Learning Enabled Semantic Communications with Speech Recognition and Synthesis," *IEEE Trans. Wireless Comm.*, 2023 |
| **Architecture** | **Hybrid:** Speech recognition model (encoder side) + Text-to-Speech synthesis model (decoder side) + Joint semantic-channel coding |
| **Channel Model** | AWGN, Rayleigh fading |
| **Metrics** | Word Error Rate (WER), PESQ, Mean Opinion Score (MOS) |
| **Dataset** | LibriSpeech, LJSpeech |
| **Key Contribution** | Instead of transmitting speech waveforms, extracts *recognized text features* at the transmitter and *synthesizes* speech at the receiver. Achieves dramatically higher compression ratios than waveform-level approaches. Preserves speaker characteristics through auxiliary speaker embedding. |

#### 2.3.3 Audio Semantic Communication with Neural Codecs (2024–2025)

| Attribute | Details |
|-----------|---------|
| **Paper** | Various authors, 2024–2025 |
| **Architecture** | **Neural audio codec** (e.g., EnCodec, SoundStream) integrated with channel-adaptive layers |
| **Channel Model** | AWGN, frequency-selective fading |
| **Metrics** | PESQ, STOI, ViSQOL |
| **Dataset** | LibriSpeech, VCTK, DNS Challenge |
| **Key Contribution** | Leverages pre-trained neural audio codecs as the semantic backbone. These codecs already produce compact, meaning-rich representations that are then channel-coded for wireless transmission. Bridges the gap between audio ML and communication theory. |

---

### 2.4 Multi-Modal Semantic Communication

#### 2.4.1 U-DeepSC (Unified Multi-Modal)

| Attribute | Details |
|-----------|---------|
| **Paper** | Various authors, 2023–2024 |
| **Architecture** | **Unified Transformer backbone** with modality-specific encoder/decoder heads. Dynamic symbol allocation based on task and channel condition. |
| **Channel Model** | AWGN, Rayleigh fading |
| **Metrics** | Task-specific: BLEU (text), PSNR/SSIM (image), WER (speech), Accuracy (classification) |
| **Dataset** | Europarl (text), CIFAR-10/ImageNet (image), LibriSpeech (speech) |
| **Key Contribution** | Single unified architecture handles text, image, and speech simultaneously. Dynamically allocates channel bandwidth to each modality based on importance. First step toward a truly "universal" semantic transceiver. |

#### 2.4.2 Cross-Modal Semantic Communication (VQA, Image Captioning)

| Attribute | Details |
|-----------|---------|
| **Paper** | Multiple works, 2024–2025 |
| **Architecture** | **Cross-attention Transformer** or **GNN-based** fusion of text and image modalities |
| **Channel Model** | AWGN, Rayleigh fading |
| **Metrics** | VQA accuracy, CIDEr/METEOR (captioning), task success rate |
| **Dataset** | VQA v2.0, COCO Captions, Flickr30k |
| **Key Contribution** | Enables task-oriented multi-modal communication — e.g., transmitting an image and a question, with the receiver answering the question rather than reconstructing the image. Explores cross-modal alignment in the semantic space. |

---

### 2.5 Summary Table of Major Works

| Year | Paper / System | Modality | Architecture | Channel | Metrics | Dataset |
|------|---------------|----------|-------------|---------|---------|---------|
| 2019 | DeepJSCC (Bourtsoulatze et al.) | Image | CNN Autoencoder | AWGN, Rayleigh | PSNR, SSIM | CIFAR-10, ImageNet |
| 2020 | DeepJSCC-f (Kurka & Gündüz) | Image | CNN + Feedback | AWGN | PSNR, SSIM | CIFAR-10, Kodak |
| 2021 | DeepSC (Xie, Qin et al.) | Text | Transformer | AWGN, Rayleigh | BLEU, Sentence Similarity | Europarl |
| 2021 | DeepSC-S (Weng, Qin et al.) | Speech | CNN | AWGN, Rayleigh | PESQ, MSE | TIMIT |
| 2022 | R-DeepSC | Text | Transformer + Adversarial | AWGN, Rayleigh, Semantic Noise | BLEU, Adversarial Robustness | Europarl |
| 2023 | DeepSC-ST (Xie, Qin et al.) | Speech | Recognition + Synthesis | AWGN, Rayleigh | WER, PESQ, MOS | LibriSpeech |
| 2023 | L-DeepSC | Text | DeLighT Transformer | AWGN, Rayleigh | BLEU, Sentence Similarity | Europarl |
| 2023 | SwinJSCC | Image | Swin Transformer | AWGN, Rayleigh | PSNR, SSIM, LPIPS | DIV2K, Kodak |
| 2023–24 | U-DeepSC | Multi-Modal | Unified Transformer | AWGN, Rayleigh | Task-specific | Multiple |
| 2024 | GAN-JSCC | Image | CNN + GAN | AWGN, Rayleigh | PSNR, FID, LPIPS | CelebA, CIFAR-10 |
| 2024–25 | Diffusion-JSCC | Image | JSCC + Diffusion Model | AWGN, Rayleigh | PSNR, FID, LPIPS | ImageNet, DIV2K |
| 2024–25 | Cross-Modal SemCom | Multi-Modal | Cross-Attention Transformer | AWGN, Rayleigh | VQA Acc., CIDEr | VQA v2.0, COCO |
| 2025 | LLM-Native SemCom | Text/Multi | LLM (GPT/BART) + Channel Adapter | AWGN, Rayleigh, OFDM | BLEU, Semantic Accuracy | Europarl, custom |
| 2025–26 | SA-RA-JSCC | Image | SNR-Adaptive JSCC | AWGN, Rayleigh, varying SNR | PSNR, SSIM | CIFAR-10, Kodak |

---

## 3. Research Gaps and Open Problems

### 3.1 Generalization Across Channel Conditions

**Current State:** Most semantic communication models are trained and evaluated on a single channel model (typically AWGN) at a fixed or narrow range of SNR values.

**The Problem:** Real wireless channels exhibit time-varying fading, frequency selectivity, Doppler shifts, and interference — conditions that differ drastically from the AWGN assumed during training. Models trained on AWGN often fail catastrophically on Rayleigh fading and vice versa.

**What's Needed:**
- Channel-agnostic or meta-learning-based semantic encoders
- Domain randomization across channel distributions during training
- Adversarial training with worst-case channel perturbations
- Few-shot adaptation mechanisms for unseen channel types

### 3.2 Semantic Noise Robustness

**The Problem:** Beyond physical channel noise, semantic communication faces a new class of distortions — **semantic noise** — where the *meaning* is perturbed while the signal may be intact. Examples include:
- Adversarial attacks that flip semantic meaning
- Ambiguous source inputs that confuse the encoder
- Distribution shift in source data (e.g., training on formal text, deploying on slang)

**What's Needed:**
- Formal definitions and metrics for semantic noise
- Certified robustness guarantees for semantic encoders
- Adversarial training frameworks specifically designed for semantic fidelity

### 3.3 Lack of Standardized Evaluation Metrics

> **⚠️ CRITICAL GAP:** This is arguably the most important open problem. There is no universally agreed-upon metric for "semantic fidelity."

**The Problem:** Different papers use different metrics (BLEU, sentence similarity, PSNR, SSIM, FID, task accuracy), making cross-paper comparison nearly impossible. BLEU score, for instance, poorly correlates with human judgment of semantic preservation. PSNR doesn't capture perceptual quality.

**What's Needed:**
- A standardized benchmark suite (akin to GLUE/SuperGLUE for NLP) specifically for semantic communication
- Human-in-the-loop evaluation protocols
- Learnable semantic similarity metrics that generalize across modalities

### 3.4 Low-Latency / Real-Time Constraints

**The Problem:** Transformer-based architectures (used in DeepSC, SwinJSCC) and diffusion models (used in generative SemCom) have high computational latency. Diffusion models, in particular, require hundreds of denoising steps, making real-time deployment impractical.

**What's Needed:**
- Distilled diffusion models with fewer than 10 sampling steps
- Lightweight Transformer variants (MobileViT, EfficientFormer) for semantic encoding
- Hardware-aware neural architecture search (NAS) for communication-specific models
- Streaming / causal architectures that process input incrementally

### 3.5 Multi-User and Multi-Task Semantic Communication

**The Problem:** Almost all existing work considers **single-user, single-task** scenarios. Real-world 6G networks will involve:
- Multiple users sharing the same spectrum
- Multiple tasks (e.g., one user needs image classification, another needs image reconstruction)
- Heterogeneous devices with different computational capabilities

**What's Needed:**
- Multi-access semantic communication protocols (semantic NOMA, semantic OFDMA)
- Task-aware resource allocation (allocate more bandwidth to safety-critical tasks)
- Hierarchical semantic encoding (base layer for all users, enhancement layers for specific tasks)

### 3.6 Security and Privacy of Semantic Representations

**The Problem:** Semantic representations encode rich, meaning-dense information. If intercepted, an eavesdropper could reconstruct the full message from fewer bits. New threat surfaces include:
- **Semantic eavesdropping:** Extracting meaning from intercepted semantic features
- **Model inversion attacks:** Recovering private training data from model parameters
- **Knowledge base poisoning:** Corrupting the shared knowledge base to cause systematic misinterpretation
- **Adversarial manipulation:** Injecting perturbations to alter the decoded meaning

**What's Needed:**
- Encryption schemes for semantic representations (not just bit-level encryption)
- Differential privacy guarantees for semantic encoders
- Federated learning approaches for distributed semantic model training
- Physical layer security integrated with semantic coding

### 3.7 Computational Complexity for Edge Deployment

**The Problem:** State-of-the-art semantic communication models have millions to billions of parameters (especially LLM-based and diffusion-based systems), making them unsuitable for edge devices (IoT sensors, drones, wearables) with limited memory, compute, and battery.

**What's Needed:**
- Knowledge distillation from large teacher models to compact student models
- Quantization-aware training (INT8/INT4 inference)
- Pruning and low-rank factorization of Transformer layers
- Split computing: offload heavy computation to edge server, run lightweight encoder on device

### 3.8 Integration with 5G/6G Physical Layer Standards

**The Problem:** Academic semantic communication models use idealized channel models (flat AWGN, i.i.d. Rayleigh). Real 5G NR uses OFDM with specific numerologies, LDPC/polar codes, HARQ, MIMO, and beam management. No existing work demonstrates seamless integration with standardized physical layers.

**What's Needed:**
- OFDM-compatible semantic encoding (mapping semantic features to OFDM subcarriers)
- Coexistence with legacy systems (backward compatibility)
- Standardization proposals for semantic communication in 3GPP
- Integration demonstrations using tools like Sionna or MATLAB 5G Toolbox

### 3.9 Explainability and Interpretability

**The Problem:** Semantic encoders are black boxes. It's unclear *what semantic features* the encoder learns to preserve or discard. This makes debugging, certification, and deployment in safety-critical domains (healthcare, autonomous driving) difficult.

**What's Needed:**
- Attention visualization and feature attribution for semantic encoders
- Probing tasks to characterize what information is preserved in semantic representations
- Explainable AI (XAI) integration with communication system design

---

## 4. Benchmark Datasets and Tools

### 4.1 Datasets by Modality

#### Text Datasets

| Dataset | Description | Size | Use in SemCom |
|---------|-------------|------|---------------|
| **Europarl** (European Parliament Proceedings) | Parallel corpus of European Parliament transcripts in 21 languages | ~2M sentences | Standard benchmark for DeepSC and text-based SemCom; used in nearly all text SemCom papers |
| **WMT Datasets** (Workshop on Machine Translation) | Large-scale parallel text corpora for machine translation | Varies by year | Useful for cross-lingual semantic communication evaluation |
| **Penn Treebank (PTB)** | English text with syntactic annotations | ~1M words | Language modeling baselines |
| **SST-2 / GLUE / SuperGLUE** | NLP benchmark suites | Various | Task-oriented evaluation of semantic preservation |

#### Image Datasets

| Dataset | Description | Size | Use in SemCom |
|---------|-------------|------|---------------|
| **CIFAR-10/100** | Small natural images in 10/100 categories | 60K images (32×32) | Rapid prototyping and training of image JSCC models |
| **ImageNet (ILSVRC)** | Large-scale natural image classification | 1.2M images (224×224+) | Full-scale evaluation of image semantic communication |
| **Kodak** | 24 high-quality photographic images | 24 images (768×512) | Standard evaluation set for image quality comparison |
| **DIV2K** | High-resolution images for super-resolution | 1000 images (2K) | Training high-resolution DeepJSCC and diffusion models |
| **CelebA** | Celebrity face images with attributes | 200K images | GAN-based and face-specific semantic communication |
| **COCO** | Common Objects in Context | 330K images + annotations | Multi-modal SemCom (image captioning, VQA) |

#### Speech/Audio Datasets

| Dataset | Description | Size | Use in SemCom |
|---------|-------------|------|---------------|
| **TIMIT** | Phonetically transcribed speech | 6300 utterances | Foundational speech SemCom (DeepSC-S) |
| **LibriSpeech** | Large-scale read English speech | 1000 hours | DeepSC-ST and large-scale speech SemCom |
| **LJSpeech** | Single-speaker English speech | 24 hours | Text-to-speech synthesis at receiver |
| **VCTK** | Multi-speaker English speech | 44 hours, 110 speakers | Speaker-adaptive speech SemCom |
| **DNS Challenge** | Noisy speech for denoising | Large-scale | Robustness evaluation |

#### Video Datasets

| Dataset | Description | Size | Use in SemCom |
|---------|-------------|------|---------------|
| **UCF-101** | Action recognition videos | 13K clips, 101 classes | Task-oriented video SemCom |
| **Kinetics-400/700** | Large-scale action recognition | 300K+ clips | Scalable video SemCom evaluation |
| **UVG** | Ultra Video Group test sequences | 16 sequences (4K) | Video codec comparison baseline |

### 4.2 Simulation Frameworks and Tools

| Tool | Developer | Key Features | URL |
|------|-----------|-------------|-----|
| **Sionna** | NVIDIA Research | GPU-accelerated, fully differentiable link-level simulator. Supports OFDM, MIMO, LDPC, Polar codes, ray tracing. Ideal for end-to-end learning with semantic communication models. PyTorch/TensorFlow integration. | [github.com/NVlabs/sionna](https://github.com/NVlabs/sionna) |
| **NVIDIA Sionna Research Kit (SRK)** | NVIDIA | Extends Sionna for 5G NR-compliant software-defined RAN. Enables deployment of AI/ML components into real-world radio access networks. | NVIDIA Developer |
| **CommPy** | Open Source | Python library for digital communication simulations (modulation, channel coding, channel models). Lightweight alternative to Sionna. | [github.com/veeresht/CommPy](https://github.com/veeresht/CommPy) |
| **MATLAB Communications Toolbox** | MathWorks | Comprehensive toolbox for communication system design. Includes 5G Toolbox, LTE Toolbox, and channel modeling. | MathWorks |
| **PyTorch / TensorFlow** | Meta / Google | Core deep learning frameworks used to implement semantic encoder/decoder architectures. | pytorch.org / tensorflow.org |
| **Hugging Face Transformers** | Hugging Face | Pre-trained NLP models (BERT, GPT, BART) that can serve as semantic backbones. | huggingface.co |
| **GNU Radio** | Open Source | SDR framework for over-the-air experiments (not required for simulation-only projects). | gnuradio.org |

### 4.3 Open-Source Semantic Communication Repositories

| Repository | Description |
|-----------|-------------|
| **DeepSC (Official)** | TensorFlow implementation of DeepSC for text semantic communication. Available on GitHub. |
| **DeepJSCC (Gündüz Lab)** | PyTorch implementations of DeepJSCC and variants (DeepJSCC-f, DeepJSCC-Q). Imperial College London. |
| **SemCom-Survey GitHub** | Curated lists of semantic communication papers, code, and datasets maintained by the research community. |

---

## 5. Trend and Gap Analysis for Novel Contribution

Based on the comprehensive survey above, here are **5 specific, narrow, and feasible problem statements** suitable for a 3-month solo project:

---

### 💡 Project Idea 1: SNR-Adaptive Semantic Communication for Images Using a Single Unified Model

> **Gap Addressed:** Most DeepJSCC models are trained at a fixed SNR and perform poorly when deployed at mismatched SNR. Training separate models for each SNR is impractical.

**Problem Statement:** Design a single DeepJSCC model for image transmission that adapts to arbitrary channel SNR at inference time, without retraining.

**Proposed Approach:**
- Implement a CNN-based DeepJSCC backbone (following Bourtsoulatze et al.)
- Add **SNR-conditional feature modulation** — inject SNR information into intermediate feature maps via FiLM (Feature-wise Linear Modulation) layers
- Train on a *distribution* of SNR values (e.g., uniform over [0, 20] dB)
- Evaluate on both seen and unseen SNR values under AWGN and Rayleigh fading

**Feasibility:**
- ✅ Software-only (PyTorch + simulated channels)
- ✅ Public datasets (CIFAR-10, Kodak)
- ✅ Single-GPU training feasible (~2–3 days on CIFAR-10)
- ✅ Clear baseline (fixed-SNR DeepJSCC)
- ✅ Publishable novelty if FiLM-based adaptation outperforms naive approaches

**Expected Deliverables:**
- PyTorch codebase with reproducible experiments
- Performance curves (PSNR/SSIM vs. SNR) comparing adaptive vs. fixed models
- Analysis of learned modulation parameters across SNR values

---

### 💡 Project Idea 2: Knowledge Distillation for Lightweight Text Semantic Communication

> **Gap Addressed:** Transformer-based DeepSC models are too large for edge devices. Knowledge distillation for semantic communication is underexplored.

**Problem Statement:** Distill a full-size Transformer-based DeepSC model into a lightweight student model (CNN or small Transformer) that preserves semantic fidelity while reducing parameters by 5–10×.

**Proposed Approach:**
- Train a standard DeepSC (Transformer) as the **teacher** model
- Design a **student** model using either (a) a 2-layer CNN or (b) a tiny Transformer (2 layers, 2 heads)
- Distill using a combination of (i) KL divergence on output distributions, (ii) intermediate feature matching, (iii) semantic similarity alignment
- Evaluate on Europarl dataset under AWGN and Rayleigh channels

**Feasibility:**
- ✅ Software-only (PyTorch)
- ✅ Public dataset (Europarl — freely available)
- ✅ Teacher model training: ~1 day on single GPU
- ✅ Distillation experiments: ~2–3 days each
- ✅ Clear research narrative: "Can we achieve 80% of DeepSC quality with 10% of parameters?"

**Expected Deliverables:**
- Pareto curve: Semantic Similarity vs. Model Parameters
- Ablation study on distillation loss components
- Latency/throughput benchmarks on CPU inference

---

### 💡 Project Idea 3: Adversarial Robustness of Image Semantic Communication Systems

> **Gap Addressed:** Semantic noise robustness is identified but not systematically studied for image modality. Most adversarial robustness research focuses on classification models, not communication systems.

**Problem Statement:** Evaluate and improve the robustness of DeepJSCC image transmission against adversarial perturbations that target semantic content (e.g., making a "stop sign" image reconstruct as a "speed limit" sign).

**Proposed Approach:**
- Implement DeepJSCC for image transmission (CNN-based)
- Design **semantic adversarial attacks**: perturbations to the *transmitted channel symbols* (not the source image) that maximize semantic distortion at the receiver, measured by both pixel-level (PSNR drop) and semantic-level (classifier accuracy drop) metrics
- Defend using (a) adversarial training, (b) randomized smoothing of channel symbols, (c) semantic consistency verification at the receiver
- Evaluate on CIFAR-10 (classification task) and Kodak (reconstruction task)

**Feasibility:**
- ✅ Software-only (PyTorch)
- ✅ Public datasets (CIFAR-10, Kodak)
- ✅ Builds directly on existing DeepJSCC codebase
- ✅ Novel angle: adversarial attacks on the channel symbol space (not the input space)
- ✅ Highly publishable in security/communication venues

---

### 💡 Project Idea 4: Standardized Semantic Communication Benchmark Suite

> **Gap Addressed:** The most critical meta-gap — no standardized evaluation framework exists for comparing semantic communication systems.

**Problem Statement:** Design and implement an open-source benchmark suite for evaluating semantic communication systems across modalities, channel conditions, and metrics.

**Proposed Approach:**
- Implement **3 baseline systems**: (a) DeepSC for text, (b) DeepJSCC for images, (c) a simple speech JSCC
- Define a **standardized evaluation protocol** with fixed:
  - Train/val/test splits for Europarl, CIFAR-10, and TIMIT
  - Channel conditions: AWGN at SNR ∈ {0, 5, 10, 15, 20} dB; Rayleigh fading at same SNRs
  - Bandwidth ratios: k/n ∈ {1/6, 1/3, 1/2}
  - Metrics: BLEU, Sentence Similarity, PSNR, SSIM, PESQ, FLOPs, latency
- Provide **leaderboard-ready evaluation scripts**
- Open-source the entire suite with documentation

**Feasibility:**
- ✅ Software-only (PyTorch)
- ✅ All public datasets
- ✅ High-impact community contribution (fills a critical gap)
- ✅ Publishable as a "benchmark paper" or "datasets & benchmarks" track paper
- ⚠️ Requires careful software engineering (test harness, reproducibility)

---

### 💡 Project Idea 5: Diffusion-Aided Semantic Communication for Extreme Image Compression

> **Gap Addressed:** Diffusion models show promise for semantic image regeneration, but practical systems with feasible inference speed and controllable fidelity are still lacking.

**Problem Statement:** Build an image semantic communication system where the transmitter sends a compact semantic sketch (segmentation map + textual caption), and the receiver uses a lightweight diffusion model to regenerate the image.

**Proposed Approach:**
- **Transmitter pipeline:** Input image → Semantic segmentation (using pre-trained DeepLabV3) + Image captioning (using pre-trained BLIP) → Channel encode the segmentation map + caption → Transmit
- **Receiver pipeline:** Channel decode → Use segmentation map + caption as conditioning input to a **ControlNet-guided Stable Diffusion** model → Generate reconstructed image
- **Channel model:** AWGN and Rayleigh fading
- **Evaluation:** PSNR, SSIM, LPIPS, FID vs. bandwidth ratio; compare against DeepJSCC baseline
- **Efficiency:** Use distilled diffusion model (LCM or SDXL-Turbo) for 4-step inference

**Feasibility:**
- ✅ Software-only (PyTorch, Hugging Face Diffusers)
- ✅ Public datasets (COCO, DIV2K)
- ✅ Pre-trained models available (ControlNet, Stable Diffusion, BLIP, DeepLabV3)
- ✅ Extreme compression ratio is inherently impressive for demonstration
- ⚠️ Requires GPU with ≥12GB VRAM for diffusion inference
- ✅ Highly publishable — generative SemCom is the hottest sub-topic

---

### Project Idea Comparison Matrix

| Criterion | Idea 1 (SNR-Adaptive) | Idea 2 (Distillation) | Idea 3 (Adversarial) | Idea 4 (Benchmark) | Idea 5 (Diffusion) |
|-----------|:----:|:----:|:----:|:----:|:----:|
| Novelty | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Feasibility (3 months) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| GPU Requirements | Low | Low | Medium | Low–Medium | High |
| Publishability | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| AIML Skill Showcase | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Community Impact | ⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

> **💡 Recommended strategy:** Start with **Idea 1 (SNR-Adaptive)** as a warm-up / fallback (simplest, fastest to complete), then pursue **Idea 3 (Adversarial Robustness)** or **Idea 5 (Diffusion-Aided)** as your primary contribution. If you want maximum community impact and a strong portfolio piece, **Idea 4 (Benchmark Suite)** is underappreciated but extremely valuable.

---

## 6. Publication Venues

### 6.1 IEEE Conferences (Primary Targets)

| Venue | Full Name | Typical Deadline | Acceptance Rate | Notes |
|-------|-----------|-----------------|-----------------|-------|
| **IEEE GLOBECOM** | IEEE Global Communications Conference | ~Apr–May | ~38–42% | Flagship venue. Has dedicated workshops on semantic communication (e.g., "Semantic Communications Meets Wireless in AI Era"). Workshop papers are 6 pages; main track is 6 pages. |
| **IEEE ICC** | IEEE International Conference on Communications | ~Oct–Nov | ~38–40% | Co-flagship with GLOBECOM. Workshop "Beyond Bits: Goal-Oriented and Semantic Communication in the Generative AI Era" directly relevant. |
| **IEEE WCNC** | IEEE Wireless Communications and Networking Conference | ~Sep–Oct | ~45% | More accessible than GLOBECOM/ICC. Good for first publications. |
| **IEEE VTC** | IEEE Vehicular Technology Conference (Spring/Fall) | Rolling | ~50% | Task-oriented and vehicular semantic communication papers. |
| **IEEE ICCC** | IEEE/CIC International Conference on Communications in China | ~Jun | ~40–45% | Has hosted workshops on "Semantic Communications for Industrial IoT." Good regional venue. |
| **IEEE ICUFN** | International Conference on Ubiquitous and Future Networks | Varies | ~50% | Accepts DL-based semantic communication papers. |
| **IEEE FNWF** | Future Networks World Forum | Varies | Moderate | Dedicated *Symposium on Semantic Communications in Future Networks*. |
| **IEEE ICSC** | International Conference on Semantic Computing | ~Nov | Moderate | Broader semantic computing, but relevant for the foundational aspects. |

### 6.2 IEEE Journals (Higher Impact, Longer Timeline)

| Journal | Full Name | Impact Factor (approx.) | Review Time | Notes |
|---------|-----------|---------------------|-------------|-------|
| **IEEE TWC** | Transactions on Wireless Communications | ~8.9 | 3–6 months | Top venue for DeepJSCC and semantic communication. DeepSC-ST was published here. |
| **IEEE TCCN** | Transactions on Cognitive Comm. & Networking | ~7.4 | 3–6 months | DeepJSCC seminal paper was published here. Directly relevant. |
| **IEEE TSP** | Transactions on Signal Processing | ~5.0 | 3–6 months | DeepSC seminal paper was published here. |
| **IEEE JSAC** | Journal on Selected Areas in Communications | ~13.8 | 4–8 months | Premium venue. Has published special issues on semantic communication. |
| **IEEE COMST** | Communications Surveys & Tutorials | ~34.4 | 6–12 months | For comprehensive survey papers. Not suitable for a first project paper. |
| **IEEE OJ-COMS** | Open Journal of the Communications Society | ~6.3 | 2–4 months | Open access. Frequently publishes special sections on semantic communication topics. Faster review cycle. **Good target for student work.** |
| **IEEE WCL** | Wireless Communications Letters | ~4.6 | 1–3 months | Short 4-page papers. Fastest IEEE publication for incremental contributions. **Excellent for student first publication.** |
| **IEEE CL** | Communications Letters | ~3.7 | 1–3 months | Similar to WCL. Good for concise contributions. |

### 6.3 Non-IEEE Venues

| Venue | Type | Notes |
|-------|------|-------|
| **Elsevier Physical Communication** | Journal | Accepts semantic communication papers. Lower bar than IEEE transactions. |
| **Elsevier Computer Networks** | Journal | For networking-oriented semantic communication (multi-user, resource allocation). |
| **Springer EURASIP JWCN** | Journal | Open access. Journal on Wireless Communications and Networking. |
| **MDPI Sensors / Electronics / Applied Sciences** | Journals | Open access. Lower bar, faster review. Good for first publications but lower prestige. |
| **EAI Endorsed Transactions** | Journal | Publishes semantic communication surveys and systems papers. |
| **ACM MobiCom / MobiSys** | Conferences | If your work has a systems / deployment angle (rare for SemCom papers). |

### 6.4 Preprint / arXiv-First Strategy

> **💡 Recommended approach for a student:**
>
> 1. **Post to arXiv first** (cs.IT, cs.LG, or eess.SP categories) to establish priority and get visibility
> 2. **Submit to an IEEE workshop** (GLOBECOM or ICC workshop — shorter papers, higher acceptance rate, faster turnaround)
> 3. **Expand to a journal paper** (IEEE WCL or IEEE OJ-COMS) based on reviewer feedback
>
> This pipeline is standard in the semantic communication community and maximizes your chances of having at least one publication within the project timeline.

### 6.5 Recent Student-Accessible Publications (Examples)

| Paper Type | Where Published | Typical Length | Timeline |
|-----------|----------------|---------------|----------|
| Workshop paper | GLOBECOM/ICC Workshop | 5–6 pages | Submit → Accept: ~3 months |
| Conference paper | WCNC, VTC | 5–6 pages | Submit → Accept: ~4 months |
| Letters | IEEE WCL, IEEE CL | 4 pages | Submit → Accept: ~2–3 months |
| Journal paper | IEEE OJ-COMS | 10–15 pages | Submit → Accept: ~3–5 months |
| Preprint | arXiv | Any length | Immediate (1–2 days) |

---

## 7. References

### Seminal Papers

1. **C. E. Shannon**, "A Mathematical Theory of Communication," *Bell System Technical Journal*, vol. 27, pp. 379–423, 1948.
2. **E. Bourtsoulatze, D. B. Kurka, D. Gündüz**, "Deep Joint Source-Channel Coding for Wireless Image Transmission," *IEEE Trans. Cognitive Comm. & Networking*, vol. 5, no. 3, pp. 567–579, Sep. 2019.
3. **H. Xie, Z. Qin, G. Y. Li, B.-H. Juang**, "Deep Learning Enabled Semantic Communication Systems," *IEEE Trans. Signal Processing*, vol. 69, pp. 2663–2675, 2021.
4. **Z. Weng, Z. Qin et al.**, "Semantic Communication Systems for Speech Transmission," *IEEE JSAC*, 2021.
5. **D. B. Kurka, D. Gündüz**, "DeepJSCC-f: Deep Joint Source-Channel Coding of Images with Feedback," *IEEE JSAC*, 2020.
6. **H. Xie, Z. Qin et al.**, "Deep Learning Enabled Semantic Communications with Speech Recognition and Synthesis," *IEEE Trans. Wireless Comm.*, 2023.

### Survey Papers

7. **W. Yang et al.**, "A Contemporary Survey on Semantic Communications: Theory of Mind, Generative AI, and Deep Joint Source-Channel Coding," *arXiv*, 2025.
8. **Various**, "A Survey on Semantic Communication Networks: Architecture, Security, and Privacy," *arXiv*, 2024.
9. **Various**, "From Bits to Meaning: A Survey of Semantic Communications for 6G Networks," *EAI Endorsed Trans.*, 2025/2026.
10. **Various**, "Deep Learning in Wireless Communication Receivers: A Survey," *arXiv*, 2025.

### Key Extended Works

11. **D. B. Kurka, D. Gündüz**, "DeepJSCC-Q: Channel Input Constrained Deep Joint Source-Channel Coding," *IEEE*, 2022.
12. **Various**, "SwinJSCC: Swin Transformer-based Deep Joint Source-Channel Coding," 2023–2024.
13. **Various**, "U-DeepSC: Unified Multi-Modal Semantic Communication," 2023–2024.
14. **Various**, "Diffusion-Aided Semantic Communication for Image Transmission," 2024–2025.
15. **Various**, "Lightweight DeepSC using DeLighT Transformers," *IEEE JSAC*, 2023.

---

> **Disclaimer:** This survey synthesizes information from publicly available research papers, arXiv preprints, IEEE Xplore, and community repositories as of July 2026. Citation counts and publication details should be verified against the original sources before inclusion in a formal publication. All project ideas are original proposals based on identified research gaps and should be validated with your project advisor.

---

*Survey compiled: July 30, 2026*
