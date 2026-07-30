# XMA-SemCom: Explainable Multimodal Adaptive Semantic Communication Using Deep Learning

> **Project Type:** Research-Oriented Major Project (Final Year Engineering)  
> **Career Track:** AI/ML Research & Engineering  
> **Phase 1 Scope:** Text + Image (3 months)  
> **Phase 2 Scope:** Audio + Video Extension (future work)  
> **Date:** July 2026

---

## Table of Contents

1. [Abstract](#1-abstract)
2. [Motivation and Problem Statement](#2-motivation-and-problem-statement)
3. [Literature Landscape and Gap Analysis](#3-literature-landscape-and-gap-analysis)
4. [Research Questions](#4-research-questions)
5. [Proposed System Architecture](#5-proposed-system-architecture)
6. [Technical Deep Dive: Component Design](#6-technical-deep-dive-component-design)
7. [Explainability Framework](#7-explainability-framework)
8. [Datasets and Experimental Setup](#8-datasets-and-experimental-setup)
9. [Evaluation Metrics and Protocol](#9-evaluation-metrics-and-protocol)
10. [Implementation Roadmap](#10-implementation-roadmap)
11. [Expected Contributions and Novelty Claims](#11-expected-contributions-and-novelty-claims)
12. [Phase 2: Audio and Video Extension](#12-phase-2-audio-and-video-extension)
13. [Publication Strategy](#13-publication-strategy)
14. [References](#14-references)

---

## 1. Abstract

Modern wireless communication systems are undergoing a paradigm shift from Shannon's classical bit-level transmission toward **semantic communication**, where only the *meaning* of data is transmitted, enabling dramatic bandwidth savings. Concurrently, deep learning models powering these systems remain **opaque black boxes** — no existing work systematically explains *what* a multimodal semantic encoder learns to preserve, *how* it trades off between modalities under degrading channel conditions, or *why* certain semantic features survive channel noise while others are lost.

This work proposes **XMA-SemCom** (Explainable Multimodal Adaptive Semantic Communication), a dual-modality (text + image) semantic communication system with an integrated explainability framework. The system comprises:

1. **A multimodal semantic communication pipeline** — independent text (Transformer-based) and image (CNN-based) semantic encoders with Joint Source-Channel Coding (JSCC), connected through a cross-modal fusion module that learns to align and prioritize features across modalities.

2. **A comprehensive XAI analysis framework** — employing Grad-CAM for spatial importance mapping in image encoders, attention rollout for text encoder interpretability, and SHAP-based feature attribution for cross-modal importance scoring, all evaluated across a systematic grid of channel conditions (AWGN and Rayleigh fading at SNR ∈ {0, 5, 10, 15, 20} dB).

3. **An importance-aware adaptive transmission module** — leveraging the explainability scores to perform dynamic bandwidth allocation, dedicating more channel uses to semantically critical features and pruning low-importance features, demonstrating that interpretability directly improves communication efficiency.

We evaluate on public benchmarks (Flickr30k for image-text pairs, CIFAR-10 for images, Europarl for text) and provide the first systematic study of:
- How multimodal semantic encoders dynamically rebalance modality importance under channel degradation
- Whether XAI-derived importance scores correlate with actual semantic reconstruction fidelity
- Whether importance-aware feature pruning can reduce bandwidth by 20–40% with less than 5% semantic quality loss

This work addresses a critical gap identified by Zia et al. (2026) in *Computer Networks* and fills the intersection of three active research fronts — multimodal semantic communication, explainable AI for wireless systems, and adaptive resource allocation — producing the first explainable multimodal semantic communication system in the literature.

---

## 2. Motivation and Problem Statement

### 2.1 Why This Problem Matters

Three independent trends are converging to create a critical research gap:

```
┌──────────────────────┐     ┌──────────────────────┐     ┌──────────────────────┐
│   Semantic Comm.     │     │   Multimodal AI      │     │   Explainable AI     │
│                      │     │                      │     │                      │
│ • DeepSC (text)      │     │ • CLIP, LLaVA        │     │ • Grad-CAM, SHAP     │
│ • DeepJSCC (image)   │     │ • Cross-attention     │     │ • LIME, probing      │
│ • U-DeepSC (multi)   │     │ • Fusion networks     │     │ • Attention analysis  │
│                      │     │                      │     │                      │
│ PROBLEM: Black box   │     │ PROBLEM: Not adapted  │     │ PROBLEM: Never       │
│ encoders, no XAI     │     │ for wireless channels │     │ applied to SemCom    │
└──────────┬───────────┘     └──────────┬───────────┘     └──────────┬───────────┘
           │                            │                            │
           └────────────────┬───────────┘────────────────────────────┘
                            │
                   ┌────────▼────────┐
                   │  XMA-SemCom     │
                   │  (This Project) │
                   │                 │
                   │  Fills the gap  │
                   │  at the center  │
                   └─────────────────┘
```

### 2.2 The Specific Problem

**No existing work answers the following questions:**

1. **What does a multimodal semantic encoder actually learn?** When a system jointly transmits text and images through a noisy wireless channel, which features does it prioritize? Does it learn to preserve object identity in images? Syntactic structure in text? Semantic relationships between modalities?

2. **How do encoders trade off between modalities under noise?** At low SNR (harsh channel), does the system sacrifice image detail to protect text meaning? Does it shift reliance from one modality to another? Is this behavior consistent or chaotic?

3. **Can we use explainability to make transmission smarter?** If we know which features are semantically important (via XAI), can we transmit only those and save bandwidth? How much bandwidth can we save before quality degrades?

4. **Why do semantic communication systems fail?** When the reconstructed text/image is wrong, is it because the encoder failed to extract the right features, or because the channel corrupted the important ones? XAI can answer this by diagnosing failures at the feature level.

### 2.3 Formal Problem Statement

> **Design and implement a multimodal (text + image) semantic communication system with integrated explainability, and use the explainability framework to (a) systematically analyze cross-modal feature importance under varying channel conditions, (b) diagnose failure modes, and (c) demonstrate that XAI-guided importance-aware transmission reduces bandwidth without significant semantic quality loss.**

---

## 3. Literature Landscape and Gap Analysis

### 3.1 What Exists Today

| Research Area | Maturity | Key Works | Count |
|---|:---:|---|:---:|
| **Single-modal SemCom (text)** | 🟢 Mature | DeepSC (Xie 2021), L-DeepSC, R-DeepSC | 50+ papers |
| **Single-modal SemCom (image)** | 🟢 Mature | DeepJSCC (Bourtsoulatze 2019), SwinJSCC, Diffusion-JSCC | 80+ papers |
| **Multimodal SemCom** | 🟡 Growing | U-DeepSC, Cross-modal VQA SemCom | 20–30 papers |
| **XAI for general DL** | 🟢 Mature | Grad-CAM, SHAP, LIME, attention analysis | 1000+ papers |
| **XAI for SemCom** | 🔴 Nascent | Zia et al. survey (2026), Liu et al. (2024), XAI-SCS | <10 papers |
| **XAI + Multimodal SemCom** | 🔴 **EMPTY** | No system-level implementation exists | **0 papers** |
| **XAI-guided bandwidth allocation** | 🟠 Conceptual | CIBA framework (theoretical), no multimodal implementation | 2–3 papers |

### 3.2 Detailed Gap Identification

#### Gap 1: No Explainability Analysis of Multimodal Semantic Encoders [CRITICAL — PRIMARY CONTRIBUTION]

**What exists:** Grad-CAM and SHAP have been applied to image classifiers and NLP models extensively. A handful of papers (Liu et al. 2024, XAI-SCS) have applied XAI to **single-modality** text-only semantic communication.

**What's missing:** Nobody has applied *any* XAI technique to a **multimodal** semantic communication system. The question "what does a multimodal semantic encoder learn about cross-modal relationships when trained end-to-end with a channel layer?" is completely unanswered.

**Why it matters:**
- Without understanding what the encoder preserves, we cannot debug, improve, or trust these systems
- Regulators (EU AI Act) increasingly require explainability for autonomous AI systems
- 6G standardization efforts need interpretable systems for certification

**Our contribution:** First systematic XAI analysis of a multimodal SemCom system, with cross-modal importance decomposition across channel conditions.

---

#### Gap 2: Cross-Modal Importance Dynamics Under Channel Degradation [NOVEL — UNIQUE ANGLE]

**What exists:** Some papers study how image quality degrades with SNR in single-modal DeepJSCC. Research on cross-modal attention exists in NLP/CV but not in the context of wireless channels.

**What's missing:** No work has measured how a multimodal system dynamically rebalances the importance it assigns to text vs. image features as channel quality changes. Does the system learn to protect text (more compact, error-resistant) at the expense of image (more bandwidth-hungry, noise-sensitive) at low SNR? Is this behavior learned automatically or does it need to be explicitly designed?

**Why it matters:**
- Understanding this behavior is essential for designing robust 6G multimodal systems
- It could reveal whether current systems have implicit failure modes (e.g., always sacrificing the same modality)
- It provides design guidance: should we protect modalities equally or asymmetrically?

**Our contribution:** Quantitative measurement of modality importance shift across SNR using SHAP values and attention weights, with the first "cross-modal importance curves" showing how text vs. image importance varies with channel quality.

---

#### Gap 3: Explainability-Driven Bandwidth Allocation [ACTIONABLE — SYSTEM CONTRIBUTION]

**What exists:** The concept of importance-aware resource allocation exists (CIBA framework, Shapley-value-based allocation). Dynamic bit allocation for non-uniform quantization has been explored. However, these are:
- Theoretical or single-modality
- Not using standard XAI tools (Grad-CAM, SHAP) as the importance oracle
- Not evaluated on multimodal inputs

**What's missing:** A concrete system that uses XAI importance scores from a multimodal encoder to decide how many channel uses (bandwidth) to allocate to each feature, with measured bandwidth savings.

**Why it matters:**
- Directly translates interpretability into engineering utility (not just "nice visualizations" but actual bandwidth reduction)
- Bridges the gap between the XAI community ("we can explain models") and the communications community ("we need efficient transmission")

**Our contribution:** An importance-aware feature pruning module that drops the bottom-K% of features by SHAP/attention importance score, with Pareto curves showing the trade-off between bandwidth savings and semantic quality.

---

#### Gap 4: Diagnostic Failure Analysis for Semantic Communication [PRACTICAL — ENGINEERING CONTRIBUTION]

**What exists:** Traditional communication systems diagnose errors via BER curves and constellation diagrams. Semantic communication papers report BLEU/PSNR curves but provide no per-sample failure diagnosis.

**What's missing:** A diagnostic tool that can answer: "This specific image was poorly reconstructed at 5 dB SNR — was it because the encoder failed to extract the object boundary (encoding failure) or because the channel corrupted the object feature in transit (channel corruption failure)?"

**Why it matters:**
- Essential for productionizing semantic communication systems
- Enables targeted hardening (if encoding failures dominate, improve the encoder; if channel failures dominate, add more protection)

**Our contribution:** Attribution-based failure decomposition — for each failed reconstruction, attribute the error to specific encoder layers vs. channel noise using integrated gradients.

---

### 3.3 Positioning Against Closest Works

| Closest Work | What They Do | What We Do Differently |
|---|---|---|
| **DeepSC (Xie 2021)** | Text-only SemCom with Transformer | We add image modality + XAI analysis |
| **DeepJSCC (Bourtsoulatze 2019)** | Image-only SemCom with CNN | We add text modality + XAI analysis |
| **U-DeepSC (Zhang et al. 2023)** | Unified multimodal SemCom | We add XAI framework + importance-aware pruning. U-DeepSC is a black box |
| **Liu et al. (2024)** | XAI for text-only SemCom | We extend to multimodal + add cross-modal importance analysis |
| **Zia et al. (2026)** | Survey of XAI for SemCom | We provide the first *implementation*, not just a survey |
| **CIBA (2025)** | Attribution-based importance for single-modal | We extend to multimodal + implement actual bandwidth savings |

---

## 4. Research Questions

This project addresses five concrete research questions:

| ID | Research Question | Method |
|:---:|---|---|
| **RQ1** | What semantic features do multimodal encoders learn to preserve across different channel conditions? | Grad-CAM heatmaps (image), attention rollout (text), linear probing |
| **RQ2** | How does the relative importance of text vs. image modality shift as channel SNR decreases? | SHAP-based cross-modal importance decomposition |
| **RQ3** | Can XAI-derived importance scores predict which transmissions will fail before decoding? | Correlation analysis: importance scores vs. reconstruction error |
| **RQ4** | Can importance-aware feature pruning reduce bandwidth without proportional quality loss? | Pareto analysis: bandwidth reduction vs. semantic quality |
| **RQ5** | What are the dominant failure modes in multimodal SemCom — encoding failures or channel corruption failures? | Integrated gradients attribution to encoder vs. channel |

---

## 5. Proposed System Architecture

### 5.1 High-Level Architecture Diagram

```
═══════════════════════════════════════════════════════════════════════════════════
                           XMA-SemCom SYSTEM ARCHITECTURE
═══════════════════════════════════════════════════════════════════════════════════

TRANSMITTER                           CHANNEL                          RECEIVER
─────────────────────────────────   ─────────────   ─────────────────────────────

                                                    
 ┌───────────┐   ┌────────────┐                      ┌────────────┐   ┌──────────┐
 │   Input   │   │  Semantic   │                      │  Semantic   │   │  Output  │
 │   Image   │──▶│  Encoder   │──┐                 ┌──│  Decoder   │──▶│  Image   │
 │  (H×W×3)  │   │  (CNN)     │  │                 │  │  (CNN†)    │   │  (H×W×3) │
 └───────────┘   └────────────┘  │                 │  └────────────┘   └──────────┘
                       │         │                 │        │
                       │ Grad-CAM│                 │        │ Grad-CAM
                       ▼         │                 │        ▼
                  ┌─────────┐    │                 │   ┌─────────┐
                  │ XAI     │    │                 │   │ XAI     │
                  │ Module  │    │                 │   │ Module  │
                  └─────────┘    │                 │   └─────────┘
                                 │                 │
                          ┌──────▼──────┐   ┌──────▼──────┐
                          │ Cross-Modal │   │ Cross-Modal │
                          │   Fusion    │   │   Splitter  │
                          │  & Channel  │   │  & Channel  │
                          │   Encoder   │   │   Decoder   │
                          └──────┬──────┘   └──────▲──────┘
                                 │                 │
                          ┌──────▼──────┐   ┌──────┴──────┐
                          │ Importance- │   │             │
                          │   Aware     │   │  Received   │
                          │  Pruner     │   │  Symbols    │
                          │ (Optional)  │   │  ẑ ∈ ℂᵏ    │
                          └──────┬──────┘   └──────▲──────┘
                                 │                 │
                          ┌──────▼──────┐          │
                          │  Channel    │   ┌──────┴──────┐
                          │  Symbols    │──▶│   WIRELESS  │
                          │  z ∈ ℂᵏ    │   │   CHANNEL   │
                          └─────────────┘   │ (AWGN /     │
                                            │  Rayleigh)  │
 ┌───────────┐   ┌────────────┐             └─────────────┘
 │   Input   │   │  Semantic   │                      
 │   Text    │──▶│  Encoder   │──┘                 ┌──│  Semantic   │──▶│  Output  │
 │  (tokens) │   │(Transformer)│                      │  Decoder   │   │  Text    │
 └───────────┘   └────────────┘                      │(Transformer†│   │ (tokens) │
                       │                              └────────────┘   └──────────┘
                       │ Attention                          │
                       │ + SHAP                             │ Attention
                       ▼                                    ▼
                  ┌─────────┐                          ┌─────────┐
                  │ XAI     │                          │ XAI     │
                  │ Module  │                          │ Module  │
                  └─────────┘                          └─────────┘

═══════════════════════════════════════════════════════════════════════════════════
```

### 5.2 Module-Level Architecture

The system has **6 core modules** and **3 XAI modules**:

#### Core Modules

| Module | Input | Output | Architecture | Trainable |
|--------|-------|--------|-------------|:---------:|
| **M1: Image Semantic Encoder** | Image x_img ∈ ℝ^(H×W×3) | Image features f_img ∈ ℝ^d | CNN (ResNet-18 backbone, last FC removed) | ✅ |
| **M2: Text Semantic Encoder** | Token sequence x_txt ∈ ℤ^L | Text features f_txt ∈ ℝ^d | Transformer (4 layers, 4 heads, d=256) | ✅ |
| **M3: Cross-Modal Fusion** | f_img, f_txt | Fused features f_fused ∈ ℝ^(2d) | Cross-attention layer + concatenation | ✅ |
| **M4: Joint Channel Encoder** | f_fused | Channel symbols z ∈ ℂ^k | Dense layers + power normalization | ✅ |
| **M5: Channel Layer** | z | ẑ (noisy symbols) | AWGN or Rayleigh (non-trainable) | ❌ |
| **M6: Joint Channel Decoder + Modal Splitter + Semantic Decoders** | ẑ | Reconstructed image x̂_img, text x̂_txt | Dense → Split → CNN†, Transformer† | ✅ |

#### XAI Modules (Non-Trainable, Analysis Only)

| Module | Applied To | Technique | Output |
|--------|-----------|-----------|--------|
| **X1: Image Explainer** | M1 (Image Encoder) | Grad-CAM on last conv layer | Spatial importance heatmap (H×W) |
| **X2: Text Explainer** | M2 (Text Encoder) | Attention rollout + token-level SHAP | Per-token importance scores |
| **X3: Cross-Modal Importance Scorer** | M3 (Fusion) + M4 (Channel Encoder) | SHAP on fused feature vector | Per-feature importance + modality-level importance ratio (text % vs. image %) |

---

## 6. Technical Deep Dive: Component Design

### 6.1 Image Semantic Encoder (M1)

```python
class ImageSemanticEncoder(nn.Module):
    """
    CNN-based encoder that maps images to semantic feature vectors.
    Uses ResNet-18 backbone (pre-trained, then fine-tuned end-to-end).
    
    Input:  x_img ∈ ℝ^(B × 3 × 32 × 32)   [CIFAR-10]
            x_img ∈ ℝ^(B × 3 × 224 × 224)  [Flickr30k]
    Output: f_img ∈ ℝ^(B × d)               [d = 256]
    """
    # Architecture:
    #   ResNet-18 (conv1 → layer4) → AdaptiveAvgPool → FC(512 → 256)
    #
    # Grad-CAM hook: Registered on layer4[-1].conv2
    # This gives spatial importance maps showing WHICH image regions
    # the encoder considers semantically important for transmission.
```

**Design rationale:** ResNet-18 is chosen over heavier architectures (ResNet-50, ViT) for:
- Feasibility on single GPU
- Sufficient capacity for CIFAR-10/Flickr30k
- Well-studied Grad-CAM behavior (clear, interpretable heatmaps)
- Can be upgraded to ViT in future work

### 6.2 Text Semantic Encoder (M2)

```python
class TextSemanticEncoder(nn.Module):
    """
    Transformer-based encoder that maps token sequences to semantic vectors.
    Lightweight variant (4 layers, 4 heads) designed for edge feasibility.
    
    Input:  x_txt ∈ ℤ^(B × L)               [L = max sequence length]
    Output: f_txt ∈ ℝ^(B × d)               [d = 256, CLS token output]
    """
    # Architecture:
    #   Embedding(vocab_size, 256) → PositionalEncoding
    #   → 4× TransformerEncoderLayer(d_model=256, nhead=4, dim_ff=512)
    #   → Extract [CLS] token → FC(256 → 256)
    #
    # XAI hooks:
    #   1. Attention rollout: Multiply attention matrices across layers
    #      to get per-token importance w.r.t. [CLS] output
    #   2. SHAP: KernelSHAP on the encoder output w.r.t. input tokens
```

**Design rationale:** Custom Transformer (not pre-trained BERT) because:
- We want to train the encoder end-to-end *with the channel layer*
- Pre-trained BERT has never "seen" channel noise; fine-tuning doesn't deeply adapt it
- Smaller model = faster training, easier XAI analysis, more interpretable attention patterns

### 6.3 Cross-Modal Fusion Module (M3)

```python
class CrossModalFusion(nn.Module):
    """
    Aligns and fuses image and text features using cross-attention.
    
    Input:  f_img ∈ ℝ^(B × d), f_txt ∈ ℝ^(B × d)
    Output: f_fused ∈ ℝ^(B × 2d)
    """
    # Architecture:
    #   1. Cross-Attention: f_img attends to f_txt (and vice versa)
    #      img_attended = CrossAttention(Q=f_img, K=f_txt, V=f_txt)
    #      txt_attended = CrossAttention(Q=f_txt, K=f_img, V=f_img)
    #   2. Concatenation: f_fused = [img_attended || txt_attended]
    #   3. LayerNorm + FC(2d → 2d)
    #
    # XAI: The cross-attention weights directly reveal
    # how much each modality "relies on" the other.
    # High img→txt attention = image encoder needs text context.
```

**Design rationale:** Cross-attention (not simple concatenation) because:
- It models inter-modal dependencies explicitly
- Attention weights are directly interpretable
- It's the standard approach in multimodal AI (CLIP, LLaVA use similar mechanisms)

### 6.4 Joint Channel Encoder (M4) and Channel Layer (M5)

```python
class JointChannelEncoder(nn.Module):
    """
    Maps fused semantic features to complex-valued channel symbols.
    
    Input:  f_fused ∈ ℝ^(B × 2d)
    Output: z ∈ ℂ^(B × k)    [k = number of channel uses]
    """
    # Architecture:
    #   FC(2d → 512) → ReLU → FC(512 → 2k) → Reshape to ℂ^k
    #   → PowerNormalization (enforce E[|z|²] = 1)
    #
    # k (bandwidth) is a hyperparameter:
    #   k/n ratio determines compression level
    #   k = 64 for CIFAR-10, k = 256 for Flickr30k

class AWGNChannel(nn.Module):
    """Non-trainable. Adds complex Gaussian noise."""
    # ẑ = z + n,  where n ~ CN(0, σ²I),  σ² = 1/(2 × 10^(SNR_dB/10))

class RayleighChannel(nn.Module):
    """Non-trainable. Applies Rayleigh fading + AWGN."""
    # ẑ = h ⊙ z + n,  where h ~ CN(0, I),  n ~ CN(0, σ²I)
    # Assumes perfect CSI at receiver (standard assumption)
```

### 6.5 Importance-Aware Pruner (Novel Module)

```python
class ImportanceAwarePruner(nn.Module):
    """
    Uses XAI importance scores to prune low-importance features
    before channel transmission, reducing effective bandwidth.
    
    Input:  f_fused ∈ ℝ^(B × 2d), importance_scores ∈ ℝ^(B × 2d)
    Output: f_pruned ∈ ℝ^(B × 2d')  where d' ≤ 2d
    """
    # Algorithm:
    #   1. Compute importance scores via SHAP or gradient-based method
    #   2. Rank features by importance
    #   3. Keep top-(1-p)% features, zero out bottom-p% features
    #   4. Transmit only non-zero features (reduced bandwidth)
    #
    # Pruning ratio p ∈ {0%, 10%, 20%, 30%, 40%, 50%}
    # Evaluated on the Pareto front: bandwidth savings vs. quality
```

---

## 7. Explainability Framework

### 7.1 XAI Techniques Used

#### 7.1.1 Grad-CAM for Image Encoder

**What it reveals:** Which spatial regions of the input image the encoder considers semantically important for transmission.

**Implementation:**
```
1. Forward pass: x_img → f_img (record activations at last conv layer A^l)
2. Compute gradients: ∂f_img/∂A^l (how does each spatial location affect the output)
3. Global average pool gradients to get importance weights α_k
4. Generate heatmap: L_GradCAM = ReLU(Σ_k α_k · A^l_k)
5. Overlay on original image for visualization
```

**What we analyze:**
- Do heatmaps focus on objects (desirable) or textures (fragile)?
- How do heatmaps change as SNR decreases? (Does the encoder "give up" on fine details and focus on gross structure?)
- Are heatmaps consistent across different channel realizations (Rayleigh fading)?

#### 7.1.2 Attention Rollout for Text Encoder

**What it reveals:** Which input tokens the Transformer considers most important for the semantic output.

**Implementation:**
```
1. Extract attention matrices from all 4 Transformer layers: {A¹, A², A³, A⁴}
2. Apply attention rollout: A_rolled = A⁴ · A³ · A² · A¹
3. Extract row corresponding to [CLS] token: importance(token_i) = A_rolled[CLS, i]
4. Normalize to get per-token importance distribution
```

**What we analyze:**
- Does the encoder learn to focus on content words (nouns, verbs) over function words (the, a, of)?
- At low SNR, does it shift focus to fewer, more critical tokens?
- How does the presence of a paired image change which tokens are deemed important?

#### 7.1.3 SHAP for Cross-Modal Importance

**What it reveals:** The quantitative contribution of each feature (and each modality) to the final reconstruction quality.

**Implementation:**
```
1. Define prediction function: f(features) = reconstruction_quality(decode(channel(encode(features))))
2. Apply KernelSHAP with background dataset of 100 reference samples
3. For each test sample, compute SHAP values φ_i for each of the 2d features
4. Aggregate: 
   - Image importance = Σ_{i ∈ image_features} |φ_i|
   - Text importance = Σ_{i ∈ text_features} |φ_i|
   - Modality ratio = Image_importance / (Image_importance + Text_importance)
```

**What we analyze:**
- How does the modality ratio change across SNR? (Our primary novel result)
- Which individual features have the highest SHAP values? Can we interpret them?
- Does feature importance correlate with reconstruction error? (validation of XAI)

### 7.2 Novel Analysis: Cross-Modal Importance Curves

The centerpiece of our XAI analysis is the **Cross-Modal Importance Curve** — a plot that has never been produced in the literature:

```
Modality
Importance (%)
    ▲
100 │                    Image Importance
    │   ●───●───●───●───●
 80 │
    │                          ← High SNR: Image dominates
 60 │
    │        Crossover Point ─────┐
 40 │                             │
    │                             ▼
 20 │   ●───●───●───●───●
    │                    Text Importance
  0 │─────────────────────────────────▶ SNR (dB)
    0    5    10   15   20
```

**Hypothesis:** At low SNR, the system will shift importance toward text features (more compact, more robust to noise) and away from image features (high-dimensional, noise-sensitive). This curve will reveal the "crossover point" where the dominant modality switches.

---

## 8. Datasets and Experimental Setup

### 8.1 Datasets

| Dataset | Modality | Size | Use | Why This Dataset |
|---------|----------|------|-----|------------------|
| **Flickr30k** | Image + Text (captions) | 31K images, 5 captions each | Primary multimodal dataset | Standard image-captioning benchmark; provides natural image-text pairs |
| **CIFAR-10** | Image only | 60K images (32×32) | Image-only baseline and rapid prototyping | Standard in DeepJSCC literature; enables direct comparison |
| **Europarl** | Text only | ~2M sentences | Text-only baseline | Standard in DeepSC literature; enables direct comparison |
| **COCO Captions** (optional) | Image + Text | 330K images, 5 captions each | Extended evaluation | Larger scale, more diverse |

### 8.2 Data Splits

| Split | Flickr30k | CIFAR-10 | Europarl |
|-------|:---------:|:--------:|:--------:|
| Train | 25,000 images | 45,000 images | 1.5M sentences |
| Validation | 3,000 images | 5,000 images | 200K sentences |
| Test | 3,000 images | 10,000 images | 300K sentences |

### 8.3 Channel Configurations

All experiments are run on a **systematic grid** of channel conditions:

| Parameter | Values |
|-----------|--------|
| **Channel type** | AWGN, Rayleigh fading |
| **SNR (dB)** | {0, 5, 10, 15, 20} |
| **Bandwidth ratio (k/n)** | {1/6, 1/3, 1/2} |
| **Total experiment grid** | 2 channels × 5 SNRs × 3 bandwidths = **30 configurations** |

### 8.4 Training Configuration

| Parameter | Value |
|-----------|-------|
| **Optimizer** | AdamW (lr=1e-4, weight_decay=1e-5) |
| **Scheduler** | CosineAnnealingLR (T_max=100) |
| **Batch size** | 64 (CIFAR-10), 32 (Flickr30k) |
| **Epochs** | 100 (CIFAR-10), 50 (Flickr30k) |
| **Loss function** | λ₁·MSE(image) + λ₂·CrossEntropy(text) + λ₃·(1-CosineSim(fused)) |
| **Loss weights** | λ₁=1.0, λ₂=1.0, λ₃=0.5 (ablated) |
| **SNR during training** | Sampled uniformly from [0, 20] dB per batch |
| **Hardware** | Single GPU (NVIDIA RTX 3060 12GB or Colab T4/A100) |

---

## 9. Evaluation Metrics and Protocol

### 9.1 Semantic Quality Metrics

| Metric | Modality | What It Measures |
|--------|----------|-----------------|
| **PSNR (dB)** | Image | Pixel-level reconstruction fidelity |
| **SSIM** | Image | Structural similarity (luminance, contrast, structure) |
| **LPIPS** | Image | Perceptual similarity (learned, correlates with human judgment) |
| **BLEU-1/2/4** | Text | N-gram overlap between original and reconstructed text |
| **Sentence Similarity** | Text | Cosine similarity of sentence embeddings (SBERT) |
| **CIDEr** | Multimodal | Caption quality metric (for Flickr30k) |

### 9.2 XAI-Specific Metrics

| Metric | What It Measures |
|--------|-----------------|
| **Modality Importance Ratio (MIR)** | % of total SHAP importance assigned to image vs. text |
| **Importance-Error Correlation (IEC)** | Pearson/Spearman correlation between XAI importance scores and reconstruction error |
| **Explanation Faithfulness (EF)** | Drop in quality when top-K important features are removed vs. random K features |
| **Attention Entropy** | Entropy of attention distribution — low = focused on few tokens/regions, high = diffuse |

### 9.3 Efficiency Metrics

| Metric | What It Measures |
|--------|-----------------|
| **Channel Uses (k)** | Number of complex symbols transmitted |
| **Bandwidth Savings (%)** | Reduction in k when using importance-aware pruning |
| **Quality-at-Budget** | PSNR/BLEU at a given bandwidth budget with and without pruning |
| **Model Parameters** | Total trainable parameters (target: <10M for Phase 1) |
| **Inference Latency** | End-to-end encoding + decoding time on GPU/CPU |

### 9.4 Evaluation Protocol

For every configuration in the 30-point grid, we report:

```
For each (channel_type, SNR, bandwidth_ratio):
    1. Run 1000 test samples through the trained model
    2. Compute average PSNR, SSIM, BLEU, Sentence Similarity
    3. For 100 representative samples:
        a. Generate Grad-CAM heatmaps (image)
        b. Generate attention rollout maps (text)
        c. Compute SHAP importance scores (cross-modal)
    4. Compute MIR, IEC, EF, Attention Entropy
    5. Run importance-aware pruning at p ∈ {0%, 10%, 20%, 30%, 40%, 50%}
    6. Compute bandwidth savings vs. quality trade-off (Pareto curve)
```

---

## 10. Implementation Roadmap

### Phase 1: Core System (12 Weeks)

| Week | Milestone | Deliverable |
|:----:|-----------|-------------|
| **1** | Project setup + literature code review | Environment setup, clone DeepJSCC + DeepSC repos, verify they run |
| **2** | Implement image semantic encoder (M1) + channel layer (M5) | Working DeepJSCC baseline on CIFAR-10, matching published PSNR |
| **3** | Implement text semantic encoder (M2) + channel layer | Working DeepSC-lite baseline on Europarl, matching published BLEU |
| **4** | Implement cross-modal fusion (M3) + joint channel codec (M4, M6) | End-to-end multimodal system training on Flickr30k |
| **5** | Train and tune the full multimodal system | Converged model with competitive PSNR + BLEU across SNR grid |
| **6** | Implement Grad-CAM (X1) + attention rollout (X2) | Visualizations of what each encoder learns; qualitative analysis |
| **7** | Implement SHAP cross-modal importance (X3) | Modality Importance Ratio curves across SNR; primary novel result |
| **8** | Implement importance-aware pruner + evaluate bandwidth savings | Pareto curves: bandwidth vs. quality |
| **9** | Run full evaluation protocol (30 configurations × all metrics) | Complete results tables and figures |
| **10** | Failure analysis: attribution-based error decomposition | Diagnostic case studies showing encoding vs. channel failures |
| **11** | Paper writing: introduction, related work, methodology | Draft paper (IEEE format, 6–10 pages) |
| **12** | Paper writing: experiments, results, conclusion + submission | Final paper, supplementary materials, code cleanup |

### Key Risk Mitigations

| Risk | Mitigation |
|------|-----------|
| Multimodal training doesn't converge | Fall back to training single-modal systems separately and fusing post-hoc |
| SHAP is too slow on full model | Use FastSHAP approximation or gradient-based importance instead |
| GPU memory insufficient for Flickr30k (224×224) | Resize to 128×128 or use CIFAR-10 as primary dataset |
| XAI results are not interpretable | Add linear probing experiments as a simpler baseline |

---

## 11. Expected Contributions and Novelty Claims

### Primary Contributions (For Paper)

1. **First explainable multimodal semantic communication system** — XMA-SemCom is the first system to integrate Grad-CAM, attention analysis, and SHAP into a multimodal (text + image) JSCC pipeline.

2. **Cross-Modal Importance Curves** — First quantitative measurement of how text vs. image importance shifts under varying channel conditions, answering the question: "Which modality does the system learn to protect?"

3. **Importance-aware adaptive transmission** — Demonstration that XAI-derived importance scores can be used to prune low-importance features, achieving 20–40% bandwidth savings with minimal quality loss.

4. **Failure diagnosis framework** — Attribution-based decomposition of reconstruction errors into encoder failures vs. channel corruption failures, providing actionable engineering insights.

### Secondary Contributions

5. **Open-source codebase** — Fully reproducible PyTorch implementation with evaluation scripts.

6. **Benchmark results** — Multimodal SemCom baseline numbers on Flickr30k under standardized channel conditions (no such benchmark exists).

---

## 12. Phase 2: Audio and Video Extension

### 12.1 Audio Integration (Phase 2a — +4 weeks)

| Component | Design |
|-----------|--------|
| **Audio Encoder** | Pre-trained neural audio codec (EnCodec) → channel-adaptive dense layers |
| **Fusion** | Extend cross-modal attention to 3-way: image ↔ text ↔ audio |
| **XAI** | Mel-spectrogram Grad-CAM for audio; SHAP across 3 modalities |
| **Dataset** | AudioCaps (audio + caption), Spoken COCO |
| **New research question** | How does audio importance compare to text/image? Does adding audio improve robustness to image noise? |

### 12.2 Video Integration (Phase 2b — +8 weeks)

| Component | Design |
|-----------|--------|
| **Video Encoder** | 3D-CNN (R3D-18) or Video Swin Transformer for spatiotemporal features |
| **Temporal fusion** | Extend cross-attention with temporal dimension; keyframe selection |
| **XAI** | Temporal Grad-CAM (which frames matter?); spatiotemporal importance maps |
| **Dataset** | MSR-VTT (video + caption), ActivityNet Captions |
| **New research question** | How does temporal redundancy affect semantic importance? Can we skip "unimportant" frames? |

### 12.3 Phase 2 Architecture Extension

```
Phase 1 (Current):    Image ──┐                    ┌── Image
                              ├── Fusion → Channel ──┤
                      Text  ──┘                    └── Text

Phase 2a (+Audio):    Image ──┐                    ┌── Image
                      Text  ──┼── Fusion → Channel ──┼── Text
                      Audio ──┘                    └── Audio

Phase 2b (+Video):    Image ──┐                    ┌── Image
                      Text  ──┤                    ├── Text
                      Audio ──┼── Fusion → Channel ──┼── Audio
                      Video ──┘                    └── Video
```

---

## 13. Publication Strategy

### 13.1 Target Venues (In Order of Priority)

| Priority | Venue | Type | Length | Deadline (Typical) | Why |
|:--------:|-------|------|--------|:-------------------:|-----|
| 1 | **arXiv** (cs.IT + cs.LG) | Preprint | Any | Immediate | Establish priority, get visibility |
| 2 | **IEEE GLOBECOM Workshop** ("Semantic Comm. Meets Wireless in AI Era") | Workshop | 6 pages | Apr–May | Directly relevant workshop; high acceptance for novel angles |
| 3 | **IEEE ICC Workshop** ("Beyond Bits: Goal-Oriented SemCom") | Workshop | 6 pages | Oct–Nov | Alternative flagship workshop |
| 4 | **IEEE Wireless Comm. Letters (WCL)** | Letter | 4 pages | Rolling | Fast review (1–3 months); good for XAI importance curve result |
| 5 | **IEEE Open Journal of ComSoc (OJ-COMS)** | Journal | 10–15 pages | Rolling | Open access; frequently runs special sections on SemCom |

### 13.2 Paper Framing Options

| Framing | Best Venue | Core Narrative |
|---------|-----------|----------------|
| **"First Explainable Multimodal SemCom System"** | GLOBECOM/ICC Workshop | System contribution + XAI analysis |
| **"What Do Multimodal Semantic Encoders Learn?"** | IEEE WCL | Analytical contribution (importance curves) |
| **"XAI-Guided Bandwidth Allocation for Multimodal SemCom"** | IEEE OJ-COMS | Engineering contribution (bandwidth savings) |

### 13.3 Suggested Timeline

```
Month 1-3:  Build system + run experiments
Month 3:    Submit to arXiv
Month 3-4:  Submit to IEEE GLOBECOM/ICC Workshop
Month 4-5:  Revise based on reviews; submit extended version to IEEE WCL/OJ-COMS
Month 6+:   Phase 2 (audio/video) for follow-up paper
```

---

## 14. References

### Directly Referenced

1. Zia et al., "A Survey on Explainable AI for Semantic Communication: Architecture, Challenges, and Future Opportunities," *Computer Networks*, Feb. 2026.
2. H. Xie, Z. Qin et al., "Deep Learning Enabled Semantic Communication Systems," *IEEE TSP*, 2021.
3. E. Bourtsoulatze, D. B. Kurka, D. Gündüz, "Deep Joint Source-Channel Coding for Wireless Image Transmission," *IEEE TCCN*, 2019.
4. Zhang et al., "A Unified Multi-Task Semantic Communication System for Multimodal Data (U-DeepSC)," *IEEE Trans. Comm.*, 2023.
5. Liu et al., "Explainable Semantic Communication for Text Tasks," Aug. 2024.
6. R. R. Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks," *ICCV*, 2017.
7. S. Lundberg, S.-I. Lee, "A Unified Approach to Interpreting Model Predictions (SHAP)," *NeurIPS*, 2017.
8. Mortaheb et al., "Cross-Modal Attention for Goal-Driven Semantic Communication," *arXiv*, Dec. 2025.
9. Du et al., "Task-Oriented LMM-based Semantic Communication for Vehicle Networks," *arXiv*, May 2025.

### Code Repositories

10. DeepJSCC PyTorch: [github.com/chunbaobao/Deep-JSCC-PyTorch](https://github.com/chunbaobao/Deep-JSCC-PyTorch)
11. U-DeepSC PyTorch: [github.com/zhang-guangyi/t-udeepsc](https://github.com/zhang-guangyi/t-udeepsc)
12. IPC Lab (Gündüz): [github.com/ipc-lab](https://github.com/ipc-lab)

---

> **Note:** This document is a living research proposal. Architecture details, hyperparameters, and evaluation protocols will be refined as implementation proceeds. All datasets are publicly available and all experiments are software-only (no hardware required).

---

*Document prepared: July 31, 2026*
