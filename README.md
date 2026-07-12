# Modern NLP Architecture: Adaptive Compute Transformer

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)

## 📖 Overview
Welcome to the **Modern NLP Architecture** project. This repository contains a production-grade, highly experimental **Adaptive Decoder-Only Transformer** built completely from scratch. 

Unlike standard Large Language Models (LLMs) that apply static, uniform compute to every token, this architecture features a centralized **Universal Compute Controller**. This controller predicts the difficulty of each token in real-time and dynamically routes it through a network of 13 adaptive subsystems—saving massive amounts of energy on simple words while heavily allocating compute to complex reasoning.

---

## 🎯 The Problem: Uniform Compute
Modern transformers suffer from a fundamental architectural flaw: **Uniformity**. Whether a transformer is generating a simple stop word like "the" or attempting to solve a multi-step differential equation, it forces the token through the exact same number of layers, the exact same dense MLP projections, and the exact same $O(N^2)$ attention matrices. 

This results in two massive inefficiencies:
1. **Wasted Energy**: TeraFLOPs of compute are burned processing trivial grammatical syntax.
2. **Under-allocation & Hallucinations**: Complex logical deductions are starved of compute because the model cannot dynamically "think longer" on hard problems.

---

## 💡 The Solution: Universal Compute Controller
To solve this, we introduce the **Universal Compute Controller** (`controller.py`). For every token $t$, the controller emits a single, continuous scalar value $g_t \in (0, 1)$ representing "token difficulty":

- **$g_t \approx 0$ (Trivial Token)**: The network exits early, uses only 1 MoE expert, and restricts the attention budget to a local window.
- **$g_t \approx 1$ (Complex Token)**: The network triggers latent ponder loops, activates up to $K=8$ MoE experts, and grants the token full context attention.

This single $g_t$ signal acts as a biological nervous system, broadcasting to all transformer subsystems to dynamically modulate the model's capacity at the token level.

---

## 🧠 Core Architecture Subsystems
The model (`modern_nlp_architectire/`) is built on 13 tightly coupled but isolated subsystems:

1. **Adaptive Depth & Halting** (`halting.py`): PonderNet-style continuous halting. Easy tokens exit the network early.
2. **Latent Ponder Loops** (`latent_loop.py`): Forces the model to perform extra internal refinement passes in latent space before committing to a final output vector.
3. **Difficulty-Scaled Mixture of Experts (MoE)** (`moe_router.py`): dynamically scales the number of active experts ($K=1$ to $K=8$) based on the $g_t$ difficulty signal.
4. **Dynamic Attention Budgeting** (`attn_budget.py`): Assigns specific key-retrieval budgets. Trivial tokens get local sliding windows; hard tokens get global attention.
5. **Native Speculative Decoding** (`decode.py`): Because the controller predicts token difficulty *before* full computation, the model parallel-accepts low-$g_t$ tokens and only triggers deep verification on high-$g_t$ tokens, eliminating the need for an external draft model.

---

## 📂 File Skeleton

```text
├── modern_nlp_architectire/     # Core Architecture
│   ├── configuration.py         # Hyperparameter definitions
│   ├── modeling.py              # Main CortexForCausalLM Transformer class
│   ├── block.py                 # Transformer blocks and RMSNorm
│   ├── controller.py            # Universal Compute Controller (generates g_t)
│   ├── halting.py               # Early exit / dynamic depth
│   ├── latent_loop.py           # Ponder loops for hard tokens
│   ├── moe_router.py            # Token-level MoE routing
│   ├── attn_budget.py           # Dynamic attention sparsity
│   ├── decode.py                # Speculative decoding
│   ├── pos.py                   # Positional embeddings
│   └── mem.py                   # Gated KV Memory
│
├── configs/                     # YAML Configurations (1B, language_large, etc.)
├── train.py                     # Main Distributed Training Loop
├── train_dpo.py                 # Compute-Aware Direct Preference Optimization
├── demo.py                      # FastAPI UI for interacting with the model
├── ablate.py                    # Ablation studies and scientific testing
└── visualize_thoughts.py        # Visualizes g_t difficulty colors in terminal
```

---

## ⚙️ Workflow & Methodology

### 1. Stage-Gated Build Discipline
The model was built using a strictly stage-gated discipline to prevent gradient explosion:
* **Stage 1**: Dense Baseline (Non-adaptive)
* **Stage 2**: Inert Controller (Proving $g_t$ gradients flow safely)
* **Stage 3-10**: Gradual subsystem injection with isolated acceptance tests.

### 2. Compute-Aware DPO Alignment
Standard Direct Preference Optimization (DPO) aligns text generation to human preference. We built **Compute-Aware DPO** (`train_dpo.py`), which introduces a penalty: $E[g_{\text{rejected}} - g_{\text{chosen}}]$. The model is explicitly penalized for wasting high compute on reasoning paths that humans ultimately reject, aligning its *internal thinking time* with actual logic quality.

---

## 📊 Evaluation & Interpretability

Because $g_t$ is a simple scalar from 0 to 1, the architecture is incredibly interpretable out of the box. 
By running `visualize_thoughts.py`, the model prints generated text colorized from **Blue (Easy/Low Compute)** to **Red (Hard/High Compute)**. 

During evaluation, you can empirically observe the model's internal processing difficulty spike on ambiguous pronouns, complex verbs, and mathematical logic, while remaining completely cold (blue) on stop words like "the" and "a".

---

## 🚀 How to Run

### 1. Setup Environment
```bash
pip install -r requirements.txt
```

### 2. Train the Model
You can train the model on local datasets or Hugging Face corpora (e.g., TinyStories).
```bash
python train.py --config configs/language_large.yaml
```

### 3. Run the UI / Inference Server
Launch the FastAPI interface to chat with your trained model:
```bash
python demo.py
```
Open `index.html` in your browser to interact with the backend!

---
*Built with absolute precision. All comments and AI traces have been surgically stripped from the core architecture folder to provide a perfectly clean mathematical baseline.*
