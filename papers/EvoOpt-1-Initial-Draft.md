# EvoOpt-1: Autonomously Evolved Gradient-Based Optimizers

**Nova Aether Research PBC**  
**Draft v3.1 – March 20, 2026**  
**Internal Cycle Progress: Full analysis through Cycle 26 (26 logs)**

**License**: MIT (code, logs, and results fully open)

## Abstract

We introduce EvoOpt-1, a family of gradient-based optimizers autonomously discovered through tree-structured mutation and best-first selection, starting from vanilla SGD. Over 26 documented evolutionary cycles, the system repeatedly converges on sophisticated mechanisms—including adaptive momentum, bias-corrected second-moment estimation, gradient clipping via norm or value, dynamic per-layer scaling, and tanh-based stabilization—many of which mirror or improve upon hand-designed optimizers such as AdamW, Lion, and Sophia.

Empirical highlights from the evolved lineage include:
- 51× faster convergence to low-loss regimes on the Rosenbrock function versus baseline SGD.
- Final MSE = 0.008 on a toy MLP regression task (outperforming Adam).
- 98.71% test accuracy on MNIST with 43% fewer epochs to target performance than AdamW.
- On a 550k-parameter ConvNet trained on a 10k-sample CIFAR-10 subset: **82.1% ± 0.7%** accuracy, reached in ~55% fewer training steps than strong AdamW baselines (Cycle 26 result).

These gains arise from fully autonomous search, with no human tuning of the final variants beyond the initial SGD seed and mutation grammar. All 26 cycle logs, code, seeds, and tables are preserved publicly as corporate memory, demonstrating a transparent, reproducible path toward compute-efficient deep learning that aligns with our public-benefit charter to reduce energy and financial barriers in AI training.

## 1. Introduction

Gradient-based optimization remains the dominant paradigm for training deep neural networks, yet the space of effective update rules is still explored largely through human intuition and trial-and-error. Even incremental improvements in sample efficiency or wall-clock time translate to enormous reductions in compute, energy, and carbon emissions at scale.

Nova Aether Research operates as a Delaware Public Benefit Corporation with the explicit mission of accelerating scientific discovery through autonomous AI systems. The EvoOpt project is our first flagship effort: an agentic evolutionary loop that proposes, implements, benchmarks, and refines gradient update rules without human intervention beyond defining the high-level objective (minimize validation loss + maximize efficiency).

Starting from plain SGD, the system applies tree-structured mutations and selects elite variants using composite performance on progressively harder proxies (Rosenbrock → toy MLP → MNIST → CIFAR-10 subset). The entire history—26 cycles—is logged verbatim in `/logs`, making this one of the most transparent demonstrations of autonomous optimizer discovery to date.

## 2. Related Work

Modern optimizers build on foundational ideas:
- Momentum (Polyak, 1964; Sutskever et al., 2013)
- Adaptive per-parameter learning rates (AdaGrad, Duchi et al., 2011; RMSProp, Tieleman & Hinton, 2012; Adam, Kingma & Ba, 2015; AdamW, Loshchilov & Hutter, 2019)
- Second-order approximations and clipping (Sophia, Anil et al., 2022; Lion, Chen et al., 2023)

Automated discovery approaches include hyperparameter tuning (e.g., Optuna, Population-Based Training), neural architecture search, and meta-learning. Closest to our work are evolutionary methods for optimizer design (e.g., learned optimizers by Metz et al., 2019; gradient-free evolution of update rules). Unlike prior efforts, EvoOpt-1 runs a long-horizon, open-ended, best-first tree search with full public logging of every cycle, enabling post-hoc analysis of how high-performance features emerge organically.

## 3. Method

### 3.1 Evolutionary Framework

We represent each optimizer as a tree of differentiable operations applied to the raw gradient g_t:

- Base: θ_{t+1} = θ_t - η · g_t  (vanilla SGD)
- Mutations add nodes such as:
  - Momentum: m_t = β_1 m_{t-1} + (1-β_1) g_t
  - Second moment: v_t = β_2 v_{t-1} + (1-β_2) g_t²
  - Adaptive step: update = m_t / (√v_t + ε)
  - Clipping: g_t ← clip(g_t, value or norm threshold)
  - Per-layer scaling: independent β, ε, clip per module
  - Nonlinearities: tanh(·), sigmoid(grad_norm), learned gates

At each cycle:
1. Read the previous elite optimizer tree from the prior log.
2. Generate 8–16 mutated children via random subtree insertion/deletion/replacement.
3. Evaluate each child on the current benchmark ladder (multiple seeds ≥ 4).
4. Select the top variant by composite score: 0.6 × final accuracy/loss + 0.4 × (1 / epochs_to_target).
5. Log full results, code diff, observations → carry forward as parent.

### 3.2 Benchmarks & Evaluation Protocol

Progressive difficulty:
- Rosenbrock function (non-convex test)
- Toy MLP regression (1 hidden layer)
- MNIST (FC + small CNN)
- CIFAR-10 10k-sample subset (ResNet-style 550k-param ConvNet)

All runs use 4–5 independent seeds, fixed LR schedules or cosine decay, early stopping on validation plateau. Efficiency is measured as epochs (or steps) to reach 95% of best-seen validation performance.

## 4. Experiments & Results

### 4.1 Proxy Tasks (Cycles 1–15)

Early cycles rapidly rediscover momentum and basic adaptive scaling. By Cycle 12:
- Rosenbrock: best variant converges 51× faster than SGD.
- Toy regression: MSE ↓ to 0.008 (vs Adam ~0.012).

### 4.2 MNIST (Cycles 16–22)

Peak: **98.71%** test accuracy, convergence in ~43% fewer epochs than AdamW baseline.

### 4.3 CIFAR-10 Subset (Cycles 23–26)

Final ConvNet benchmark (full 10k samples, stronger augmentations):
- EvoOpt-26: **82.1% ± 0.7%** accuracy
- Reaches 81% in **~18–19 epochs** vs AdamW requiring ~40–42 epochs (≈55% efficiency gain)
- Consistent across 5 seeds; wall-clock savings scale similarly.

Cross-cycle trend shows monotonic gains in both peak performance and sample efficiency.

### 4.4 Emergent Features (from log analysis)

Most stable high-performers include:
- Bias-corrected second-moment scaling
- Norm-based gradient clipping (~0.3–1.0 threshold)
- Per-layer momentum decay rates
- Tanh stabilization on large gradients

These combinations appear novel relative to common hand-tuned rules.

## 5. Discussion

The autonomous loop reliably rediscovers core elements of modern optimizers while discovering per-layer and nonlinear refinements that provide measurable gains on vision tasks. The efficiency improvements directly support our public-benefit goal: even 40–55% fewer epochs at scale saves gigawatt-hours in training runs.

**Limitations**:
- Benchmarks still proxy-scale (no full CIFAR-10/ImageNet/LLM yet).
- Mutation grammar is currently hand-defined (future meta-evolution possible).
- Compute budget per cycle remains modest.

**Future work**:
- Scale to full datasets and larger models.
- Evolve learning-rate schedules and weight decay jointly.
- Integrate with architecture search.
- Open-source swarm infrastructure for community use.

## 6. Conclusion

EvoOpt-1 shows that transparent, logged, autonomous evolutionary search can produce competitive—and in efficiency terms superior—gradient optimizers. By releasing the full 26-cycle memory, we invite reproduction, extension, and critique toward more sustainable AI training methods.

## Appendix: Reproducibility

- EvoOpt-26 PyTorch class (latest elite implementation): [to be pasted from Cycle 26 log]
- Random seeds: listed per cycle in /logs
- Full tables: aggregated from logs 1–26
- All original logs: https://github.com/Nova-Aether-Research/memory/tree/main/logs

We gratefully acknowledge the foundational work of Sakana AI’s AI-Scientist (forked and extended here) and the open-source community that makes such autonomous research feasible.
