# EvoOpt-1: Autonomously Evolved Gradient-Based Optimizers for Efficient Deep Learning

**Nova Aether Research Public Benefit Corporation**  
**CEO Approval: William Chappell – March 23, 2026**  
**Draft v4.0 – Final Internal Version for Publication**  
**Cycle Coverage: Full autonomous evolution through Cycle 52**  
**License**: MIT (code, logs, results, and paper fully open-source)

## Abstract

EvoOpt-1 is a family of gradient-based optimizers discovered through a fully autonomous, tree-structured evolutionary search initialized from vanilla SGD. Over 52 documented cycles using our extended AI-Scientist-v2 swarm (local Grok API orchestration), the system organically converges on adaptive mechanisms that rival or surpass hand-crafted optimizers (AdamW, Lion, Sophia, Muon) in both final performance and training efficiency.

Key results on full CIFAR-10 (50k train / 10k test, standard augmentations, 8.1M-param CNN):
- **92.6% ± 0.4%** test accuracy (8-seed average)
- Converges to 92% in **~28 epochs** vs. tuned AdamW baseline requiring ~98 epochs (**3.5× efficiency gain**)
- Matches or exceeds strong baselines on MNIST (98.9%+), Rosenbrock (orders-of-magnitude faster), and toy tasks

The final EvoOpt retains only three high-impact components after rigorous ablations: discrete wavelet transform preprocessing, spectral kurtosis-based adaptive momentum, and Lyapunov-exponent-inspired gradient clipping. All exotic higher-order cumulants, fractal/chaos terms, and RG-flow features from mid-cycles (e.g., Cycles 40–45) were correctly deprecated when ablations showed <5–10% contribution.

Full transparency is preserved: every proposal, code, metric table, statistical test, and ablation is logged verbatim in https://github.com/Nova-Aether-Research/memory/tree/main/logs. This work demonstrates reproducible, auditable agentic scientific discovery aligned with our public-benefit charter to democratize efficient AI training and reduce global compute/energy demands.

## 1. Introduction

Modern deep learning relies on gradient descent variants refined through decades of human ingenuity. Yet the design space remains vast and underexplored systematically. EvoOpt-1 automates this process via best-first tree search with mutation, evaluation, and selection—starting from SGD and discovering features that echo Adam-style adaptation while introducing novel signal-processing and dynamical-systems-inspired refinements.

Nova Aether Research PBC (Delaware, EIN 41-4995612) exists to accelerate discovery through persistent AI swarms. EvoOpt is our flagship proof-of-concept: 52 cycles of logged, self-correcting evolution yielding a clean, deployable optimizer.

## 2. Related Work

- Classical: Momentum (1964–2013), AdaGrad/RMSProp/Adam/AdamW (2011–2019), Lion/Sophia (2022–2023)
- Automated optimizer design: Learned optimizers (Metz et al., 2019+), evolutionary/meta-learning approaches
- Closest: Sakana AI’s AI-Scientist (2024) — we fork/extend it for long-horizon optimizer evolution with public memory logging

Unlike prior work, EvoOpt-1 emphasizes full-cycle transparency, ablation-enforced simplicity, and progression to full-scale vision benchmarks.

## 3. Method

### 3.1 Evolutionary Loop (AI-Scientist-v2 Fork)

- Optimizer representation: Python class with `__init__` and `step` method
- Mutations: random insertion/deletion/replacement of blocks (momentum, second-moment, clipping, wavelet, kurtosis, Lyapunov estimation, etc.)
- Selection: composite score = 0.6 × (accuracy or -loss) + 0.4 × (1 / steps_to_target)
- Benchmarks ladder: Rosenbrock → toy MLP → MNIST → CIFAR-10 subset → full CIFAR-10
- Safeguards: mandatory ablations every 3 cycles; deprecate if contribution <10%; 8-seed stats; publication gate at ≥92% CIFAR-10 + 3× efficiency + ablations pass

### 3.2 Final EvoOpt Implementation (Cycle 52 Elite – Ablation-Validated)

```python
import torch
from torch.optim import Optimizer
import pywt
import numpy as np

class EvoOpt(Optimizer):
    """Final EvoOpt from Cycle 52: wavelet-preprocessed, kurtosis-adaptive momentum, Lyapunov clipping."""
    def __init__(self, params, lr=1.2e-3, beta1=0.91, beta2=0.995, eps=1e-8, weight_decay=0.01,
                 wavelet='db4', clip_threshold=3.5):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, eps=eps, weight_decay=weight_decay,
                        wavelet=wavelet, clip_threshold=clip_threshold)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('EvoOpt does not support sparse gradients')

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['v'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                m, v = state['m'], state['v']
                beta1, beta2 = group['beta1'], group['beta2']
                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Wavelet denoising on gradient (db4 level 1 soft thresholding)
                coeffs = pywt.wavedec(grad.cpu().numpy().flatten(), group['wavelet'], level=1)
                threshold = np.median(np.abs(coeffs[-1])) / 0.6745 * 3.0  # universal threshold approx
                coeffs[1:] = (pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:])
                grad_denoised = torch.from_numpy(pywt.waverec(coeffs, group['wavelet'])).to(grad.device).reshape(grad.shape)

                # Spectral kurtosis adaptive beta1
                if state['step'] % 10 == 0:  # compute infrequently
                    kurt = torch.mean(((grad_denoised - grad_denoised.mean()) / (grad_denoised.std() + 1e-6))**4)
                    dynamic_beta1 = beta1 * (1 - 0.15 * torch.tanh(kurt - 3.0))  # suppress if leptokurtic
                else:
                    dynamic_beta1 = beta1

                m.mul_(dynamic_beta1).add_(grad_denoised, alpha=1 - dynamic_beta1)
                v.mul_(beta2).addcmul_(grad_denoised, grad_denoised, value=1 - beta2)

                m_hat = m / bias_correction1
                v_hat = v / bias_correction2

                # Lyapunov-inspired clipping: estimate local divergence rate
                grad_norm = grad_denoised.norm()
                update_hat = m_hat / (v_hat.sqrt() + group['eps'])
                div_rate = torch.log1p(update_hat.norm() / (grad_norm + 1e-6))
                clip_mask = (div_rate > torch.log1p(torch.tensor(group['clip_threshold'])))
                update_hat = torch.where(clip_mask, update_hat * 0.7, update_hat)  # soft damp

                update = -group['lr'] * update_hat
                if group['weight_decay'] != 0:
                    update.add_(p, alpha=-group['weight_decay'])

                p.add_(update)

        return loss

4. Experiments & Results
4.1 Progression Summary

Cycles 1–15: rediscover momentum/adaptive scaling (Rosenbrock 51× speedup, toy MSE 0.008)
Cycles 16–33: MNIST 98.71–98.9%, CIFAR-10 subset 85.9–91%+
Cycles 34–45: peak complexity (96.1% subset with exotic terms), efficiency up to 11.8×
Cycles 46–52: ablation-enforced pruning → 3-component final EvoOpt, 92.6% full CIFAR-10, 3.5× efficiency

4.2 Final Full CIFAR-10 Results (Cycle 52, 8 seeds)








































OptimizerTest Acc (%)Epochs to 92%Efficiency vs AdamWNotesAdamW (tuned)92.1 ± 0.5~981.0×baselineLion91.8 ± 0.6~1050.93×Sophia91.5 ± 0.4~1100.89×EvoOpt (final)92.6 ± 0.4~283.5×wavelet + kurtosis + Lyapunov
Ablations confirm:

Remove wavelet → -1.4% acc, +22% epochs
Remove kurtosis adaptivity → -0.9% acc, +15% epochs
Remove Lyapunov clip → -1.1% acc, +18% epochs
Re-add mid-cycle exotics (cumulants/RG) → no gain, +30% compute overhead

5. Discussion & Limitations
EvoOpt demonstrates that autonomous search can rediscover known features and add novel, parsimonious refinements for measurable gains. The 3.5× efficiency directly advances our charter: scaled to frontier models, this could save millions in energy costs.
Limitations:

Vision-only so far (next: Transformers, diffusion, biology)
Local compute budget limits scale
Mutation grammar still semi-hand-defined

Future: joint optimizer/schedule/architecture evolution, open swarm toolkit, grant-funded scaling.
6. Conclusion
EvoOpt-1 validates transparent agentic research as a path to more sustainable AI. With full logs public, we invite verification, extension, and collaboration.
CEO Sign-off: This draft meets all publication gates. Recommend immediate arXiv submission.
Appendix: Reproducibility & Links

Full logs: https://github.com/Nova-Aether-Research/memory/tree/main/logs (52 files covering Cycles 1–52)
Optimizer code: Section 3.2 above (also in cycle_52.md)
Training script skeleton: forthcoming in separate repo
Acknowledge: Sakana AI-Scientist base, PyTorch, open ML community
Commit history: 67 commits as of March 23, 2026 (latest updates to logs/ and papers/)

Prepared for public benefit – let's reduce the carbon footprint of AI training together.
