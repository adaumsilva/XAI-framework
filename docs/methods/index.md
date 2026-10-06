# Methods Overview

XAI-Framework implements native, self-contained explainability algorithms designed from foundational principles in cooperative game theory, local surrogate modeling, and permutation sampling.

All explainers return a standardized [`Explanation`][xai_framework.explanation.Explanation] object, making their outputs directly comparable and capable of being combined into a [`ConsensusExplanation`][xai_framework.explanation.ConsensusExplanation].

---

## Explainer Matrix

| Explainer | Key Foundation | Primary Scope | Mathematical Intuition |
| :--- | :--- | :--- | :--- |
| **[`coalition`](coalition.md)** | Shapley values (Shapley 1953; Lundberg & Lee 2017) | Local & Global | Treats features as players in a cooperative game; calculates average marginal contributions across feature subsets. |
| **[`surrogate`](surrogate.md)** | Local Linear Surrogates (LIME, Ribeiro et al. 2016) | Local | Perturbs instances in feature space, weights them by proximity kernel, and fits a sparse weighted linear model. |
| **[`permutation`](permutation.md)** | Permutation Importance (Breiman 2001; Fisher et al. 2019) | Global | Randomly permutes feature columns on held-out data and measures performance drop. |
| **[`consensus`](consensus.md)** | Multi-method Synthesis (*ours*) | Local & Global | Normalizes disparate attributions to a shared scale, computes ensemble rankings, and quantifies inter-method agreement ($\rho$). |

---

## Core Properties

1. **Native Pure-Python Implementations**: Built using `numpy` and `pandas` without external dependencies on `shap` or `lime`.
2. **Unified Representation**: Every local explainer assigns an additive attribution value $\phi_i$ to feature $i$, where positive values push toward the target outcome.
3. **Consensus & Diagnostics**: Instead of trusting a single algorithm's assumptions, XAI-Framework computes consensus and reports whether the explainers agree or diverge.
