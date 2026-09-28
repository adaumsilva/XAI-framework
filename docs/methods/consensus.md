# Consensus & Agreement

Attribution methods evaluate models from distinct perspectives: Shapley values measure average marginal coalition contributions, while local surrogates measure directional gradients within a perturbed neighborhood. In practice, different explainers frequently produce differing rankings on the same model and query instance.

Instead of forcing practitioners to arbitrarily pick a single algorithm, XAI-Framework computes a **Consensus Explanation** accompanied by an **Agreement Score**.

---

## 1. Scale Normalization

Different explainability methods output attributions on fundamentally different scales (e.g. log-odds shifts vs. raw probability differences). Before combination, raw attributions are $L_1$-normalized to represent proportional influence:

$$\tilde{\phi}_i^{(m)} = \frac{\phi_i^{(m)}}{\sum_{j=1}^n |\phi_j^{(m)}|}$$

If all attributions for a method are zero, the normalized vector remains zero.

---

## 2. Aggregation Strategies

XAI-Framework provides two aggregation strategies configurable via `aggregation="mean"` or `aggregation="rank"`:

### Weighted Mean (`"mean"`, default)
Computes the weighted linear combination of normalized attributions across all $M$ methods:

$$\phi_i^{\text{consensus}} = \frac{\sum_{m=1}^M w_m \cdot \tilde{\phi}_i^{(m)}}{\sum_{m=1}^M w_m}$$

This approach preserves both the direction (sign) and relative magnitude of attributions.

### Borda Rank (`"rank"`)
Computes a weighted Borda count across the magnitude ranks of each method, retaining the sign from the mean attribution:

$$\text{score}_i = \sum_{m=1}^M w_m \cdot \text{rank}(|\phi_i^{(m)}|)$$

$$\phi_i^{\text{consensus}} = \text{sign}\left( \sum_{m=1}^M w_m \tilde{\phi}_i^{(m)} \right) \cdot \frac{\text{score}_i}{\sum_{j=1}^n \text{score}_j}$$

Rank aggregation provides extra robustness against an outlier method producing extreme attribution magnitudes.

---

## 3. Quantifying Inter-Method Agreement

### Pairwise Spearman Rank Correlation ($\rho$)
For each pair of methods $(u, v)$, the framework computes the Spearman rank correlation $\rho_{u, v}$ over the feature magnitude vectors $|v_u|$ and $|v_v|$:

$$\rho_{u, v} = 1 - \frac{6 \sum_{i=1}^n d_i^2}{n(n^2 - 1)}$$

where $d_i$ is the rank difference between methods $u$ and $v$ for feature $i$.

The overall consensus agreement $\rho$ is the arithmetic mean of all non-diagonal pairwise correlations:

$$\bar{\rho} = \frac{2}{M(M-1)} \sum_{u < v} \rho_{u, v}$$

### Sign Agreement on Top Features
In addition to rank correlation, the framework checks whether methods agree on the *direction of effect* (positive or negative) for the top-$k$ most important features:

$$\text{sign\_agreement} = \frac{1}{k} \sum_{i \in \text{top-k}} \mathbb{I}\left( \text{sign}(\phi_i^{(u)}) = \text{sign}(\phi_i^{(v)}) \quad \forall u, v \right)$$

---

## 4. Agreement Tiers

The framework categorizes consensus reliability into three intuitive tiers:

| Tier | Condition | Interpretation |
| :--- | :--- | :--- |
| **High** | $\bar{\rho} \ge 0.8$ | Methods strongly align. The feature rankings and top attributions are highly robust across algorithmic assumptions. |
| **Moderate** | $0.5 \le \bar{\rho} < 0.8$ | Top features are generally consistent and reliable, but middle-to-lower rankings may vary across methods. |
| **Low** | $\bar{\rho} < 0.5$ | Significant divergence between methods. Model predictions may rely on complex non-linear feature interactions where local approximations disagree. |

---

## Usage Example

```python
from xai_framework import explain

exp = explain(
    model,
    X_train,
    instance=X_test.iloc[0],
    methods=["coalition", "surrogate"],
    weights={"coalition": 2.0, "surrogate": 1.0},
    aggregation="mean",
)

# Inspect agreement diagnostics
print("Agreement score (rho):", exp.agreement)
print("Agreement level:", exp.agreement_level())
print("Pairwise matrix:\n", exp.agreement_matrix)
```
