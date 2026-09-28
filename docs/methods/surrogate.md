# Local Surrogate Explainer

The Local Surrogate explainer approximates complex black-box model decisions locally around an individual prediction using an interpretable, weighted linear model (inspired by Ribeiro et al., 2016).

---

## Mathematical Formulation

Complex nonlinear decision boundaries are often locally smooth. Around a specific query point $x$, the black box $f(z)$ can be approximated by a linear surrogate model $g(z)$:

$$g(z) = w^T z + b$$

### 1. Perturbation Generation
To understand the local decision landscape, the explainer generates $K$ perturbed instances $\{z_k\}_{k=1}^K$ in the vicinity of $x$:
- **Continuous Features**: Sampled from a normal distribution $\mathcal{N}(x_j, \sigma_j^2)$ scaled by the feature standard deviations observed in training data.
- **Categorical Features**: Perturbed based on their observed empirical training frequencies.

### 2. Proximity Kernel Weighting
Each perturbed instance $z_k$ is weighted by an exponential distance kernel $\pi_x(z_k)$ reflecting its proximity to the original instance $x$:

$$\pi_x(z_k) = \exp\left( -\frac{D(x, z_k)^2}{\sigma^2} \right)$$

where $D(x, z_k)$ is the normalized Euclidean distance in standardized feature space, and $\sigma$ represents the kernel bandwidth.

### 3. Weighted Linear Optimization
The surrogate weights $w$ and intercept $b$ are solved via weighted Ridge regression:

$$\min_{w, b} \sum_{k=1}^K \pi_x(z_k) \left( f(z_k) - (w^T z_k + b) \right)^2 + \lambda \|w\|_2^2$$

---

## Additive Attributions & Local Fidelity

To ensure consistency with Shapley values and the consensus framework, raw regression coefficients $w_i$ are converted into additive instance attributions:

$$\phi_i = w_i \cdot (x_i - \mu_i)$$

where $\mu_i$ denotes the reference or background mean. Under this transformation:
- $\phi_i > 0$ indicates that feature $i$ pushes the prediction toward the target class or a higher predicted value.
- $\phi_i < 0$ indicates an opposing effect.

### Surrogate Quality Metric ($R^2$)
The explainer calculates the weighted coefficient of determination ($R_{local}^2$) of the linear surrogate on the perturbed sample:
- High $R_{local}^2$ ($\ge 0.85$) confirms that the local boundary is approximately linear and the attributions are trustworthy.
- Low $R_{local}^2$ alerts the user to high local non-linearity or complex interactions.

---

## Usage Example

```python
from xai_framework import LocalSurrogateExplainer

# Configure surrogate explainer with 2000 perturbation samples
explainer = LocalSurrogateExplainer(
    model=model,
    X_background=X_train,
    n_samples=2000,
    kernel_width=None,  # Auto-selected from feature dimensions
    random_state=42,
)

# Explain single query instance
exp = explainer.explain_instance(X_test.iloc[0])

# Inspect local fit goodness
print(f"Local R^2: {exp.metadata.get('local_r2'):.3f}")
```
