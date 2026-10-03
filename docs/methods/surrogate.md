# Local Surrogate Explainer

The Local Surrogate explainer approximates complex black-box model decisions locally around an individual prediction using an interpretable, weighted linear model (inspired by Ribeiro et al., 2016).

---

## Mathematical Formulation

Complex nonlinear decision boundaries are often locally smooth. Around a specific query point $x$, the black box $f(z)$ can be approximated by a linear surrogate model $g(z)$:

$$g(z) = w^T z + b$$

### 1. Perturbation Generation
To understand the local decision landscape, the explainer generates $K$ perturbed instances $\{z_k\}_{k=1}^K$:
- **Continuous Features**: By default, perturbations are centered on the background mean $\mu_j$ and drawn from the training distribution $\mathcal{N}(\mu_j, \sigma_j^2)$. To draw perturbations tightly centered around the query instance ($\mathcal{N}(x_j, \sigma_j^2)$), set `sample_around_instance=True`.
- **Categorical Features**: Perturbed by sampling from their empirical training distribution frequencies.
- In both cases, the query instance $x$ is explicitly preserved as the first sample ($z_0 = x$).

### 2. Proximity Kernel Weighting
Perturbations are mapped into an interpretable standardized representation $Z$: continuous features are z-scored, and categorical features are represented via indicator encoding $1[z_{k, j} == x_j]$.

Each sample $z_k$ is weighted by an exponential proximity kernel:

$$\pi_x(z_k) = \exp\left( -\frac{D(x, z_k)^2}{2 \cdot \text{kernel\_width}^2} \right)$$

where $D(x, z_k) = \|Z_k - Z_0\|_2$ is the Euclidean distance in standardized space, and $\text{kernel\_width}$ defaults to $0.75 \sqrt{n_{\text{features}}}$.

### 3. Weighted Linear Optimization
The surrogate weights $w$ and unpenalized intercept $b$ are solved via closed-form weighted Ridge regression:

$$\min_{w, b} \sum_{k=1}^K \pi_x(z_k) \left( f(z_k) - (w^T z_k + b) \right)^2 + \alpha \|w\|_2^2$$

---

## Additive Attributions & Local Fidelity

In default `mode="contribution"`, raw regression coefficients $w_i$ are converted into additive feature contributions using standardized instance values:
- **Continuous Features**:
  $$\phi_i = w_i \cdot \frac{x_i - \mu_i}{\sigma_i}$$
- **Categorical Features**:
  $$\phi_j = w_j \cdot 1[x_j == x_j] = w_j$$

Under this formulation, $b + \sum_{i=1}^n \phi_i = g(x)$ recovers the surrogate's local prediction, ensuring direct comparability with Shapley values and consensus aggregation. Setting `mode="coefficient"` reports raw regression weights $w$.

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
