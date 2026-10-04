# Coalition Explainer (Shapley Values)

The Coalition explainer computes feature attributions based on the classic **Shapley value** from cooperative game theory (Shapley, 1953; Lundberg & Lee, 2017).

---

## Mathematical Formulation

A prediction task is modeled as a cooperative game where the set of features $N = \{1, 2, \dots, n\}$ acts as players collaborating to produce the prediction $f(x)$.

The attribution $\phi_i$ assigned to feature $i$ is its average marginal contribution across all possible coalitions $S \subseteq N \setminus \{i\}$:

$$\phi_i(v) = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(n - |S| - 1)!}{n!} \left( v(S \cup \{i\}) - v(S) \right)$$

### Characteristic Valuation Function

For a coalition $S$, the characteristic function $v(S)$ represents the expected model prediction when features in $S$ are fixed to their values in the explained instance $x$, and features outside $S$ ($N \setminus S$) are drawn from the marginal background distribution $D_{bg}$:

$$v(S) \approx \frac{1}{|D_{bg}|} \sum_{z \in D_{bg}} f(x_S, z_{N \setminus S})$$

---

## Axiomatic Guarantees

Shapley values are uniquely characterized by four fundamental axioms:

1. **Efficiency**: The sum of feature attributions equals the difference between the prediction $f(x)$ and the baseline expected value $\mathbb{E}[f(X)]$:
   $$\sum_{i=1}^n \phi_i = f(x) - \mathbb{E}[f(X)]$$
2. **Symmetry**: If two features contribute equally to all possible subsets ($v(S \cup \{i\}) = v(S \cup \{j\})$ for all $S \subseteq N \setminus \{i, j\}$), then $\phi_i = \phi_j$.
3. **Dummy / Null Player**: If a feature never changes the prediction ($v(S \cup \{i\}) = v(S)$ for all $S$), then $\phi_i = 0$.
4. **Additivity**: If a model is a linear combination of two models $f = f_1 + f_2$, the attributions add up: $\phi_i(f) = \phi_i(f_1) + \phi_i(f_2)$.

---

## Computational Strategies

Computing exact Shapley values requires evaluating all $2^n - 2$ non-trivial coalitions. `CoalitionExplainer` automatically chooses the optimal strategy based on the feature dimensionality:

```
                  +---------------------------+
                  |  Number of features (n)   |
                  +-------------+-------------+
                                |
               +----------------+----------------+
               |                                 |
           n <= 11                            n > 11
               |                                 |
     [ Exact Enumeration ]             [ Kernel Shapley Sampling ]
     Evaluates all 2^n - 2             Importance-samples pairs
     subsets explicitly.               weighted by Shapley kernel:
                                       pi(s) = (n-1) / (comb(n,s) * s * (n-s))
```

### 1. Exact Enumeration ($n \le 11$)
When $n \le 11$, all $2^n - 2$ coalitions are enumerated explicitly and weighted by the Shapley kernel:
$$\pi(s) = \frac{n - 1}{\binom{n}{s} s (n - s)}$$
Weighted least-squares regression over all coalitions recovers the exact mathematical Shapley values.

### 2. Importance-Sampled Approximation ($n > 11$)
When $n > 11$, the explainer samples complementary coalition pairs $(S, N \setminus S)$ according to the kernel probability distribution $\pi(s)$, prioritizing very small and very large coalitions where marginal variance is greatest.

### 3. Linear Regressor Closed Form
For standard linear regressors ($f(x) = w^T x + b$), the marginal contribution simplifies analytically:
$$\phi_i = w_i (x_i - \mathbb{E}[X_i])$$
Evaluating linear models requires zero sampling and runs instantaneously.

---

## Usage Example

```python
from xai_framework import CoalitionExplainer

# Initialize explainer with background reference data
explainer = CoalitionExplainer(model, X_train, n_background=50, random_state=42)

# Explain a single local instance
exp = explainer.explain_instance(X_test.iloc[0])

# Global explanation across a dataset sample
glob_exp = explainer.explain_global(X_test.iloc[:50])
```
