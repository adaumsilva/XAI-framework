# Permutation Feature Importance

The Permutation explainer calculates global feature importance by assessing the degradation in model performance when the relationship between a feature and the target variable is severed through random shuffling (Breiman, 2001; Fisher et al., 2019).

---

## Mathematical Formulation

Let $X \in \mathbb{R}^{m \times n}$ denote a dataset of $m$ samples and $n$ features, with corresponding true targets $y$, and let $L(y, \hat{y})$ denote a scoring loss function.

### 1. Baseline Performance
Compute the baseline model error on the uncorrupted evaluation dataset:

$$L_{\text{base}} = L(y, f(X))$$

### 2. Feature Permutation
For each feature $j \in \{1, \dots, n\}$:
1. Construct permuted matrix $X^{\pi_j}$ by randomly shuffling column $j$ across all rows, destroying any statistical association between feature $j$ and target $y$ while preserving the marginal feature distribution.
2. Evaluate corrupted model loss:
   $$L_{\text{perm}}^{(j)} = L(y, f(X^{\pi_j}))$$
3. Compute raw importance score as the performance degradation:
   $$I(j) = L_{\text{perm}}^{(j)} - L_{\text{base}}$$

### 3. Repeated Resampling & Uncertainty
To reduce sampling variance, shuffling is repeated across $B$ repetitions ($b = 1, \dots, B$). The explainer computes both the mean importance drop $\bar{I}(j)$ and its population standard deviation (`importances_std`, using `ddof=0`):

$$\bar{I}(j) = \frac{1}{B} \sum_{b=1}^B I_b(j), \quad \sigma_I(j) = \sqrt{\frac{1}{B} \sum_{b=1}^B (I_b(j) - \bar{I}(j))^2}$$

Note that `importances_std` reflects the population standard deviation across the $B$ permutation runs, rather than a sample standard deviation or standard error of the mean.

---

## Metric Selection

`PermutationExplainer` requires metrics where higher scores represent better model performance.

### Built-in Metrics
- **Classifiers**:
  - `"log_loss"` (*default*): Smooth negative cross-entropy computed from predicted probabilities. Captures subtle shifts in class probability distributions even when top-predicted discrete labels do not flip.
  - `"accuracy"`: Discrete classification accuracy.
- **Regressors**:
  - `"r2"` (*default*): Coefficient of determination $R^2$.
  - `"neg_mse"`: Negative mean squared error ($-MSE$).
  - `"neg_mae"`: Negative mean absolute error ($-MAE$).

> [!NOTE]
> Passing unregistered metric names such as `"mse"` or `"roc_auc"` will raise a `ValueError`.

### Custom Scoring Callables
For non-standard metrics such as ROC AUC, pass a callable `metric(y_true, model_output) -> float` when initializing `PermutationExplainer` (where higher indicates better performance):

```python
from sklearn.metrics import roc_auc_score

# Classifier model_output is the array of predicted probabilities
explainer = PermutationExplainer(
    model=model,
    X_background=X_test,
    metric=lambda y_true, y_prob: roc_auc_score(y_true, y_prob[:, 1]),
    n_repeats=10,
    random_state=42,
)
glob_exp = explainer.explain_global(X_test, y=y_test)
```

---

## Importance of Held-Out Data

> [!WARNING]
> Permutation importance should always be evaluated on **held-out validation or test data**, never solely on training data.
>
> On unregularized models or deep decision trees, the model may achieve near-zero training error even with noisy, overfitted features. Evaluating permutation importance on training data can incorrectly assign high importance to spurious patterns.

---

## Usage Example

```python
from xai_framework import PermutationExplainer

# Initialize permutation explainer with held-out validation data
explainer = PermutationExplainer(
    model=model,
    X_background=X_test,
    metric="log_loss",
    n_repeats=10,
    random_state=42,
)

# Compute global explanations using true labels y_test
glob_exp = explainer.explain_global(X_test, y=y_test)

# Display ranked global importances
for feature, score in glob_exp.top(5):
    print(f"{feature}: {score:.4f}")
```
