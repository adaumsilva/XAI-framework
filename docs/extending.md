# Extending XAI-Framework

XAI-Framework is designed to be easily extensible. You can register custom attribution algorithms internally or build third-party plugins that integrate seamlessly with the framework's consensus engine.

---

## The Explainer Contract

Every explainer inherits from [`BaseExplainer`][xai_framework.explainers.base.BaseExplainer] and produces a standardized [`Explanation`][xai_framework.explanation.Explanation] object.

### Step 1: Subclass `BaseExplainer`

Create a new class inheriting from `BaseExplainer`:

```python
from __future__ import annotations
import numpy as np
from xai_framework import BaseExplainer, register_explainer

@register_explainer("custom_gradient")
class CustomGradientExplainer(BaseExplainer):
    """Example custom explainer estimating gradients via finite differences."""

    name = "custom_gradient"
    supports_local = True
    supports_global = False
    requires_y = False

    def _explain_instance(self, x: np.ndarray, target: int | None) -> Explanation:
        # x is guaranteed to be a 1D float numpy array
        # Talk to the black-box model only through self.adapter
        base_pred = self.adapter.predict_scalar(x.reshape(1, -1), target=target)

        eps = 1e-4
        n_features = len(x)
        attributions = np.zeros(n_features)

        for i in range(n_features):
            x_perturbed = x.copy()
            x_perturbed[i] += eps
            pred_perturbed = self.adapter.predict_scalar(x_perturbed.reshape(1, -1), target=target)
            attributions[i] = (pred_perturbed - base_pred) / eps

        # Build standard Explanation instance using helper
        return self._make_explanation(
            values=attributions,
            instance=x,
            target=target,
            prediction=base_pred,
            base_value=None,
            metadata={"step_size": eps},
        )
```

---

## Core Guidelines & Invariants

When implementing a custom explainer, adhere to the following framework conventions:

1. **Model Communication via `self.adapter`**:
   Never call `model.predict()` or `model.predict_proba()` directly. Always communicate through `self.adapter` ([`ModelAdapter`][xai_framework.model.ModelAdapter]), which abstracts classification, regression, single-output functions, and feature name resolution.

2. **Strict Determinism via `self.rng`**:
   Never use global random seeds (`np.random.seed` or `np.random.*`). Always use the instance-level random generator `self.rng` initialized with the user-provided `random_state`.

3. **Standard Sign Convention**:
   Attributions must be formatted such that a positive value $\phi_i > 0$ means feature $i$ pushes the prediction towards the target outcome (higher value or higher probability for class `target`).

4. **Optional Dependencies & Lazy Imports**:
   If your explainer relies on optional external libraries (such as PyTorch or XGBoost), override `is_available()` and perform imports lazily inside `__init__`.

---

## External Entry-Point Plugins

External Python packages can register custom explainers into XAI-Framework without modifying the `xai-framework` repository.

In your external package's `pyproject.toml`, declare entry points under `xai_framework.explainers`:

```toml
[project.entry-points."xai_framework.explainers"]
anchors = "my_xai_package.anchors:AnchorsExplainer"
integrated_gradients = "my_xai_package.ig:IntegratedGradientsExplainer"
```

Once installed in the environment, your explainer will be automatically detected and available to `explain()`:

```python
from xai_framework import explain

# Custom explainer is loaded via entry points automatically
exp = explain(model, X_train, instance=x_query, methods=["coalition", "anchors"])
```
