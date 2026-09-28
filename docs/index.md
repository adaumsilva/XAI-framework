# XAI-Framework

**One call, several attribution methods, and a measure of how much they agree.**

`xai-framework` is a model-agnostic explainable-AI library with its own native implementations of the ideas behind the most popular attribution methods — Shapley values over feature coalitions, local linear surrogates, and permutation importance — unified behind a single interface.

It puts their outputs on a common scale and reports a **consensus explanation** together with an **agreement score**. When the methods agree you can trust the ranking; when they do not, the framework highlights the disagreement instead of silently picking one.

---

## Key Features

- **Consensus Explanations**: Combines multiple explainer algorithms (e.g. Shapley coalitions + local linear surrogates) into a unified, balanced attribution.
- **Agreement Diagnostics**: Measures rank agreement using Spearman's rank correlation ($\rho$) and top-$k$ sign consistency.
- **Pure Python & Zero Bloat**: Native implementations in `numpy` and `pandas`. No heavy third-party explainer dependencies.
- **Model Agnostic**: Seamlessly explains scikit-learn estimators, gradient boosters (XGBoost, LightGBM, CatBoost), and custom prediction callables.
- **Flexible Outputs**: Export explanations to plain text, Markdown tables, pandas DataFrames, Python dictionaries, or Matplotlib visualisations.

---

## Quick Example

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from xai_framework import explain

# Prepare dataset and train model
data = load_breast_cancer(as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, random_state=42)
model = RandomForestClassifier(random_state=42).fit(X_train, y_train)

# Explain a single test instance
exp = explain(model, X_train, instance=X_test.iloc[0])
print(exp.to_text())
```

Output:
```text
Predicted class 'malignant' with probability 0.980 (consensus).
Features that push towards 'malignant':
  + worst concave points = 0.2051  (+0.2548)
  + mean concave points = 0.08172  (+0.1159)
  + worst concavity = 0.5106  (+0.1124)
  + mean concavity = 0.1445  (+0.08179)
  + worst texture = 29.66  (+0.06424)
Agreement between coalition + surrogate: moderate (rho = 0.76) - the top features are reliable, lower ranks less so.
```

---

## Next Steps

- Check out the [Getting Started](getting-started.md) guide for installation and walkthrough tutorials.
- Explore [Methods Overview](methods/index.md) to understand the mathematical foundations of each explainer.
- Learn about the consensus mechanism in [Consensus & Agreement](methods/consensus.md).
- Extend the framework with custom explainers in [Extending](extending.md).
- Read the [API Reference](api/index.md) for full parameter and method documentation.
