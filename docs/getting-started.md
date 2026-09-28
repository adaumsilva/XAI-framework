# Getting Started

This guide walks you through installing XAI-Framework, configuring dependencies, and generating your first local and global model explanations.

---

## Installation

### From GitHub

Install directly from GitHub using `pip`:

```bash
pip install git+https://github.com/adaumsilva/XAI-framework.git
```

### With Optional Plotting Support

To generate visual explanation plots via Matplotlib:

```bash
pip install "xai-framework[plot] @ git+https://github.com/adaumsilva/XAI-framework.git"
```

### Development Installation

To set up an editable installation for development and testing:

```bash
git clone https://github.com/adaumsilva/XAI-framework.git
cd XAI-framework
python -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate
pip install -e ".[dev,docs]"
pytest
```

---

## Local Explanations (Instance-Level)

A local explanation explains a specific model prediction for an individual instance $x$.

```python
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from xai_framework import explain

iris = load_iris(as_frame=True)
X, y = iris.data, iris.target

clf = RandomForestClassifier(random_state=42).fit(X, y)

# Explain the first sample
exp = explain(clf, X, instance=X.iloc[0])

# Inspect summary
print(exp.to_text())
```

### Inspecting Results

The returned `Explanation` (or `ConsensusExplanation`) object provides several inspection methods:

- **Top Features**: `exp.top(3)` returns the top 3 most influential features and their attribution scores.
- **Agreement Level**: `exp.agreement_level()` indicates whether consensus methods have `"high"`, `"moderate"`, or `"low"` rank agreement.
- **Pairwise Correlation**: `exp.agreement` provides the overall Spearman $\rho$ correlation.
- **Exporting Data**:
  - `exp.to_dataframe()` returns a pandas DataFrame with features, instance values, consensus values, and per-explainer attributions.
  - `exp.to_markdown()` outputs a clean GitHub Flavored Markdown table.
  - `exp.to_dict()` provides a serializable Python dictionary.
  - `exp.plot()` displays a horizontal bar chart showing attributions and individual method points.

```python
# Convert to Markdown table
print(exp.to_markdown())

# Plot explanation
exp.plot()
```

---

## Global Explanations (Model-Level)

Global explanations describe overall feature importance across a dataset:

```python
from xai_framework import explain_global

# Requires true labels y for permutation importance
glob = explain_global(clf, X, y=y)
print(glob.to_text())
```

When ground truth `y` is provided, `explain_global()` calculates a consensus between the mean absolute Shapley values (from `coalition`) and permutation importance. Without `y`, it reports global attributions from `coalition`.

---

## Selecting Specific Methods & Weights

You can specify which explainer methods to run, assign custom weights, and configure aggregation strategies:

```python
# Run only surrogate
exp_surr = explain(clf, X, instance=X.iloc[0], methods="surrogate")

# Weighted Borda count between coalition and surrogate
exp_custom = explain(
    clf,
    X,
    instance=X.iloc[0],
    methods=["coalition", "surrogate"],
    weights={"coalition": 2.0, "surrogate": 1.0},
    aggregation="rank",
)
```
