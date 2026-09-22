# XAI-Framework

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/adaumsilva/XAI-framework/actions/workflows/ci.yml/badge.svg)](https://github.com/adaumsilva/XAI-framework/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/adaumsilva/XAI-framework/branch/main/graph/badge.svg)](https://codecov.io/gh/adaumsilva/XAI-framework)
[![GitHub issues](https://img.shields.io/github/issues/adaumsilva/XAI-framework)](https://github.com/adaumsilva/XAI-framework/issues)
[![Good first issues](https://img.shields.io/github/issues/adaumsilva/XAI-framework/good%20first%20issue?label=good%20first%20issues&color=7057ff)](https://github.com/adaumsilva/XAI-framework/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/adaumsilva/XAI-framework/blob/main/CONTRIBUTING.md)
[![GitHub contributors](https://img.shields.io/github/contributors/adaumsilva/XAI-framework)](https://github.com/adaumsilva/XAI-framework/graphs/contributors)
[![GitHub stars](https://img.shields.io/github/stars/adaumsilva/XAI-framework?style=social)](https://github.com/adaumsilva/XAI-framework/stargazers)

**One call, several attribution methods, and a measure of how much they agree.**

XAI-Framework is a model-agnostic explainable-AI library with its **own native
implementations** of the ideas behind the most popular attribution methods - Shapley
values over feature coalitions, local linear surrogates, permutation importance -
unified behind a single interface. It puts their outputs on a common scale and reports
a **consensus explanation** together with an **agreement score**. When the methods
agree you can trust the ranking; when they don't, the framework tells you instead of
silently picking one.

Pure `numpy` + `pandas`. No compiled dependencies, no third-party explainer packages.

```python
from xai_framework import explain

exp = explain(model, X_train, instance=X_test.iloc[0])
print(exp.to_text())
```

```
Predicted class 'malignant' with probability 0.980 (consensus).
Features that push towards 'malignant':
  + worst concave points = 0.2051  (+0.2548)
  + mean concave points = 0.08172  (+0.1159)
  + worst concavity = 0.5106  (+0.1124)
  + mean concavity = 0.1445  (+0.08179)
  + worst texture = 29.66  (+0.06424)
Agreement between coalition + surrogate: moderate (rho = 0.76) - the top features are reliable, lower ranks less so.
```

> **Status:** alpha (v0.1). The API is small and stable enough to build on, but expect
> additions. Contributions are very welcome - see [Contributing](#contributing) and the
> [open issues](https://github.com/adaumsilva/XAI-framework/issues).

## The methods

| Explainer | Inspired by | What it does | Scope |
|---|---|---|---|
| **`coalition`** | Shapley values (Shapley 1953; Lundberg & Lee 2017) | Treats features as players in a cooperative game and attributes the prediction by each feature's average marginal contribution over feature coalitions. Exact when the feature count allows, importance-sampled beyond that, closed-form for linear regressors. Attributions are *additive*: `base_value + sum(values) == prediction`. | local + global |
| **`surrogate`** | LIME (Ribeiro et al. 2016) | Perturbs the instance, weights samples by proximity, and fits a sparse weighted linear model whose coefficients describe the black box *right here*. Reports additive contributions so it is directly comparable with `coalition`. | local |
| **`permutation`** | Breiman 2001; Fisher et al. 2019 | Shuffles one feature at a time and measures the drop in score. Smooth log-loss default for classifiers so small effects still register. Needs `y`. | global |
| **`consensus`** | *(ours)* | Runs the methods above, L1-normalises, combines (weighted mean or Borda rank), and reports pairwise Spearman agreement plus sign agreement on the top features. | local + global |

Everything returns the same `Explanation` object, so you get `to_text()`,
`to_dataframe()`, `to_dict()` and `plot()` regardless of the method - and any new
method that follows the contract can join the consensus.

## Install

The package is not on PyPI yet - install it straight from GitHub:

```bash
pip install git+https://github.com/adaumsilva/XAI-framework.git
```

With plotting support (adds matplotlib for `.plot()`):

```bash
pip install "xai-framework[plot] @ git+https://github.com/adaumsilva/XAI-framework.git"
```

Requires Python 3.10+. Tested on 3.10 - 3.14. Only `numpy` and `pandas` are pulled in.

To work on the code itself, clone it and install in editable mode with the dev tools:

```bash
git clone https://github.com/adaumsilva/XAI-framework.git
cd XAI-framework
pip install -e ".[dev]"
pytest
```

## Usage

### Explain one prediction (local)

```python
from sklearn.ensemble import RandomForestClassifier
from xai_framework import explain

model = RandomForestClassifier().fit(X_train, y_train)

exp = explain(model, X_train, instance=X_test.iloc[0])   # coalition + surrogate consensus
exp.top(3)                    # [('worst concave points', 0.255), ...]
exp.agreement                 # 0.76  (Spearman rho between the methods' rankings)
exp.agreement_matrix          # pairwise table
exp.components["coalition"]   # the underlying Shapley-value explanation
exp.to_dataframe()            # feature | feature_value | consensus | coalition | surrogate | rank
exp.plot()                    # bar chart with one marker per method
```

`instance` can be a row (array / Series / one-row DataFrame) or an integer index into `X`.
For classifiers, pass `target="benign"` (label or index) to explain a specific class;
the default is the predicted class.

### Explain the whole model (global)

```python
glob = explain(model, X_test, y=y_test)   # coalition mean|phi| + permutation importance
glob.to_text()
```

Without `y`, only the coalition explainer runs. Use held-out data for permutation
importance: on the training rows of a fully grown ensemble every feature looks
unimportant.

### Pick methods explicitly

```python
explain(model, X, instance=0, methods="surrogate")                       # single method
explain(model, X, instance=0, methods=["coalition", "surrogate"],
        weights={"coalition": 2, "surrogate": 1}, aggregation="rank")    # weighted Borda
explain(model, X, instance=0,
        explainer_kwargs={"surrogate": {"n_samples": 2000},
                          "coalition": {"n_background": 100}})
```

### Use the explainer classes directly

```python
from xai_framework import (
    CoalitionExplainer, LocalSurrogateExplainer, PermutationExplainer, ConsensusExplainer,
)

coal = CoalitionExplainer(model, X_train, n_background=50)       # exact/sampling chosen automatically
surr = LocalSurrogateExplainer(model, X_train, n_samples=5000, categorical_features=["sex"])
perm = PermutationExplainer(model, X_train, metric="accuracy")

coal.explain_instance(x)
coal.explain(X_test.head(20))                                    # batch
coal.explain_global(X_test)
perm.explain_global(X_test, y_test)

consensus = ConsensusExplainer(model, X_train, methods=(coal, surr))
consensus.explain_instance(x)
```

### How the coalition explainer scales

Cost per local explanation is `n_coalitions x n_background` model evaluations,
batched into a handful of `predict` calls:

| features | coalitions (auto) | result |
|---|---|---|
| <= 11 | all `2^n - 2` | exact Shapley values |
| 30 | 2 108 | sizes 1, 2, 28, 29 enumerated exactly; the rest importance-sampled |
| any, linear regressor | - | closed form, instant |

A 30-feature random forest explains locally in ~0.2 s and globally (100 rows) in a few
seconds on a laptop. Tune with `n_coalitions`, `n_background`, `n_global`.

### Works with

* scikit-learn estimators and `Pipeline`s (anything with `predict_proba` / `predict`)
* XGBoost, LightGBM, CatBoost and any other library with the same interface
* plain callables `f(X) -> predictions` (probabilities or values)
* NumPy arrays and pandas DataFrames (column names become feature names)

Models fitted on DataFrames are handed DataFrames again, so you won't see
"X does not have valid feature names" warnings.

## The `Explanation` object

| Attribute / method | Meaning |
|---|---|
| `feature_names`, `values` | one attribution per feature; positive pushes towards the target class / higher value |
| `feature_values` | the explained instance (local only) |
| `prediction`, `base_value`, `target` | model output, reference value, class explained |
| `top(k)`, `ranks()`, `normalized()` | quick views |
| `to_text()`, `to_markdown()`, `to_dataframe()`, `to_dict()`, `plot()` | outputs |
| `metadata` | method-specific extras (surrogate `local_r2`, coalition `algorithm` / `exact`, permutation `importances_std`, ...) |

`ConsensusExplanation` adds `components`, `agreement`, `agreement_matrix` and
`agreement_level()` (`"high"` >= 0.8, `"moderate"` >= 0.5, `"low"`).

## Adding your own explainer

Subclass `BaseExplainer`, implement `_explain_instance` and/or `_explain_global`,
and register it:

```python
from xai_framework import BaseExplainer, register_explainer

@register_explainer("my_method")
class MyExplainer(BaseExplainer):
    supports_global = False

    def _explain_instance(self, x, target):
        values = ...  # one number per feature
        return self._make_explanation(values, x, target, base_value=None)
```

It is now usable as `explain(model, X, instance=0, methods=["coalition", "my_method"])`.
Third-party packages can register through the `xai_framework.explainers` entry-point
group - see [CONTRIBUTING.md](CONTRIBUTING.md).

## Roadmap

See [ROADMAP.md](ROADMAP.md) and the issue tracker. Highlights: a fast exact path for
tree ensembles, rule-based (anchor) and counterfactual explainers, text and image
support, faithfulness metrics, a CLI and an HTML report, and a docs site.

## Contributing

Bug reports, explainers, docs and benchmarks are all welcome. Start with the issues
labelled `good first issue`, and read [CONTRIBUTING.md](CONTRIBUTING.md) for the
development setup and conventions. Please follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## References

* Shapley (1953). *A Value for n-Person Games.* Contributions to the Theory of Games II.
* Lundberg & Lee (2017). *A Unified Approach to Interpreting Model Predictions.* NeurIPS.
* Ribeiro, Singh & Guestrin (2016). *"Why Should I Trust You?": Explaining the Predictions of Any Classifier.* KDD.
* Breiman (2001). *Random Forests.* Machine Learning. / Fisher, Rudin & Dominici (2019). *All Models are Wrong, but Many are Useful.* JMLR.

## License

MIT - see [LICENSE](LICENSE).
