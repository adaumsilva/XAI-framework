# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- `Explanation.to_markdown()`: render explanations as Markdown tables with headlines and agreement summaries for reports, pull requests, and notebooks (#1).

## [0.1.0] - 2026-09-21

Initial MVP.

### Added
- `explain()` / `explain_global()` one-call API with automatic method selection.
- `ModelAdapter`: uniform `predict()` over scikit-learn estimators, Pipelines,
  XGBoost/LightGBM/CatBoost and plain callables; infers task, model family,
  feature names and class names.
- `CoalitionExplainer`: native Shapley-value attribution over feature coalitions -
  exact enumeration when affordable, size-enumerated + importance-sampled regression
  otherwise, closed form for linear regressors. Additive by construction.
- `LocalSurrogateExplainer`: native weighted local linear surrogate with categorical
  support and additive "contribution" mode.
- `PermutationExplainer`: global importance with log-loss / accuracy / R2 / MSE /
  MAE or a custom metric.
- `ConsensusExplainer` and `combine_explanations()`: mean or Borda-rank
  aggregation with Spearman agreement and sign agreement.
- `Explanation` / `ConsensusExplanation` with `to_text()`, `to_dataframe()`,
  `to_dict()`, `plot()`.
- Explainer registry with entry-point plugin discovery.
- Test suite, CI workflow, contributor docs and issue drafts.
