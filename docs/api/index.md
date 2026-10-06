# API Reference

Complete documentation generated directly from docstrings across the XAI-Framework codebase.

---

## Core Modules

- **[`auto`](auto.md)**: Top-level convenience functions [`explain()`][xai_framework.auto.explain] and [`explain_global()`][xai_framework.auto.explain_global], plus explainer factory [`build_explainer()`][xai_framework.auto.build_explainer].
- **[`explainers`](explainers.md)**: Base classes and concrete explainer implementations ([`CoalitionExplainer`][xai_framework.explainers.coalition.CoalitionExplainer], [`LocalSurrogateExplainer`][xai_framework.explainers.surrogate.LocalSurrogateExplainer], [`PermutationExplainer`][xai_framework.explainers.permutation.PermutationExplainer], and [`ConsensusExplainer`][xai_framework.explainers.consensus.ConsensusExplainer]).
- **[`explanation`](explanation.md)**: Output containers [`Explanation`][xai_framework.explanation.Explanation] and [`ConsensusExplanation`][xai_framework.explanation.ConsensusExplanation].
- **[`model`](model.md)**: Standardized wrapper [`ModelAdapter`][xai_framework.model.ModelAdapter] providing uniform inference over estimators, pipelines, and callables.
- **[`registry`](registry.md)**: Explainer registry, dynamic discovery, and entry-point plugin mechanics.
