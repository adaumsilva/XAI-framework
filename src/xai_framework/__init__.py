"""XAI-Framework: model-agnostic explanations from several methods, combined.

Quick start::

    from xai_framework import explain

    exp = explain(model, X_train, instance=X_test.iloc[0])
    print(exp.to_text())
    exp.plot()
"""

from .auto import build_explainer, explain, explain_global
from .summary import summarize
from .explainers import (
    BaseExplainer,
    CoalitionExplainer,
    ConsensusExplainer,
    LocalSurrogateExplainer,
    PermutationExplainer,
    combine_explanations,
)
from .explanation import ConsensusExplanation, Explanation
from .model import ModelAdapter
from .registry import available_explainers, get_explainer, register_explainer

__version__ = "0.1.0"

__all__ = [
    "BaseExplainer",
    "CoalitionExplainer",
    "ConsensusExplainer",
    "ConsensusExplanation",
    "Explanation",
    "LocalSurrogateExplainer",
    "ModelAdapter",
    "PermutationExplainer",
    "__version__",
    "available_explainers",
    "build_explainer",
    "combine_explanations",
    "explain",
    "explain_global",
    "get_explainer",
    "register_explainer",
    "summarize",
]
