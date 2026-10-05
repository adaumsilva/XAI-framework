"""One-call explanations: ``explain(model, X, instance)``.

Picks sensible explainers for the model automatically and returns a consensus
explanation when more than one applies. This is the entry point most users want;
the explainer classes remain available for fine control.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .explainers.base import BaseExplainer
from .explainers.consensus import ConsensusExplainer
from .explanation import Explanation
from .model import ModelAdapter, to_numpy
from .registry import available_explainers, get_explainer

DEFAULT_LOCAL_METHODS: tuple[str, ...] = ("coalition", "surrogate")
DEFAULT_GLOBAL_METHODS: tuple[str, ...] = ("coalition", "permutation")


def resolve_methods(methods: str | Sequence[str], *, scope: str, has_y: bool) -> list[str]:
    """Turn ``"auto"`` (or an explicit list) into the explainers that can actually run."""
    if isinstance(methods, str) and methods != "auto":
        return [methods]
    if methods != "auto":
        return list(methods)

    info = available_explainers()
    wanted = DEFAULT_LOCAL_METHODS if scope == "local" else DEFAULT_GLOBAL_METHODS
    chosen = []
    for name in wanted:
        meta = info.get(name)
        if meta is None or not meta["available"]:
            continue
        if scope == "global" and meta["class"].requires_y and not has_y:
            continue
        chosen.append(name)
    if not chosen:
        raise RuntimeError("no explainer is available for this request")
    return chosen


def build_explainer(
    model: Any,
    X: Any,
    methods: str | Sequence[str] = "auto",
    *,
    scope: str = "local",
    has_y: bool = False,
    feature_names: list[str] | None = None,
    class_names: list[Any] | None = None,
    task: str | None = None,
    random_state: int | None = None,
    weights: dict[str, float] | None = None,
    aggregation: str = "mean",
    explainer_kwargs: dict[str, dict[str, Any]] | None = None,
) -> BaseExplainer:
    """Construct the explainer :func:`explain` would use, without running it."""
    adapter = (
        model
        if isinstance(model, ModelAdapter)
        else ModelAdapter(model, task=task, feature_names=feature_names, class_names=class_names)
    )
    names = resolve_methods(methods, scope=scope, has_y=has_y)
    if len(names) == 1:
        cls = get_explainer(names[0])
        return cls(
            adapter, X, random_state=random_state, **(explainer_kwargs or {}).get(names[0], {})
        )
    return ConsensusExplainer(
        adapter,
        X,
        methods=names,
        weights=weights,
        aggregation=aggregation,  # type: ignore[arg-type]
        explainer_kwargs=explainer_kwargs,
        random_state=random_state,
    )

def explain(
    model: Any,
    X: Any,
    instance: Any = None,
    *,
    instances: Any = None,
    methods: str | Sequence[str] = "auto",
    target: Any = None,
    y: Any = None,
    feature_names: list[str] | None = None,
    class_names: list[Any] | None = None,
    task: str | None = None,
    random_state: int | None = None,
    weights: dict[str, float] | None = None,
    aggregation: str = "mean",
    explainer_kwargs: dict[str, dict[str, Any]] | None = None,
) -> Explanation | list[Explanation]:
    """Explain a model with one call.

    Parameters
    ----------
    model
        A fitted scikit-learn style estimator, a tree-boosting model, a Pipeline,
        or a callable ``f(X) -> predictions``.
    X
        Background data (training rows work well). A DataFrame supplies feature names.
    instance
        One row to explain: an array/Series, or an integer index into ``X``.
        Kept for backward compatibility.
    instances
        Multiple rows to explain. Accepts a DataFrame, a 2-D array, or a list
        of integer row indices into ``X``.
    methods
        ``"auto"`` picks ``coalition + surrogate`` locally and
        ``coalition + permutation`` globally.
    target
        Class to explain for classifiers. Defaults to the predicted class.
    y
        Ground-truth labels for ``X``; enables permutation importance globally.
    weights, aggregation, explainer_kwargs
        Forwarded to :class:`~xai_framework.explainers.consensus.ConsensusExplainer`.

    Returns
    -------
    Explanation or list[Explanation]
        A single explanation for ``instance`` or a list of explanations for
        ``instances``.
    """

    if instance is not None and instances is not None:
        raise ValueError("provide either 'instance' or 'instances', not both")

    # Multiple local instances
    if instances is not None:
        explainer = build_explainer(
            model,
            X,
            methods,
            scope="local",
            has_y=y is not None,
            feature_names=feature_names,
            class_names=class_names,
            task=task,
            random_state=random_state,
            weights=weights,
            aggregation=aggregation,
            explainer_kwargs=explainer_kwargs,
        )

        # A list of integers means row indices into X.
        if isinstance(instances, (list, tuple)) and all(
            isinstance(i, (int, np.integer)) and not isinstance(i, bool)
            for i in instances
        ):
            selected = to_numpy(X)[list(instances)]
            return explainer.explain(selected, target=target)

        # Otherwise instances is expected to be a DataFrame or 2-D array.
        return explainer.explain(instances, target=target)

    # Existing single/global behavior
    scope = "global" if instance is None else "local"

    explainer = build_explainer(
        model,
        X,
        methods,
        scope=scope,
        has_y=y is not None,
        feature_names=feature_names,
        class_names=class_names,
        task=task,
        random_state=random_state,
        weights=weights,
        aggregation=aggregation,
        explainer_kwargs=explainer_kwargs,
    )

    if scope == "global":
        return explainer.explain_global(X, y, target=target)

    if isinstance(instance, (int, np.integer)) and not isinstance(instance, bool):
        instance = to_numpy(X)[int(instance)]

    return explainer.explain_instance(instance, target=target)


def explain_global(model: Any, X: Any, y: Any = None, **kwargs: Any) -> Explanation:
    """Shorthand for ``explain(model, X, instance=None, y=y, ...)``."""
    return explain(model, X, None, y=y, **kwargs)
