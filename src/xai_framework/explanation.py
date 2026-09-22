"""The :class:`Explanation` container returned by every explainer.

Whatever method produced it (coalition, surrogate, permutation importance, or a
consensus of several), an explanation is always the same shape: one attribution per feature plus
enough context to read it - the instance, the prediction, the baseline and the
method. That common shape is what lets the framework compare and combine methods.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

Scope = Literal["local", "global"]


def normalize_attributions(values: np.ndarray) -> np.ndarray:
    """Scale attributions to unit L1 norm so different methods become comparable.

    All-zero vectors are returned unchanged rather than producing NaNs.
    """
    values = np.asarray(values, dtype=float)
    total = np.abs(values).sum()
    return values / total if total > 0 else values.copy()


def rank_by_magnitude(values: np.ndarray) -> np.ndarray:
    """1-based rank of each feature by ``|value|`` (1 = most important)."""
    order = np.argsort(-np.abs(np.asarray(values, dtype=float)), kind="stable")
    ranks = np.empty(len(order), dtype=int)
    ranks[order] = np.arange(1, len(order) + 1)
    return ranks


@dataclass
class Explanation:
    """Feature attributions for one instance (``scope="local"``) or a dataset (``"global"``).

    Attributes
    ----------
    feature_names
        One name per feature, in model input order.
    values
        Attribution per feature. Positive pushes the prediction up (towards the
        target class for classifiers), negative pushes it down. For global
        explanations this is an importance score whose sign is method-specific.
    method
        Name of the explainer that produced this (``"coalition"``, ``"surrogate"``, ...).
    scope
        ``"local"`` or ``"global"``.
    feature_values
        The explained instance (local only).
    base_value
        The reference prediction the attributions are measured from, when the
        method provides one (mean background prediction, surrogate intercept).
    prediction
        Model output for the instance (probability of ``target`` or regression value).
    target
        Class label being explained (classification only).
    task
        ``"classification"`` or ``"regression"``.
    metadata
        Free-form, method-specific extras (e.g. the surrogate's local R^2).
    """

    feature_names: list[str]
    values: np.ndarray
    method: str
    scope: Scope = "local"
    feature_values: np.ndarray | None = None
    base_value: float | None = None
    prediction: float | None = None
    target: Any = None
    task: str = "classification"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.feature_names = [str(f) for f in self.feature_names]
        self.values = np.asarray(self.values, dtype=float).reshape(-1)
        if len(self.values) != len(self.feature_names):
            raise ValueError(
                f"{len(self.values)} attributions for {len(self.feature_names)} feature names"
            )
        if self.feature_values is not None:
            self.feature_values = np.asarray(self.feature_values).reshape(-1)
            if len(self.feature_values) != len(self.feature_names):
                raise ValueError("feature_values must have one entry per feature")
        if self.base_value is not None:
            self.base_value = float(self.base_value)
        if self.prediction is not None:
            self.prediction = float(self.prediction)

    # ------------------------------------------------------------------ basic views
    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    @property
    def is_local(self) -> bool:
        return self.scope == "local"

    def ranks(self) -> np.ndarray:
        """1-based importance rank per feature (1 = largest ``|value|``)."""
        return rank_by_magnitude(self.values)

    def normalized(self) -> np.ndarray:
        """Attributions scaled to unit L1 norm (see :func:`normalize_attributions`)."""
        return normalize_attributions(self.values)

    def top(self, k: int = 5) -> list[tuple[str, float]]:
        """The ``k`` most important ``(feature, value)`` pairs, largest ``|value|`` first."""
        order = np.argsort(-np.abs(self.values), kind="stable")[:k]
        return [(self.feature_names[i], float(self.values[i])) for i in order]

    def as_series(self) -> pd.Series:
        return pd.Series(self.values, index=self.feature_names, name=self.method)

    def to_dataframe(self) -> pd.DataFrame:
        """One row per feature, sorted by importance."""
        df = pd.DataFrame(
            {
                "feature": self.feature_names,
                "value": self.values,
                "abs_value": np.abs(self.values),
                "rank": self.ranks(),
            }
        )
        if self.feature_values is not None:
            df.insert(1, "feature_value", self.feature_values)
        return df.sort_values("rank").reset_index(drop=True)

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly representation."""
        return {
            "method": self.method,
            "scope": self.scope,
            "task": self.task,
            "target": _jsonable(self.target),
            "prediction": self.prediction,
            "base_value": self.base_value,
            "features": [
                {
                    "name": name,
                    "value": float(v),
                    **(
                        {"feature_value": _jsonable(self.feature_values[i])}
                        if self.feature_values is not None
                        else {}
                    ),
                }
                for i, (name, v) in enumerate(zip(self.feature_names, self.values, strict=True))
            ],
            "metadata": {k: _jsonable(v) for k, v in self.metadata.items()},
        }

    # ------------------------------------------------------------------ narration
    def to_text(self, k: int = 5) -> str:
        """A short plain-language summary suitable for logs, notebooks or end users."""
        lines = [self._headline()]
        pos = [(f, v) for f, v in self.top(k) if v > 0]
        neg = [(f, v) for f, v in self.top(k) if v < 0]
        verb_up, verb_down = self._direction_verbs()
        if pos:
            lines.append(f"Features that {verb_up}:")
            lines.extend(f"  + {self._feature_line(f, v)}" for f, v in pos)
        if neg:
            lines.append(f"Features that {verb_down}:")
            lines.extend(f"  - {self._feature_line(f, v)}" for f, v in neg)
        if not pos and not neg:
            lines.append("  (all attributions are zero)")
        extra = self._extra_text()
        if extra:
            lines.append(extra)
        return "\n".join(lines)

    def _headline(self) -> str:
        if self.scope == "global":
            return f"Global feature importance ({self.method}):"
        if self.task == "classification":
            head = f"Predicted class {self.target!r}"
            if self.prediction is not None:
                head += f" with probability {self.prediction:.3f}"
        else:
            head = "Predicted value"
            if self.prediction is not None:
                head += f" {self.prediction:.4g}"
        head += f" ({self.method}"
        if self.base_value is not None:
            head += f", baseline {self.base_value:.3f}"
        return head + ")."

    def _direction_verbs(self) -> tuple[str, str]:
        if self.scope == "global":
            return "matter most", "have negative importance"
        if self.task == "classification":
            return f"push towards {self.target!r}", f"push away from {self.target!r}"
        return "increase the prediction", "decrease the prediction"

    def _feature_line(self, name: str, value: float) -> str:
        if self.feature_values is not None:
            fv = self.feature_values[self.feature_names.index(name)]
            return f"{name} = {_fmt(fv)}  ({value:+.4g})"
        return f"{name}  ({value:+.4g})"

    def _extra_text(self) -> str:
        r2 = self.metadata.get("local_r2")
        return f"Local surrogate fit R^2 = {r2:.2f}." if r2 is not None else ""

    def to_markdown(self, k: int = 10) -> str:
        """Render the explanation as a Markdown table.

        Parameters
        ----------
        k
            Maximum number of top features to include, ordered by importance
            (largest ``|value|`` first). Default 10.

        Returns
        -------
        str
            A Markdown string with a headline, table, and optional agreement/extra summary.
        """
        order = np.argsort(-np.abs(self.values), kind="stable")[:k]
        has_values = self.feature_values is not None

        headers = ["feature"]
        if has_values:
            headers.append("value")
        headers.append("attribution")

        comp_headers, comp_data = self._markdown_component_data()
        headers.extend(comp_headers)

        rows: list[list[str]] = []
        for i in order:
            name = self.feature_names[i]
            val = self.values[i]
            row = [name]
            if has_values:
                assert self.feature_values is not None
                row.append(_fmt(self.feature_values[i]))
            row.append(f"{val:+.4g}")
            for comp_name in comp_headers:
                row.append(comp_data[comp_name].get(name, "0"))
            rows.append(row)

        lines = [
            self._headline(),
            "",
            "| " + " | ".join(_escape_markdown(h) for h in headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
        ]
        for row in rows:
            lines.append("| " + " | ".join(_escape_markdown(c) for c in row) + " |")

        extra = self._extra_text()
        if extra:
            lines.extend(["", extra])

        return "\n".join(lines)

    def _markdown_component_data(self) -> tuple[list[str], dict[str, dict[str, str]]]:
        return [], {}

    # ------------------------------------------------------------------ plotting
    def plot(
        self, k: int = 10, ax: Any = None, show: bool = False, title: str | None = None
    ) -> Any:
        """Horizontal bar chart of the top-``k`` attributions (requires matplotlib)."""
        from .plotting import plot_explanation

        return plot_explanation(self, k=k, ax=ax, show=show, title=title)

    def __repr__(self) -> str:
        head = ", ".join(f"{f}={v:+.3g}" for f, v in self.top(3))
        return f"<Explanation {self.method} {self.scope} target={self.target!r} top: {head}>"


@dataclass
class ConsensusExplanation(Explanation):
    """An explanation built by combining several component explanations.

    ``values`` holds the combined attribution (weighted mean of each component's
    L1-normalised attributions, or a Borda rank score - see
    :class:`~xai_framework.explainers.consensus.ConsensusExplainer`).

    Attributes
    ----------
    components
        The individual explanations keyed by method name.
    agreement
        Mean pairwise Spearman rank correlation between the components' importance
        orderings, in ``[-1, 1]``. ``nan`` when there is only one component.
    agreement_matrix
        Pairwise rank correlations, methods x methods.
    """

    components: dict[str, Explanation] = field(default_factory=dict)
    agreement: float = float("nan")
    agreement_matrix: pd.DataFrame | None = None

    @property
    def methods(self) -> list[str]:
        return list(self.components)

    def agreement_level(self) -> str:
        """``"high"``, ``"moderate"``, ``"low"`` or ``"n/a"`` from :attr:`agreement`."""
        if np.isnan(self.agreement):
            return "n/a"
        if self.agreement >= 0.8:
            return "high"
        if self.agreement >= 0.5:
            return "moderate"
        return "low"

    def to_dataframe(self) -> pd.DataFrame:
        df = super().to_dataframe().rename(columns={"value": "consensus"})
        for name, comp in self.components.items():
            series = pd.Series(comp.normalized(), index=comp.feature_names)
            df[name] = series.reindex(df["feature"]).to_numpy()
        return df

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["agreement"] = None if np.isnan(self.agreement) else float(self.agreement)
        d["agreement_level"] = self.agreement_level()
        d["components"] = {name: comp.to_dict() for name, comp in self.components.items()}
        return d

    def _extra_text(self) -> str:
        level = self.agreement_level()
        methods = " + ".join(self.methods)
        if level == "n/a":
            if len(self.components) == 1:
                return f"Single method ({methods}); no cross-method agreement to report."
            flat = [n for n, c in self.components.items() if np.all(c.values == c.values[0])]
            return (
                f"Agreement between {methods} is undefined: "
                f"{', '.join(flat) or 'a method'} produced constant attributions."
            )
        hint = {
            "high": "the ranking is reliable",
            "moderate": "the top features are reliable, lower ranks less so",
            "low": "treat this ranking with caution and inspect each method",
        }[level]
        return f"Agreement between {methods}: {level} (rho = {self.agreement:.2f}) - {hint}."

    def _markdown_component_data(self) -> tuple[list[str], dict[str, dict[str, str]]]:
        headers = list(self.components.keys())
        data: dict[str, dict[str, str]] = {name: {} for name in headers}
        for name, comp in self.components.items():
            norm = comp.normalized()
            for feat, val in zip(comp.feature_names, norm, strict=False):
                data[name][feat] = f"{val:+.4g}"
        return headers, data

    def __repr__(self) -> str:
        head = ", ".join(f"{f}={v:+.3g}" for f, v in self.top(3))
        return (
            f"<ConsensusExplanation [{'+'.join(self.methods)}] {self.scope} "
            f"agreement={self.agreement:.2f} top: {head}>"
        )


def _fmt(value: Any) -> str:
    if isinstance(value, (float, np.floating)):
        return f"{value:.4g}"
    return str(value)


def _escape_markdown(text: Any) -> str:
    """Escape text for safe inclusion in Markdown table headers and cells.

    Escapes backslashes, table delimiter pipes, and converts embedded
    newlines into '<br>' so table rows do not break.
    """
    s = str(text)
    s = s.replace("\\", "\\\\")
    s = s.replace("|", "\\|")
    return s.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "<br>")


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.DataFrame):
        return value.to_dict()
    return value
