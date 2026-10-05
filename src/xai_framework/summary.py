from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd

from .explanation import ConsensusExplanation, Explanation


def summarize(explanations: Sequence[Explanation]) -> pd.DataFrame:
    """Summarize a collection of explanations as a DataFrame.

    Each row represents one explanation and contains its prediction,
    target, top-3 features with their attributions, and agreement
    when the explanation is a consensus explanation.
    """
    rows: list[dict[str, Any]] = []

    for explanation in explanations:
        top_features = explanation.top(3)

        row: dict[str, Any] = {
            "prediction": explanation.prediction,
            "target": explanation.target,
        }

        for i in range(3):
            if i < len(top_features):
                feature, attribution = top_features[i]
                row[f"feature_{i + 1}"] = feature
                row[f"attribution_{i + 1}"] = attribution
            else:
                row[f"feature_{i + 1}"] = None
                row[f"attribution_{i + 1}"] = None

        if isinstance(explanation, ConsensusExplanation):
            row["agreement"] = explanation.agreement
        else:
            row["agreement"] = float("nan")

        rows.append(row)

    return pd.DataFrame(rows)