"""Executable smoke tests for documentation code examples."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from xai_framework import (
    BaseExplainer,
    CoalitionExplainer,
    Explanation,
    LocalSurrogateExplainer,
    PermutationExplainer,
    explain,
    explain_global,
    register_explainer,
)


@pytest.fixture(scope="module")
def iris_data_and_model():
    iris = load_iris(as_frame=True)
    X, y = iris.data, iris.target
    model = RandomForestClassifier(n_estimators=10, random_state=42).fit(X, y)
    return X, y, model


@pytest.fixture(scope="module")
def cancer_data_and_model():
    cancer = load_breast_cancer(as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        cancer.data, cancer.target, test_size=0.25, random_state=42
    )
    model = RandomForestClassifier(n_estimators=10, random_state=42).fit(X_train, y_train)
    return X_train, X_test, y_train, y_test, model


def test_quickstart_index_example(cancer_data_and_model):
    X_train, X_test, y_train, y_test, model = cancer_data_and_model
    exp = explain(model, X_train, instance=X_test.iloc[0])
    text = exp.to_text()
    assert isinstance(text, str)
    assert len(text) > 0


def test_getting_started_local_and_global(iris_data_and_model):
    X, y, model = iris_data_and_model

    # Local explanation
    exp = explain(model, X, instance=X.iloc[0])
    top_3 = exp.top(3)
    assert len(top_3) == 3
    assert exp.agreement_level() in ("high", "moderate", "low")
    assert isinstance(exp.agreement, float)

    df = exp.to_dataframe()
    assert not df.empty
    md = exp.to_markdown()
    assert "|" in md
    d = exp.to_dict()
    assert "features" in d
    assert "agreement" in d

    # Global explanation
    glob = explain_global(model, X, y=y)
    assert glob.scope == "global"
    assert len(glob.to_text()) > 0

    # Custom methods & aggregation
    exp_surr = explain(model, X, instance=X.iloc[0], methods="surrogate")
    assert exp_surr.method == "surrogate"

    exp_rank = explain(
        model,
        X,
        instance=X.iloc[0],
        methods=["coalition", "surrogate"],
        weights={"coalition": 2.0, "surrogate": 1.0},
        aggregation="rank",
    )
    assert exp_rank.method == "consensus"
    assert exp_rank.metadata["aggregation"] == "rank"


def test_coalition_explainer_example(cancer_data_and_model):
    X_train, X_test, _, _, model = cancer_data_and_model
    explainer = CoalitionExplainer(model, X_train, n_background=20, random_state=42)
    exp = explainer.explain_instance(X_test.iloc[0])
    assert exp.scope == "local"
    assert len(exp.values) == X_train.shape[1]

    glob_exp = explainer.explain_global(X_test.iloc[:10])
    assert glob_exp.scope == "global"


def test_surrogate_explainer_example(cancer_data_and_model):
    X_train, X_test, _, _, model = cancer_data_and_model
    explainer = LocalSurrogateExplainer(
        model=model,
        X_background=X_train,
        n_samples=200,
        kernel_width=None,
        random_state=42,
    )
    exp = explainer.explain_instance(X_test.iloc[0])
    assert exp.metadata.get("local_r2") is not None
    assert "coefficients" in exp.metadata


def test_permutation_explainer_example(cancer_data_and_model):
    _, X_test, _, y_test, model = cancer_data_and_model
    explainer = PermutationExplainer(
        model=model,
        X_background=X_test,
        metric="log_loss",
        n_repeats=3,
        random_state=42,
    )
    glob_exp = explainer.explain_global(X_test, y=y_test)
    assert glob_exp.scope == "global"
    assert len(glob_exp.top(5)) == 5
    assert "importances_std" in glob_exp.metadata

    # Custom metric callable example
    custom_expl = PermutationExplainer(
        model=model,
        X_background=X_test,
        metric=lambda y_true, y_prob: roc_auc_score(y_true, y_prob[:, 1]),
        n_repeats=3,
        random_state=42,
    )
    custom_exp = custom_expl.explain_global(X_test, y=y_test)
    assert custom_exp.scope == "global"


def test_custom_gradient_explainer_example(iris_data_and_model):
    """Smoke test the exact custom explainer code published in docs/extending.md."""
    X, _, model = iris_data_and_model

    @register_explainer("custom_gradient_smoke")
    class CustomGradientSmokeExplainer(BaseExplainer):
        name = "custom_gradient_smoke"
        supports_local = True
        supports_global = False
        requires_y = False

        def _explain_instance(self, x: np.ndarray, target: int | None) -> Explanation:
            base_pred = float(self._predict_scalar(x.reshape(1, -1), target)[0])
            eps = 1e-4
            n_features = len(x)
            attributions = np.zeros(n_features)

            for i in range(n_features):
                x_perturbed = x.copy()
                x_perturbed[i] += eps
                pred_perturbed = float(self._predict_scalar(x_perturbed.reshape(1, -1), target)[0])
                attributions[i] = (pred_perturbed - base_pred) / eps

            return self._make_explanation(
                values=attributions,
                x=x,
                target=target,
                prediction=base_pred,
                base_value=None,
                step_size=eps,
            )

    expl = CustomGradientSmokeExplainer(model, X)
    exp = expl.explain_instance(X.iloc[0])
    assert exp.method == "custom_gradient_smoke"
    assert exp.metadata.get("step_size") == 1e-4
    assert len(exp.values) == X.shape[1]

    # Consensus with the new explainer
    combined = explain(model, X, instance=X.iloc[0], methods=["coalition", "custom_gradient_smoke"])
    assert combined.method == "consensus"
    assert "custom_gradient_smoke" in combined.components
