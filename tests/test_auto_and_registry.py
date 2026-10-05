import numpy as np
import pandas as pd
import pytest
from xai_framework import (
    BaseExplainer,
    ConsensusExplanation,
    Explanation,
    available_explainers,
    build_explainer,
    explain,
    explain_global,
    get_explainer,
    register_explainer,
    summarize,
)
from xai_framework.auto import resolve_methods


def test_registry_lists_builtins():
    info = available_explainers()
    assert {"surrogate", "coalition", "permutation", "consensus"} <= set(info)
    assert info["surrogate"]["local"] and not info["surrogate"]["global"]
    assert info["permutation"]["global"] and not info["permutation"]["local"]
    assert get_explainer("surrogate").name == "surrogate"
    with pytest.raises(KeyError):
        get_explainer("does-not-exist")


def test_custom_explainer_registration(linreg, regression):
    X, _ = regression

    @register_explainer("constant")
    class ConstantExplainer(BaseExplainer):
        def _explain_instance(self, x, target):
            return self._make_explanation(np.ones(len(x)), x, target)

    assert get_explainer("constant") is ConstantExplainer
    exp = explain(linreg, X, instance=X[0], methods="constant")
    assert exp.method == "constant"
    np.testing.assert_array_equal(exp.values, 1.0)


def test_resolve_methods():
    assert resolve_methods("auto", scope="local", has_y=False) == ["coalition", "surrogate"]
    assert resolve_methods("auto", scope="global", has_y=False) == ["coalition"]
    assert resolve_methods("auto", scope="global", has_y=True) == ["coalition", "permutation"]
    assert resolve_methods("surrogate", scope="local", has_y=False) == ["surrogate"]
    assert resolve_methods(["surrogate", "coalition"], scope="local", has_y=False) == [
        "surrogate",
        "coalition",
    ]


def test_explain_local_returns_consensus(iris_rf, iris):
    X, _ = iris
    exp = explain(
        iris_rf, X, instance=0, random_state=0, explainer_kwargs={"surrogate": {"n_samples": 500}}
    )
    assert isinstance(exp, ConsensusExplanation)
    assert exp.methods == ["coalition", "surrogate"]
    np.testing.assert_array_equal(exp.feature_values, X.iloc[0].to_numpy())


def test_explain_single_method_returns_plain_explanation(linreg, regression):
    X, _ = regression
    exp = explain(linreg, X, instance=X[0], methods="surrogate", random_state=0)
    assert type(exp) is Explanation
    assert exp.method == "surrogate"


def test_explain_global_with_and_without_y(linreg, regression):
    X, y = regression
    only_coalition = explain_global(linreg, X, random_state=0)
    assert only_coalition.method == "coalition"
    both = explain_global(
        linreg, X, y, random_state=0, explainer_kwargs={"permutation": {"n_repeats": 1}}
    )
    assert isinstance(both, ConsensusExplanation)
    assert set(both.components) == {"coalition", "permutation"}
    assert both.top(1)[0][0] == "f0"


def test_build_explainer_without_running(iris_rf, iris):
    X, _ = iris
    explainer = build_explainer(iris_rf, X, methods="surrogate")
    assert explainer.name == "surrogate"


def test_feature_and_class_names_passthrough(binary_gb, binary):
    X, _ = binary
    exp = explain(
        binary_gb,
        X,
        instance=X[0],
        methods="coalition",
        feature_names=["alpha", "beta", "gamma"],
        class_names=["no", "yes"],
        random_state=0,
    )
    assert exp.feature_names == ["alpha", "beta", "gamma"]
    assert exp.target in {"no", "yes"}


def test_explain_multiple_instances_dataframe(iris_rf, iris):
    X, _ = iris

    explanations = explain(
        iris_rf,
        X,
        instances=X.iloc[:3],
        methods="coalition",
        random_state=0,
    )

    assert isinstance(explanations, list)
    assert len(explanations) == 3
    assert all(isinstance(exp, Explanation) for exp in explanations)


def test_explain_multiple_instances_numpy(iris_rf, iris):
    X, _ = iris

    explanations = explain(
        iris_rf,
        X,
        instances=X.iloc[:3].to_numpy(),
        methods="coalition",
        random_state=0,
    )

    assert len(explanations) == 3
    np.testing.assert_array_equal(
        explanations[0].feature_values,
        X.iloc[0].to_numpy(),
    )


def test_explain_multiple_instances_indices(iris_rf, iris):
    X, _ = iris

    explanations = explain(
        iris_rf,
        X,
        instances=[0, 2, 4],
        methods="coalition",
        random_state=0,
    )

    assert len(explanations) == 3

    np.testing.assert_array_equal(
        explanations[0].feature_values,
        X.iloc[0].to_numpy(),
    )

    np.testing.assert_array_equal(
        explanations[1].feature_values,
        X.iloc[2].to_numpy(),
    )

    np.testing.assert_array_equal(
        explanations[2].feature_values,
        X.iloc[4].to_numpy(),
    )


def test_explain_rejects_instance_and_instances(iris_rf, iris):
    X, _ = iris

    with pytest.raises(ValueError, match="either 'instance' or 'instances'"):
        explain(
            iris_rf,
            X,
            instance=0,
            instances=[1, 2],
            methods="coalition",
        )


def test_summarize_batch_explanations(iris_rf, iris):
    X, _ = iris

    explanations = explain(
        iris_rf,
        X,
        instances=X.iloc[:3],
        random_state=0,
    )

    summary = summarize(explanations)

    assert isinstance(summary, pd.DataFrame)
    assert len(summary) == 3

    assert "prediction" in summary.columns
    assert "target" in summary.columns
    assert "feature_1" in summary.columns
    assert "attribution_1" in summary.columns
    assert "feature_2" in summary.columns
    assert "attribution_2" in summary.columns
    assert "feature_3" in summary.columns
    assert "attribution_3" in summary.columns
    assert "agreement" in summary.columns
