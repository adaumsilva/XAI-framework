import json

import numpy as np
import pytest

from xai_framework import ConsensusExplanation, Explanation
from xai_framework.explanation import normalize_attributions, rank_by_magnitude


def make(values, **kw):
    names = [f"f{i}" for i in range(len(values))]
    return Explanation(feature_names=names, values=np.array(values, dtype=float), method="t", **kw)


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        Explanation(feature_names=["a"], values=np.array([1.0, 2.0]), method="t")


def test_ranks_and_top():
    exp = make([0.1, -0.9, 0.5])
    np.testing.assert_array_equal(exp.ranks(), [3, 1, 2])
    assert exp.top(2) == [("f1", -0.9), ("f2", 0.5)]


def test_normalized_unit_l1():
    exp = make([2.0, -2.0, 4.0])
    np.testing.assert_allclose(np.abs(exp.normalized()).sum(), 1.0)
    np.testing.assert_allclose(normalize_attributions(np.zeros(3)), np.zeros(3))


def test_rank_by_magnitude_is_stable_for_ties():
    np.testing.assert_array_equal(rank_by_magnitude(np.array([1.0, 1.0, 2.0])), [2, 3, 1])


def test_dataframe_sorted_by_rank_with_feature_values():
    exp = make([0.1, -0.9, 0.5], feature_values=np.array([1, 2, 3]))
    df = exp.to_dataframe()
    assert list(df["feature"]) == ["f1", "f2", "f0"]
    assert list(df.columns) == ["feature", "feature_value", "value", "abs_value", "rank"]


def test_to_dict_is_json_serialisable():
    exp = make(
        [0.1, -0.9],
        feature_values=np.array([1.5, 2.5]),
        target=np.int64(1),
        prediction=np.float32(0.7),
        metadata={"arr": np.arange(2), "scalar": np.float64(1.0)},
    )
    payload = json.dumps(exp.to_dict())
    assert '"target": 1' in payload


def test_text_mentions_direction_and_values():
    exp = make([0.3, -0.2, 0.0], feature_values=np.array([1, 2, 3]), target="yes", prediction=0.8)
    text = exp.to_text()
    assert "Predicted class 'yes' with probability 0.800" in text
    assert "push towards 'yes'" in text
    assert "push away from 'yes'" in text
    assert "f0 = 1" in text


def test_text_for_regression_and_global():
    local = make([1.0, -1.0], task="regression", prediction=42.0)
    assert "Predicted value 42" in local.to_text()
    assert "increase the prediction" in local.to_text()
    glob = make([1.0, 0.5], scope="global")
    assert "Global feature importance" in glob.to_text()


def test_consensus_agreement_levels():
    comps = {"a": make([1.0, 0.5]), "b": make([0.9, 0.6])}
    for rho, level in [(0.9, "high"), (0.6, "moderate"), (0.1, "low"), (float("nan"), "n/a")]:
        exp = ConsensusExplanation(
            feature_names=["f0", "f1"],
            values=np.array([1.0, 0.5]),
            method="consensus",
            components=comps,
            agreement=rho,
        )
        assert exp.agreement_level() == level
        assert level in exp.to_text() or level == "n/a"


def test_consensus_dataframe_has_component_columns():
    comps = {"a": make([1.0, 0.5]), "b": make([0.9, 0.6])}
    exp = ConsensusExplanation(
        feature_names=["f0", "f1"],
        values=np.array([1.0, 0.5]),
        method="consensus",
        components=comps,
    )
    df = exp.to_dataframe()
    assert {"consensus", "a", "b"} <= set(df.columns)
    d = exp.to_dict()
    assert set(d["components"]) == {"a", "b"}
    assert d["agreement"] is None


def test_to_markdown_local_classification():
    exp = make(
        [0.3, -0.9, 0.0],
        feature_values=np.array([1, 2, 3]),
        target="yes",
        prediction=0.8,
        base_value=0.5,
    )
    md = exp.to_markdown()
    lines = md.splitlines()

    assert lines[0] == "Predicted class 'yes' with probability 0.800 (t, baseline 0.500)."
    assert lines[1] == ""
    assert lines[2] == "| feature | value | attribution |"
    assert lines[3] == "| --- | --- | --- |"
    # Ranked by magnitude: f1 (-0.9), f0 (0.3), f2 (0.0)
    assert lines[4] == "| f1 | 2 | -0.9 |"
    assert lines[5] == "| f0 | 1 | +0.3 |"
    assert lines[6] == "| f2 | 3 | +0 |"


def test_to_markdown_k_limits_rows():
    exp = make([0.1, -0.9, 0.5], feature_values=np.array([1, 2, 3]))
    md = exp.to_markdown(k=2)
    lines = md.splitlines()

    assert lines[2] == "| feature | value | attribution |"
    assert lines[3] == "| --- | --- | --- |"
    # Only top 2 features rendered
    assert len(lines) == 6
    assert lines[4] == "| f1 | 2 | -0.9 |"
    assert lines[5] == "| f2 | 3 | +0.5 |"


def test_to_markdown_global_omits_value_column():
    glob = make([1.0, -0.5], scope="global")
    md = glob.to_markdown()
    lines = md.splitlines()

    assert lines[0] == "Global feature importance (t):"
    assert lines[1] == ""
    assert lines[2] == "| feature | attribution |"
    assert lines[3] == "| --- | --- |"
    assert lines[4] == "| f0 | +1 |"
    assert lines[5] == "| f1 | -0.5 |"


def test_to_markdown_regression():
    reg = make([1.5, -2.0], task="regression", prediction=42.0, feature_values=np.array([10, 20]))
    md = reg.to_markdown()
    assert "Predicted value 42 (t)." in md
    assert "| feature | value | attribution |" in md
    assert "| f1 | 20 | -2 |" in md


def test_to_markdown_consensus_includes_component_columns_and_agreement():
    comps = {"coalition": make([1.0, 0.5]), "surrogate": make([0.9, 0.6])}
    exp = ConsensusExplanation(
        feature_names=["f0", "f1"],
        values=np.array([1.0, 0.5]),
        method="consensus",
        scope="local",
        feature_values=np.array([10, 20]),
        components=comps,
        agreement=0.9,
    )
    md = exp.to_markdown()
    lines = md.splitlines()

    assert lines[2] == "| feature | value | attribution | coalition | surrogate |"
    assert lines[3] == "| --- | --- | --- | --- | --- |"
    # coalition normalized: [1.0/1.5, 0.5/1.5] -> [+0.6667, +0.3333]
    # surrogate normalized: [0.9/1.5, 0.6/1.5] -> [+0.6, +0.4]
    assert lines[4] == "| f0 | 10 | +1 | +0.6667 | +0.6 |"
    assert lines[5] == "| f1 | 20 | +0.5 | +0.3333 | +0.4 |"
    assert lines[6] == ""
    assert (
        lines[7]
        == "Agreement between coalition + surrogate: high (rho = 0.90) - the ranking is reliable."
    )


def test_to_markdown_global_consensus():
    comps = {"a": make([1.0, 0.5]), "b": make([0.5, 1.0])}
    exp = ConsensusExplanation(
        feature_names=["f0", "f1"],
        values=np.array([1.5, 1.5]),
        method="consensus",
        scope="global",
        components=comps,
        agreement=0.5,
    )
    md = exp.to_markdown()
    lines = md.splitlines()

    assert lines[0] == "Global feature importance (consensus):"
    assert lines[2] == "| feature | attribution | a | b |"
    assert lines[3] == "| --- | --- | --- | --- |"
    assert "Agreement between a + b: moderate (rho = 0.50)" in lines[-1]


def test_to_markdown_surrogate_local_r2():
    exp = make([0.5, -0.5], feature_values=np.array([1, 2]), metadata={"local_r2": 0.94})
    md = exp.to_markdown()
    assert "Local surrogate fit R^2 = 0.94." in md


def test_escape_markdown_helper():
    from xai_framework.explanation import _escape_markdown

    assert _escape_markdown("simple") == "simple"
    assert _escape_markdown("pipe|char") == "pipe\\|char"
    assert _escape_markdown("back\\slash") == "back\\\\slash"
    assert _escape_markdown("new\nline") == "new<br>line"
    assert _escape_markdown("crlf\r\nline") == "crlf<br>line"
    assert _escape_markdown("combo\\path|part\nend") == "combo\\\\path\\|part<br>end"


def test_to_markdown_escapes_special_characters():
    """Verify pipes, backslashes, and newlines in feature names, feature values, and headers are escaped."""
    exp = Explanation(
        feature_names=["income|monthly", "path\\name", "normal"],
        values=np.array([1.5, -0.8, 0.2]),
        method="tree_shap",
        scope="local",
        feature_values=np.array(["line1\nline2", "val|pipe", "val\\slash"]),
    )
    md = exp.to_markdown()
    lines = md.splitlines()

    # Headers line
    assert lines[2] == "| feature | value | attribution |"
    assert lines[3] == "| --- | --- | --- |"

    # Row 1: feature 'income|monthly' -> 'income\|monthly', value 'line1\nline2' -> 'line1<br>line2'
    assert lines[4] == "| income\\|monthly | line1<br>line2 | +1.5 |"
    # Row 2: feature 'path\name' -> 'path\\name', value 'val|pipe' -> 'val\|pipe'
    assert lines[5] == "| path\\\\name | val\\|pipe | -0.8 |"
    # Row 3: feature 'normal', value 'val\slash' -> 'val\\slash'
    assert lines[6] == "| normal | val\\\\slash | +0.2 |"

    # Ensure table rows do not break (exactly 3 data rows)
    assert len(lines) == 7


def test_to_markdown_consensus_escapes_method_headers_and_values():
    """Verify custom consensus method headers with pipes, backslashes, and newlines are properly escaped."""
    feat_names = ["feat|a", "feat\\b"]
    comp1 = Explanation(feature_names=feat_names, values=np.array([1.0, 0.5]), method="t")
    comp2 = Explanation(feature_names=feat_names, values=np.array([0.8, 0.4]), method="t")
    comps = {
        "custom|method": comp1,
        "algo\\v2": comp2,
        "method\nwith\nnewline": comp1,
    }
    exp = ConsensusExplanation(
        feature_names=feat_names,
        values=np.array([1.0, 0.5]),
        method="consensus",
        scope="local",
        feature_values=np.array(["multi\r\nline", "regular"]),
        components=comps,
        agreement=0.85,
    )
    md = exp.to_markdown()
    lines = md.splitlines()

    # Headers line should escape pipes, backslashes, and newlines in component method names
    assert (
        lines[2]
        == "| feature | value | attribution | custom\\|method | algo\\\\v2 | method<br>with<br>newline |"
    )
    assert lines[3] == "| --- | --- | --- | --- | --- | --- |"

    # Data rows should also escape features and values
    assert lines[4] == "| feat\\|a | multi<br>line | +1 | +0.6667 | +0.6667 | +0.6667 |"
    assert lines[5] == "| feat\\\\b | regular | +0.5 | +0.3333 | +0.3333 | +0.3333 |"
