import pandas as pd
import pytest

from fraudguard.ui_viz import ARCHITECTURE_STEPS, architecture_html, gauge_html, hero_html


def test_gauge_flags_at_or_above_threshold():
    flagged = gauge_html(0.62, 0.30)
    allowed = gauge_html(0.05, 0.30)
    assert "FLAGGED FOR REVIEW" in flagged and "62.0%" in flagged
    assert "ALLOWED" in allowed and "5.0%" in allowed
    assert "FLAGGED" not in allowed


def test_gauge_clamps_out_of_range_values():
    html = gauge_html(1.7, -0.2)
    assert "100.0%" in html and "threshold 0.00" in html
    assert "nan" not in gauge_html(0.0, 1.0).lower()


def test_gauge_arc_length_follows_probability():
    half = gauge_html(0.5, 0.9)
    assert 'stroke-dasharray="141.37 282.74"' in half  # half of pi * 90


def test_hero_escapes_untrusted_text():
    html = hero_html('<script>alert(1)</script> rf v2', repo_url='https://x.test/?a="b"')
    assert "<script>" not in html and "&lt;script&gt;" in html
    assert "href='https://x.test/?a=&quot;b&quot;'" in html
    assert "View on GitHub" not in hero_html("rf v2")


def test_architecture_lists_every_step_in_order():
    html = architecture_html()
    positions = [html.index(title) for title, _ in ARCHITECTURE_STEPS]
    assert positions == sorted(positions)
    assert html.count("&rarr;") >= len(ARCHITECTURE_STEPS) - 1


def test_altair_charts_build():
    pytest.importorskip("altair")
    from fraudguard.ui_viz import cost_chart, effects_chart, precision_recall_chart

    effects = [{"feature": f"V{i}", "value": 0.1 * i, "reference": 0.0, "effect": (-1) ** i * 0.05 * i} for i in range(1, 12)]
    assert effects_chart(effects).to_dict()["mark"]
    sweep = pd.DataFrame({"threshold": [0.1, 0.5, 0.9], "total_cost": [3.0, 2.0, 4.0], "precision": [0.5, 0.8, 1.0], "recall": [1.0, 0.7, 0.2]})
    assert cost_chart(sweep, 0.3, 0.5).to_dict()
    assert precision_recall_chart(sweep, 0.3, 0.5).to_dict()
