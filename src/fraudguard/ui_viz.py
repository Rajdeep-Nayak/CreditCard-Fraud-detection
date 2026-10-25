"""Visual building blocks for the Streamlit app.

The HTML cards are self-contained (own dark background, inline CSS and SVG) so they look right in any Streamlit
theme and need no extra dependencies. Chart builders import Altair lazily (it ships with Streamlit).
"""

from __future__ import annotations

import html
import math
from collections.abc import Sequence
from typing import Any

import pandas as pd

_BASE_CSS = """
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;color:#E6EAF2;background:transparent}
a{color:inherit}
"""

GREEN, AMBER, RED = "#22c55e", "#f59e0b", "#ef4444"


def _document(body: str, css: str) -> str:
    return f"<!doctype html><html><head><meta charset='utf-8'><style>{_BASE_CSS}{css}</style></head><body>{body}</body></html>"


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return min(max(float(value), low), high)


# --------------------------------------------------------------------------------------
# Risk gauge
# --------------------------------------------------------------------------------------
def gauge_html(probability: float, threshold: float) -> str:
    """Half-circle gauge: arc = fraud probability, tick = decision threshold."""
    p, t = _clamp(probability), _clamp(threshold)
    arc = math.pi * 90
    flagged = p >= t
    color = RED if flagged else (AMBER if p >= 0.5 * t else GREEN)
    cos_t, sin_t = math.cos(math.pi * t), math.sin(math.pi * t)
    x1, y1 = 100 - 78 * cos_t, 100 - 78 * sin_t
    x2, y2 = 100 - 104 * cos_t, 100 - 104 * sin_t
    verdict = "FLAGGED FOR REVIEW" if flagged else "ALLOWED"
    css = """
    .card{background:#161B26;border:1px solid #262E3F;border-radius:16px;padding:14px 18px;text-align:center}
    svg{width:100%;max-width:330px;height:auto}
    .verdict{display:inline-block;margin-top:2px;padding:5px 14px;border-radius:999px;font-weight:700;
             font-size:13px;letter-spacing:.06em}
    .meta{font-size:12px;color:#93A0B8;margin-top:8px}
    """
    body = f"""
    <div class="card">
      <svg viewBox="0 0 200 118" role="img" aria-label="fraud probability gauge">
        <path d="M 10 100 A 90 90 0 0 1 190 100" fill="none" stroke="#2A3142" stroke-width="14" stroke-linecap="round"/>
        <path d="M 10 100 A 90 90 0 0 1 190 100" fill="none" stroke="{color}" stroke-width="14" stroke-linecap="round"
              stroke-dasharray="{p * arc:.2f} {arc:.2f}"/>
        <line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="#E6EAF2" stroke-width="2.5"/>
        <text x="100" y="86" text-anchor="middle" font-size="30" font-weight="700" fill="#E6EAF2">{p:.1%}</text>
        <text x="100" y="104" text-anchor="middle" font-size="9" fill="#93A0B8">fraud probability</text>
      </svg>
      <div class="verdict" style="background:{color}22;color:{color};border:1px solid {color}66">{verdict}</div>
      <div class="meta">decision threshold {t:.2f} &middot; the white tick on the arc</div>
    </div>"""
    return _document(body, css)


# --------------------------------------------------------------------------------------
# Hero header
# --------------------------------------------------------------------------------------
def hero_html(model_label: str, repo_url: str | None = None) -> str:
    chips = ["FastAPI", "MLflow registry", "Docker Compose", "Time-based evaluation", "Cost-based threshold"]
    chip_html = "".join(f"<span class='chip'>{html.escape(c)}</span>" for c in chips)
    link = (
        f"<a class='btn' href='{html.escape(repo_url, quote=True)}' target='_blank' rel='noopener'>View on GitHub &rarr;</a>"
        if repo_url
        else ""
    )
    css = """
    .hero{background:linear-gradient(120deg,#1e3a8a 0%,#5b21b6 55%,#be185d 100%);border-radius:18px;padding:26px 30px;
          position:relative;overflow:hidden}
    .hero:after{content:'';position:absolute;right:-60px;top:-60px;width:240px;height:240px;border-radius:50%;
          background:rgba(255,255,255,.08)}
    h1{font-size:34px;letter-spacing:-.02em;font-weight:800}
    .tag{margin-top:6px;font-size:16px;color:#E0E7FF;max-width:720px;line-height:1.45}
    .row{margin-top:16px;display:flex;flex-wrap:wrap;gap:8px;align-items:center}
    .chip{background:rgba(255,255,255,.16);border:1px solid rgba(255,255,255,.25);padding:4px 11px;border-radius:999px;
          font-size:12px;font-weight:600}
    .model{background:rgba(0,0,0,.28)}
    .btn{margin-left:auto;background:#fff;color:#312e81;text-decoration:none;font-weight:700;font-size:13px;
         padding:7px 14px;border-radius:10px}
    """
    body = f"""
    <div class="hero">
      <h1>&#128737;&#65039; fraudguard</h1>
      <div class="tag">Credit card fraud detection, shipped like a product: honest evaluation on imbalanced data,
      a cost-aware decision threshold, a versioned model registry and a containerised API behind this app.</div>
      <div class="row">{chip_html}<span class="chip model">champion: {html.escape(model_label)}</span>{link}</div>
    </div>"""
    return _document(body, css)


# --------------------------------------------------------------------------------------
# Architecture diagram
# --------------------------------------------------------------------------------------
ARCHITECTURE_STEPS: list[tuple[str, str]] = [
    ("Data", "Kaggle / ULB card transactions, 0.17 % fraud"),
    ("Training", "time-based split, class weights, cost-based threshold"),
    ("MLflow", "every run tracked, every model versioned"),
    ("Champion", "best on validation PR-AUC, promoted by alias"),
    ("FastAPI", "validated /predict and /predict/batch"),
    ("This app", "scoring, explanations, cost simulator"),
]


def architecture_html(steps: Sequence[tuple[str, str]] = ARCHITECTURE_STEPS) -> str:
    nodes = []
    for i, (title, sub) in enumerate(steps):
        nodes.append(
            f"<div class='node'><div class='n'>{i + 1}</div><div class='t'>{html.escape(title)}</div>"
            f"<div class='s'>{html.escape(sub)}</div></div>"
        )
        if i < len(steps) - 1:
            nodes.append("<div class='arrow'>&rarr;</div>")
    css = """
    .wrap{background:#161B26;border:1px solid #262E3F;border-radius:16px;padding:18px}
    .flow{display:flex;flex-wrap:nowrap;align-items:stretch;gap:6px}
    .node{flex:1 1 0;min-width:0;background:#1D2433;border:1px solid #2C3650;border-radius:12px;padding:12px}
    .n{width:22px;height:22px;border-radius:50%;background:#4F8BF9;color:#fff;font-size:12px;font-weight:700;
       display:flex;align-items:center;justify-content:center;margin-bottom:6px}
    .t{font-weight:700;font-size:14px}
    .s{font-size:11.5px;color:#93A0B8;margin-top:3px;line-height:1.35}
    .arrow{flex:0 0 14px;align-self:center;color:#4F8BF9;font-size:18px;text-align:center}
    .band{margin-top:14px;text-align:center;font-size:12px;color:#93A0B8;border-top:1px dashed #2C3650;padding-top:10px}
    @media (max-width:760px){
      .flow{flex-wrap:wrap}
      .node{flex:1 1 44%}
      .arrow{display:none}
    }
    """
    body = (
        f"<div class='wrap'><div class='flow'>{''.join(nodes)}</div>"
        "<div class='band'>API, UI and MLflow run together with <b>docker compose up</b></div></div>"
    )
    return _document(body, css)


# --------------------------------------------------------------------------------------
# Altair charts
# --------------------------------------------------------------------------------------
def effects_chart(effects: Sequence[dict[str, Any]], top: int = 10):
    """Diverging bars: which features pushed this transaction's score up (red) or down (green)."""
    import altair as alt

    df = pd.DataFrame(list(effects)[:top])
    df["direction"] = ["raises risk" if v > 0 else "lowers risk" for v in df["effect"]]
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("effect:Q", title="Change in fraud probability when the feature is neutralised"),
            y=alt.Y("feature:N", sort=list(df["feature"]), title=None),
            color=alt.Color(
                "direction:N",
                scale=alt.Scale(domain=["raises risk", "lowers risk"], range=[RED, GREEN]),
                legend=alt.Legend(title=None, orient="bottom"),
            ),
            tooltip=["feature", alt.Tooltip("value:Q", format=".3f"), alt.Tooltip("effect:Q", format="+.3f")],
        )
        .properties(height=max(120, 26 * len(df) + 50))
    )


def _marker_rules(model_threshold: float, chosen: float):
    import altair as alt

    markers = pd.DataFrame(
        {"threshold": [model_threshold, chosen], "marker": ["model-tuned threshold", "your threshold"]}
    )
    return (
        alt.Chart(markers)
        .mark_rule(strokeDash=[5, 4], strokeWidth=2)
        .encode(
            x="threshold:Q",
            color=alt.Color(
                "marker:N",
                scale=alt.Scale(domain=["model-tuned threshold", "your threshold"], range=["#F59E0B", "#E6EAF2"]),
                legend=alt.Legend(title=None, orient="bottom"),
            ),
        )
    )


def cost_chart(sweep: pd.DataFrame, model_threshold: float, chosen: float):
    import altair as alt

    line = (
        alt.Chart(sweep)
        .mark_line(color="#4F8BF9", strokeWidth=3)
        .encode(
            x=alt.X("threshold:Q", title="Decision threshold", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("total_cost:Q", title="Estimated total cost (missed fraud + false alarms)"),
            tooltip=[
                alt.Tooltip("threshold:Q", format=".2f"),
                alt.Tooltip("total_cost:Q", format=",.0f"),
                alt.Tooltip("recall:Q", format=".2f"),
                alt.Tooltip("precision:Q", format=".2f"),
            ],
        )
    )
    return (line + _marker_rules(model_threshold, chosen)).properties(height=280)


def precision_recall_chart(sweep: pd.DataFrame, model_threshold: float, chosen: float):
    import altair as alt

    long = sweep.melt(id_vars="threshold", value_vars=["precision", "recall"], var_name="metric", value_name="value")
    lines = (
        alt.Chart(long)
        .mark_line(strokeWidth=3)
        .encode(
            x=alt.X("threshold:Q", title="Decision threshold", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("value:Q", title="Fraud-class precision / recall", scale=alt.Scale(domain=[0, 1.02])),
            color=alt.Color(
                "metric:N",
                scale=alt.Scale(domain=["precision", "recall"], range=["#A78BFA", "#34D399"]),
                legend=alt.Legend(title=None, orient="bottom"),
            ),
        )
    )
    return (lines + _marker_rules(model_threshold, chosen)).properties(height=280)
