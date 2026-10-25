"""fraudguard UI.   streamlit run app/streamlit_app.py

Talks to the FastAPI service if FRAUD_API_URL is set, otherwise loads artifacts/champion in-process.
Optional environment: FRAUD_REPO_URL, FRAUD_PUBLIC_API_URL, FRAUD_PUBLIC_MLFLOW_URL (links shown in the sidebar).
"""

import json
import os
import time
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from fraudguard.data import FEATURE_COLS
from fraudguard.ui_core import (
    apply_threshold,
    best_cost_threshold,
    evaluate_labelled,
    explain_by_occlusion,
    make_backend,
    operating_point,
    score_dataframe,
    threshold_sweep,
)
from fraudguard.ui_viz import (
    architecture_html,
    cost_chart,
    effects_chart,
    gauge_html,
    hero_html,
    precision_recall_chart,
)

ROOT = Path(__file__).resolve().parent
REPO_URL = os.environ.get("FRAUD_REPO_URL", "https://github.com/Rajdeep-Nayak/CreditCard-Fraud-detection")
PUBLIC_API_URL = os.environ.get("FRAUD_PUBLIC_API_URL", "").rstrip("/")
PUBLIC_MLFLOW_URL = os.environ.get("FRAUD_PUBLIC_MLFLOW_URL", "").rstrip("/")
MAX_UPLOAD_ROWS = 20_000

st.set_page_config(page_title="fraudguard | fraud detection", page_icon="🛡️", layout="wide")

st.markdown(
    """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.block-container {padding-top: 1.2rem; max-width: 1200px;}
[data-testid="stMetric"] {background: rgba(128,128,128,0.10); border: 1px solid rgba(128,128,128,0.25);
                          border-radius: 14px; padding: 10px 16px;}
.stTabs [data-baseweb="tab-list"] {gap: 6px;}
.stTabs [data-baseweb="tab"] {padding: 8px 16px; border-radius: 10px 10px 0 0;}
.insight {background: rgba(79,139,249,0.10); border-left: 4px solid #4F8BF9; border-radius: 10px;
          padding: 12px 16px; margin: 6px 0 16px 0;}
</style>
""",
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------ helpers
def show_chart(chart):
    """Altair chart that fills the width on both old and new Streamlit versions."""
    try:
        st.altair_chart(chart, width="stretch")
    except TypeError:
        st.altair_chart(chart, use_container_width=True)


@st.cache_resource(show_spinner="Connecting to the model...")
def get_backend():
    return make_backend()


@st.cache_data(show_spinner="Scoring the demo batch...")
def scored_demo_batch(_backend, model_version: str):
    path = ROOT / "demo_batch.csv"
    if not path.exists():
        return None
    return score_dataframe(_backend, pd.read_csv(path))


def load_samples():
    path = ROOT / "demo_samples.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    record = json.loads((ROOT.parent / "docs" / "sample_transaction.json").read_text(encoding="utf-8"))
    return [{"name": "Example transaction (first row of the dataset)", "actual_class": 0, "record": record}]


# ------------------------------------------------------------------ backend, header
try:
    backend = get_backend()
    info = backend.info()
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not reach the model: {exc}")
    st.info("Start the API (`uvicorn fraudguard.api.main:app`) and set FRAUD_API_URL, or export the champion to artifacts/champion.")
    st.stop()

model_threshold = float(info["threshold"])
validation = info.get("validation_metrics", {})
model_label = f"{info.get('model_type')} v{info.get('version')}"

components.html(hero_html(model_label, REPO_URL), height=215)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Champion model", model_label)
k2.metric("Validation PR-AUC", f"{validation.get('pr_auc', float('nan')):.3f}")
k3.metric("Fraud recall / precision", f"{validation.get('recall', float('nan')):.0%} / {validation.get('precision', float('nan')):.0%}")
k4.metric("Cost-tuned threshold", f"{model_threshold:.3f}")

with st.expander("New here? A 60-second guided tour"):
    st.markdown(
        """
1. **Score a transaction**: pick *Held-out fraud #1*, press **Score**, and read the gauge and the explanation.
2. **Batch analysis**: load the demo batch and see precision and recall on real held-out data.
3. **Cost simulator**: drag the threshold in the sidebar and watch cost, recall and false alarms trade off.
   Then raise *cost of a false alarm* to 50 and see the best threshold move.
4. **How it works** shows the pipeline behind this page, and the **Model card** lists what it cannot do.
"""
    )

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("### Decision settings")
    threshold = st.slider(
        "Flag as fraud when probability is at least",
        min_value=0.01,
        max_value=0.99,
        value=float(min(max(round(model_threshold, 2), 0.01), 0.99)),
        step=0.01,
        help="Lower catches more fraud but raises more false alarms.",
    )
    false_alarm_cost = st.number_input(
        "Cost of one false alarm",
        min_value=0.0,
        value=5.0,
        step=1.0,
        help="A missed fraud costs its amount. A false alarm costs this much (review time, customer friction).",
    )
    st.caption(f"The model's own cost-tuned threshold is {model_threshold:.4f}.")

    st.markdown("### Model")
    st.caption(f"Version {info.get('version')} of the registry champion, source `{info.get('source')}`.")

    st.markdown("### Links")
    st.markdown(f"[Source code on GitHub]({REPO_URL})")
    if PUBLIC_API_URL:
        st.markdown(f"[API docs (OpenAPI)]({PUBLIC_API_URL}/docs)")
    if PUBLIC_MLFLOW_URL:
        st.markdown(f"[MLflow experiments and registry]({PUBLIC_MLFLOW_URL})")

tab_score, tab_batch, tab_sim, tab_how, tab_card = st.tabs(
    ["🔍 Score a transaction", "📦 Batch analysis", "💰 Cost simulator", "⚙️ How it works", "📄 Model card"]
)

# ------------------------------------------------------------------ 1. single transaction
with tab_score:
    left, right = st.columns(2, gap="large")
    samples = load_samples()

    with left:
        st.subheader("Pick a transaction")
        choice = st.selectbox("Start from a sample", [s["name"] for s in samples])
        sample = next(s for s in samples if s["name"] == choice)
        st.caption("Held-out samples are real transactions from the test window. The model never saw them in training.")
        original = {c: float(sample["record"][c]) for c in FEATURE_COLS}
        with st.expander("Edit the 30 feature values"):
            st.caption("V1-V28 are anonymised PCA components. Try changing Amount.")
            table = pd.DataFrame({"feature": FEATURE_COLS, "value": [original[c] for c in FEATURE_COLS]})
            edited = st.data_editor(table, disabled=["feature"], hide_index=True, key=f"editor-{choice}", height=320)
        run = st.button("Score transaction", type="primary")

        if run:
            try:
                record = {row.feature: float(row.value) for row in edited.itertuples()}
                started = time.perf_counter()
                prediction = backend.predict([record])[0]
                latency_ms = (time.perf_counter() - started) * 1000
                with st.spinner("Explaining the score..."):
                    contributions = explain_by_occlusion(backend, record)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Scoring failed: {exc}")
            else:
                st.session_state["single"] = {
                    "probability": prediction["fraud_probability"],
                    "contributions": contributions,
                    "latency_ms": latency_ms,
                    "sample": choice,
                    "actual_class": sample.get("actual_class"),
                    "edited": record != original,
                }

    with right:
        st.subheader("Result")
        result = st.session_state.get("single")
        if result is None:
            st.info("Press **Score transaction** to see the risk gauge, the decision and the explanation.")
        else:
            probability = result["probability"]
            flagged = probability >= threshold
            components.html(gauge_html(probability, threshold), height=250)

            m1, m2 = st.columns(2)
            m1.metric("Scoring latency", f"{result['latency_ms']:.0f} ms")
            actual = result["actual_class"]
            if actual is None or result["edited"]:
                m2.metric("Ground truth", "n/a" if actual is None else "values edited")
            else:
                label = "Fraud" if actual == 1 else "Legitimate"
                outcome = (
                    "caught" if (actual == 1 and flagged)
                    else "MISSED" if actual == 1
                    else "false alarm" if flagged
                    else "correctly allowed"
                )
                m2.metric("Ground truth", label, outcome, delta_color="off")

    if st.session_state.get("single"):
        st.markdown("#### Why this score?")
        st.caption(
            "Each bar is how much the probability drops when that feature is replaced by a neutral value "
            "(0 for V1-V28, a typical amount for Amount). Red pushed the score up, green pulled it down. "
            "This one-feature-at-a-time estimate ignores interactions, so it is an approximation, not SHAP."
        )
        show_chart(effects_chart(st.session_state["single"]["contributions"]))

# ------------------------------------------------------------------ 2. batch
with tab_batch:
    st.subheader("Score a whole file")
    st.caption(
        f"Required columns: Time, V1-V28, Amount. An optional Class column (0/1) adds precision and recall. "
        f"Up to {MAX_UPLOAD_ROWS:,} rows."
    )
    uploaded = st.file_uploader("CSV file", type=["csv"])
    demo_path = ROOT / "demo_batch.csv"
    demo_clicked = demo_path.exists() and st.button("Use the built-in demo batch (real held-out transactions)")

    if uploaded is not None:
        st.session_state["batch_input"] = pd.read_csv(uploaded)
        st.session_state["batch_scored"] = None
    elif demo_clicked:
        st.session_state["batch_input"] = pd.read_csv(demo_path)
        st.session_state["batch_scored"] = None

    batch_input = st.session_state.get("batch_input")
    if batch_input is not None:
        if len(batch_input) > MAX_UPLOAD_ROWS:
            st.warning(f"Only the first {MAX_UPLOAD_ROWS:,} rows are scored.")
            batch_input = batch_input.head(MAX_UPLOAD_ROWS)
        st.write(f"{len(batch_input):,} rows loaded.")
        if st.button("Score batch", type="primary"):
            try:
                with st.spinner("Scoring..."):
                    started = time.perf_counter()
                    st.session_state["batch_scored"] = score_dataframe(backend, batch_input)
                    st.session_state["batch_seconds"] = time.perf_counter() - started
            except Exception as exc:  # noqa: BLE001
                st.error(f"Scoring failed: {exc}")

    scored = st.session_state.get("batch_scored")
    if scored is not None:
        view = apply_threshold(scored, threshold)
        flagged_count = int(view["is_fraud"].sum())
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Transactions", f"{len(view):,}")
        c2.metric("Flagged", f"{flagged_count:,}")
        c3.metric("Flag rate", f"{flagged_count / len(view):.2%}")
        c4.metric("Scored in", f"{st.session_state.get('batch_seconds', 0.0):.2f} s")

        if "Class" in view.columns and view["Class"].nunique() > 1:
            st.markdown("**Against the Class column**")
            if "weight" in view.columns:
                point = operating_point(view, false_alarm_cost, threshold)
                frauds = int(point["caught"] + point["missed"])
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Frauds caught", f"{int(point['caught'])} of {frauds}")
                m2.metric("Recall", f"{point['recall']:.0%}")
                m3.metric("Precision (full window)", f"{point['precision']:.0%}")
                m4.metric("False alarms per 10,000 legit", f"{point['fp_per_10k_legit']:.1f}")
                st.caption(
                    f"Legitimate rows in this batch are re-weighted so the figures describe the whole "
                    f"{view['weight'].sum():,.0f}-transaction test window, where fraud is about "
                    f"{frauds / view['weight'].sum():.2%} of transactions."
                )
            else:
                metrics = evaluate_labelled(view, threshold)
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Precision", f"{metrics['precision']:.2f}")
                m2.metric("Recall", f"{metrics['recall']:.2f}")
                m3.metric("F1", f"{metrics['f1']:.2f}")
                m4.metric("PR-AUC", f"{metrics['pr_auc']:.2f}")
                st.caption(
                    f"Caught {int(metrics['tp'])} frauds, missed {int(metrics['fn'])}, "
                    f"{int(metrics['fp'])} false alarms, {int(metrics['tn'])} correctly allowed."
                )

        st.markdown("**Highest-risk transactions**")
        shown = ["fraud_probability", "is_fraud", "Amount", "Time"] + (["Class"] if "Class" in view.columns else [])
        st.dataframe(
            view.sort_values("fraud_probability", ascending=False).head(50)[shown],
            hide_index=True,
            column_config={
                "fraud_probability": st.column_config.ProgressColumn(
                    "Fraud probability", min_value=0.0, max_value=1.0, format="%.3f"
                )
            },
        )
        st.download_button(
            "Download scored CSV",
            view.to_csv(index=False).encode("utf-8"),
            file_name="scored_transactions.csv",
            mime="text/csv",
        )

# ------------------------------------------------------------------ 3. cost simulator
with tab_sim:
    st.subheader("Why 0.5 is the wrong threshold")
    st.caption(
        "A fraud model outputs a probability. The decision threshold turns it into an action, and the right value "
        "depends on money: a missed fraud costs its amount, a false alarm costs a fixed review cost."
    )
    demo = scored_demo_batch(backend, str(info.get("version")))
    if demo is None:
        st.info("Run `python scripts/make_demo_samples.py` to create the demo batch this simulator uses.")
    else:
        sweep = threshold_sweep(demo, false_alarm_cost)
        best_threshold = best_cost_threshold(sweep)
        point = operating_point(demo, false_alarm_cost, threshold)
        frauds = int(point["caught"] + point["missed"])

        st.markdown(
            f"<div class='insight'>At threshold <b>{threshold:.2f}</b> the model catches "
            f"<b>{point['recall']:.0%}</b> of frauds ({int(point['caught'])} of {frauds}) at the price of about "
            f"<b>{point['fp_per_10k_legit']:.1f} false alarms per 10,000 legitimate transactions</b>, "
            f"saving <b>{point['cost_saving_pct']:.0f}%</b> of the cost versus having no model. "
            f"The cheapest threshold on this data is <b>{best_threshold:.2f}</b>.</div>",
            unsafe_allow_html=True,
        )

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Recall", f"{point['recall']:.0%}")
        s2.metric("Precision", f"{point['precision']:.0%}")
        s3.metric("False alarms / 10k legit", f"{point['fp_per_10k_legit']:.1f}")
        s4.metric("Cost saved", f"{point['cost_saving_pct']:.0f}%")

        c_left, c_right = st.columns(2, gap="large")
        with c_left:
            st.markdown("**Total cost by threshold**")
            show_chart(cost_chart(sweep, model_threshold, threshold))
        with c_right:
            st.markdown("**Precision and recall by threshold**")
            show_chart(precision_recall_chart(sweep, model_threshold, threshold))
        st.caption(
            f"Built from the demo batch: all {frauds} test-window frauds plus a sample of legitimate transactions that "
            "is re-weighted to the real fraud rate. With so few frauds the curves are noisy, so read the shape, not the decimals."
        )

# ------------------------------------------------------------------ 4. how it works
with tab_how:
    st.subheader("From notebook to product")
    components.html(architecture_html(), height=270)
    with st.expander("Why a time-based split?"):
        st.write(
            "Fraud patterns drift. Training on the past and testing on the future is how a deployed model is actually "
            "used, and it avoids the optimistic numbers a random split gives."
        )
    with st.expander("Why PR-AUC instead of accuracy?"):
        st.write(
            "Only about 0.17 % of transactions are fraud, so a model that flags nothing is 99.8 % accurate and useless. "
            "PR-AUC and fraud-class precision and recall measure what matters."
        )
    with st.expander("Why a cost-based threshold?"):
        st.write(
            "The best cut-off depends on what a miss and a false alarm cost. The threshold is tuned on a validation "
            "window from those costs, never on the test set. The Cost simulator tab lets you see it."
        )
    with st.expander("Why a model registry?"):
        st.write(
            "Every trained model becomes a version with its metrics attached. A challenger only replaces the champion "
            "if it is better on validation data, so the serving model cannot silently get worse."
        )
    st.markdown("**Run the whole stack yourself**")
    st.code("git clone " + REPO_URL + "\ncd CreditCard-Fraud-detection\ndocker compose up --build", language="bash")

# ------------------------------------------------------------------ 5. model card
with tab_card:
    st.subheader("Model card")
    st.markdown(
        f"""
**Model.** {info.get('model_type')} (registry version {info.get('version')}), a scikit-learn pipeline: log-scaled amount,
hour-of-day features and the 28 anonymised components.

**Intended use.** Ranking card transactions by fraud risk for review. This is a portfolio demonstration of an
end-to-end ML workflow, **not** a production fraud system.

**Data.** Two days of European card transactions from 2013 (ULB / Kaggle), 283,726 rows after removing exact duplicates,
0.17 % fraud. Features V1-V28 are PCA components, so they cannot be interpreted directly.

**Evaluation.** Train on the earliest transactions, tune the threshold on the next window, report on the most recent
20 %. Models are compared on validation PR-AUC only.

**Validation performance of this champion.** PR-AUC {validation.get('pr_auc', float('nan')):.3f}, fraud precision
{validation.get('precision', float('nan')):.0%}, recall {validation.get('recall', float('nan')):.0%}.
Test-window results for all three model families are in the [README]({REPO_URL}#results).

**Limitations.**
- The test window holds only 74 frauds, so differences of a few transactions between models are within noise.
- The data is from 2013 and from one region; fraud behaviour has changed since, so the model would need retraining.
- Costs are illustrative assumptions (a missed fraud costs its amount, a false alarm costs a fixed 5).
- The per-feature explanation is a quick estimate, not SHAP.
- No fairness analysis is possible: the features are anonymised and carry no demographic information.
"""
    )

st.markdown("---")
st.caption(f"fraudguard · built with scikit-learn, MLflow, FastAPI, Streamlit and Docker · [source]({REPO_URL})")
