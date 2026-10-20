"""fraudguard UI.   streamlit run app/streamlit_app.py

Talks to the FastAPI service if FRAUD_API_URL is set, otherwise loads artifacts/champion in-process.
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from fraudguard.data import FEATURE_COLS
from fraudguard.ui_core import (
    apply_threshold,
    evaluate_labelled,
    explain_by_occlusion,
    make_backend,
    score_dataframe,
)

ROOT = Path(__file__).resolve().parent
MAX_UPLOAD_ROWS = 20_000

st.set_page_config(page_title="fraudguard", page_icon="🛡️", layout="wide")


@st.cache_resource(show_spinner="Connecting to the model...")
def get_backend():
    return make_backend()


def load_samples():
    path = ROOT / "demo_samples.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    fallback = ROOT.parent / "docs" / "sample_transaction.json"
    record = json.loads(fallback.read_text(encoding="utf-8"))
    return [{"name": "Example transaction (first row of the dataset)", "actual_class": 0, "record": record}]


# ------------------------------------------------------------------ header & backend
st.title("🛡️ fraudguard: credit card fraud scoring")
st.caption(
    "Score a transaction or a CSV with the champion model from the registry. "
    "Built on the public ULB/Kaggle credit card dataset (features V1-V28 are anonymised PCA components)."
)

try:
    backend = get_backend()
    info = backend.info()
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not reach the model: {exc}")
    st.info("Start the API (`uvicorn fraudguard.api.main:app`) and set FRAUD_API_URL, or export the champion to artifacts/champion.")
    st.stop()

model_threshold = float(info["threshold"])
validation = info.get("validation_metrics", {})

with st.sidebar:
    st.header("Model")
    st.write(f"Version **{info.get('version')}** ({info.get('model_type')})")
    st.caption(f"Source: {info.get('source')}")
    if validation:
        st.caption(
            "Validation: "
            f"PR-AUC {validation.get('pr_auc', float('nan')):.3f}, "
            f"precision {validation.get('precision', float('nan')):.2f}, "
            f"recall {validation.get('recall', float('nan')):.2f}"
        )
    st.header("Decision threshold")
    threshold = st.slider(
        "Flag as fraud when probability is at least",
        min_value=0.01,
        max_value=0.99,
        value=float(min(max(round(model_threshold, 2), 0.01), 0.99)),
        step=0.01,
    )
    st.caption(
        f"The model's cost-tuned threshold is {model_threshold:.4f}. "
        "Lower it to catch more fraud at the price of more false alarms; raise it for the opposite."
    )

tab_single, tab_batch, tab_about = st.tabs(["Single transaction", "Batch (CSV)", "About"])

# ------------------------------------------------------------------ single transaction
with tab_single:
    st.subheader("Score one transaction")
    samples = load_samples()
    names = [s["name"] for s in samples]
    choice = st.selectbox("Start from a sample", names)
    sample = next(s for s in samples if s["name"] == choice)
    st.caption("Edit any value, then score it. Held-out samples are real transactions the model never saw in training.")

    table = pd.DataFrame({"feature": FEATURE_COLS, "value": [float(sample["record"][c]) for c in FEATURE_COLS]})
    edited = st.data_editor(table, disabled=["feature"], hide_index=True, key=f"editor-{choice}", height=320)

    if st.button("Score transaction", type="primary"):
        try:
            record = {row.feature: float(row.value) for row in edited.itertuples()}
            with st.spinner("Scoring..."):
                prediction = backend.predict([record])[0]
                contributions = explain_by_occlusion(backend, record)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Scoring failed: {exc}")
        else:
            st.session_state["single"] = {
                "probability": prediction["fraud_probability"],
                "contributions": contributions,
                "sample": choice,
                "actual_class": sample.get("actual_class"),
            }

    result = st.session_state.get("single")
    if result:
        probability = result["probability"]
        flagged = probability >= threshold
        col1, col2, col3 = st.columns(3)
        col1.metric("Fraud probability", f"{probability:.1%}")
        col2.metric("Decision", "FLAGGED" if flagged else "Not flagged")
        col3.metric("Threshold", f"{threshold:.2f}")
        st.progress(min(max(float(probability), 0.0), 1.0))
        if flagged:
            st.error("This transaction would be flagged for review.")
        else:
            st.success("This transaction would be allowed.")
        if result["actual_class"] is not None:
            label = "fraud" if result["actual_class"] == 1 else "legitimate"
            st.caption(f"Ground truth of the original, unedited sample ({result['sample']}): {label}.")

        st.markdown("**Which features moved the score?** (approximate)")
        st.caption(
            "Each bar is how much the probability drops when that feature is replaced by a neutral value "
            "(0 for V1-V28, a typical amount for Amount). Positive means it pushed the score up. "
            "This is a one-feature-at-a-time estimate, not SHAP."
        )
        top = pd.DataFrame(result["contributions"][:10])
        st.bar_chart(top.set_index("feature")["effect"])
        st.dataframe(top, hide_index=True)

# ------------------------------------------------------------------ batch
with tab_batch:
    st.subheader("Score a CSV")
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
                    st.session_state["batch_scored"] = score_dataframe(backend, batch_input)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Scoring failed: {exc}")

    scored = st.session_state.get("batch_scored")
    if scored is not None:
        view = apply_threshold(scored, threshold)
        flagged_count = int(view["is_fraud"].sum())
        col1, col2, col3 = st.columns(3)
        col1.metric("Transactions", f"{len(view):,}")
        col2.metric("Flagged", f"{flagged_count:,}")
        col3.metric("Flag rate", f"{flagged_count / len(view):.2%}")

        metrics = evaluate_labelled(view, threshold)
        if metrics:
            st.markdown("**Against the Class column**")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Precision", f"{metrics['precision']:.2f}")
            m2.metric("Recall", f"{metrics['recall']:.2f}")
            m3.metric("F1", f"{metrics['f1']:.2f}")
            m4.metric("PR-AUC", f"{metrics['pr_auc']:.2f}")
            st.caption(
                f"Caught {int(metrics['tp'])} frauds, missed {int(metrics['fn'])}, "
                f"{int(metrics['fp'])} false alarms, {int(metrics['tn'])} correctly allowed."
            )

        columns = ["fraud_probability", "is_fraud", "Amount", "Time"] + (["Class"] if "Class" in view.columns else [])
        st.markdown("**Highest-risk transactions**")
        st.dataframe(view.sort_values("fraud_probability", ascending=False).head(50)[columns], hide_index=True)
        st.download_button(
            "Download scored CSV",
            view.to_csv(index=False).encode("utf-8"),
            file_name="scored_transactions.csv",
            mime="text/csv",
        )

# ------------------------------------------------------------------ about
with tab_about:
    st.markdown(
        """
**What this is.** A portfolio project that turns a fraud-detection notebook into a reproducible pipeline:
time-based evaluation, PR-AUC and fraud-class metrics, a cost-based decision threshold, MLflow tracking and
a model registry, a FastAPI service, and this UI.

**How the model was chosen.** Models are compared on a validation window, never the test set. The decision
threshold is picked from costs (a missed fraud costs its amount, a false alarm costs a fixed review cost).

**Limits worth knowing.**
- The test window holds only 74 frauds, so differences of a few transactions between models are within noise.
- The dataset is from two days of European card transactions in 2013 and its features are anonymised, so this
  is a demonstration of the engineering, not a production fraud system.
- The feature explanation is a quick one-at-a-time estimate, not SHAP.
"""
    )
