"""Streamlit front-end. It only talks to the FastAPI backend over HTTP.

Run:  streamlit run ui/app.py
"""
from __future__ import annotations

import datetime as dt
import os
import time

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Free AutoML", page_icon="🤖", layout="wide")


# ====================================================================== API client
class ApiError(Exception):
    pass


def call(method: str, path: str, **kwargs) -> requests.Response:
    url = st.session_state.api_url.rstrip("/") + path
    try:
        resp = requests.request(method, url, timeout=kwargs.pop("timeout", 120), **kwargs)
    except requests.RequestException as exc:
        raise ApiError(f"Cannot reach the API at {st.session_state.api_url} ({exc.__class__.__name__})") from exc
    if not resp.ok:
        try:
            detail = resp.json().get("detail", resp.text)
        except ValueError:
            detail = resp.text
        raise ApiError(f"{detail}")
    return resp


@st.cache_data(ttl=300, show_spinner=False)
def fetch_profile(api_url: str, ds_id: str, target: str) -> dict:
    resp = requests.get(f"{api_url.rstrip('/')}/datasets/{ds_id}/profile", params={"target": target}, timeout=60)
    if not resp.ok:
        raise ApiError(resp.json().get("detail", resp.text))
    return resp.json()


def fmt_time(ts: float | None) -> str:
    return dt.datetime.fromtimestamp(ts).strftime("%d %b %H:%M") if ts else "-"


def goto(page: str) -> None:
    st.session_state["_goto"] = page
    st.rerun()


# ====================================================================== page: train
def page_train() -> None:
    st.header("Train a model")

    # ---- 1. dataset --------------------------------------------------
    st.subheader("1. Dataset")
    source = st.radio("Source", ["Upload new CSV", "Use existing dataset"], horizontal=True, label_visibility="collapsed")

    if source == "Upload new CSV":
        upload = st.file_uploader("Choose a CSV file", type=["csv"])
        if upload is not None and st.button("Upload dataset"):
            with st.spinner("Uploading..."):
                ds = call("POST", "/datasets", files={"file": (upload.name, upload.getvalue(), "text/csv")}).json()
            st.session_state.ds_id = ds["id"]
            st.success(f"Uploaded {ds['name']} ({ds['n_rows']:,} rows, {ds['n_cols']} columns)")
    else:
        datasets = call("GET", "/datasets").json()
        if not datasets:
            st.info("No datasets yet. Upload one first.")
        else:
            ids = [d["id"] for d in datasets]
            labels = {d["id"]: f"{d['name']}  ({d['n_rows']:,} rows, uploaded {fmt_time(d['created_at'])})" for d in datasets}
            current = st.session_state.get("ds_id")
            st.session_state.ds_id = st.selectbox(
                "Dataset", ids, index=ids.index(current) if current in ids else 0, format_func=labels.get
            )

    ds_id = st.session_state.get("ds_id")
    if not ds_id:
        st.stop()

    ds = call("GET", f"/datasets/{ds_id}").json()
    st.caption(f"Selected: **{ds['name']}** - {ds['n_rows']:,} rows, {ds['n_cols']} columns")
    with st.expander("Preview data and column summary"):
        prev = call("GET", f"/datasets/{ds_id}/preview", params={"n": 10}).json()
        st.dataframe(pd.DataFrame(prev["rows"]), width="stretch")
        st.dataframe(pd.DataFrame(ds["columns"]), width="stretch", hide_index=True)

    # ---- 2. target ---------------------------------------------------
    st.subheader("2. What do you want to predict?")
    col_names = [c["name"] for c in ds["columns"]]
    target = st.selectbox("Target column", col_names, index=len(col_names) - 1)

    try:
        prof = fetch_profile(st.session_state.api_url, ds_id, target)
    except ApiError as exc:
        st.error(str(exc))
        st.stop()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rows used", f"{prof['n_rows']:,}")
    m2.metric("Features used", len(prof["numeric_columns"]) + len(prof["categorical_columns"]))
    m3.metric("Detected task", prof["task"])
    m4.metric("Distinct target values", prof["n_unique_target"])

    for w in prof["warnings"]:
        st.warning(w)
    with st.expander("Feature details"):
        st.write("**Numeric:**", ", ".join(prof["numeric_columns"]) or "none")
        st.write("**Categorical:**", ", ".join(prof["categorical_columns"]) or "none")
        if prof["dropped_columns"]:
            st.write("**Ignored columns:**")
            st.dataframe(
                pd.DataFrame({"column": list(prof["dropped_columns"]), "reason": list(prof["dropped_columns"].values())}),
                hide_index=True,
            )
    if "class_counts" in prof:
        st.caption("Class distribution")
        st.bar_chart(pd.Series(prof["class_counts"]))

    # ---- 3. settings -------------------------------------------------
    st.subheader("3. Settings")
    opts = call("GET", "/options").json()

    c1, c2 = st.columns(2)
    choice = c1.selectbox("Task type", [f"Auto-detect ({prof['task']})", "classification", "regression"])
    task = prof["task"] if choice.startswith("Auto") else choice

    metrics = list(opts["metrics"][task])
    if task == "classification":
        # roc_auc only works for 2 classes, roc_auc_ovr only for more than 2
        metrics.remove("roc_auc_ovr" if prof["n_unique_target"] == 2 else "roc_auc")
    metric = c2.selectbox("Optimise for", metrics, help="First item is the recommended default.")

    estimators = st.multiselect("Models to try", opts["estimators"][task], default=opts["estimators"][task])
    c3, c4 = st.columns(2)
    time_budget = c3.slider("Search time (seconds)", 10, 900, 60, step=10,
                            help="More time lets AutoML try more models and settings.")
    test_size = c4.slider("Hold-out test share", 0.1, 0.4, 0.2, step=0.05,
                          help="Data kept aside to measure the final model honestly.")
    name = st.text_input("Experiment name (optional)")

    if st.button("Start training", type="primary", disabled=not estimators or prof["n_rows"] < 30):
        payload = {
            "dataset_id": ds_id, "target": target, "task": task, "metric": metric,
            "time_budget": time_budget, "test_size": test_size, "estimators": estimators,
            "name": name or None,
        }
        exp = call("POST", "/experiments", json=payload).json()
        st.session_state.sel_exp = exp["id"]
        goto("Experiments")


# ====================================================================== page: experiments
def page_experiments() -> None:
    st.header("Experiments")
    exps = call("GET", "/experiments").json()
    if not exps:
        st.info("No experiments yet. Train a model first.")
        return

    ids = [e["id"] for e in exps]
    labels = {e["id"]: f"{e['name']}  [{e['status']}]  {fmt_time(e['created_at'])}" for e in exps}
    current = st.session_state.get("sel_exp")
    exp_id = st.selectbox("Experiment", ids, index=ids.index(current) if current in ids else 0, format_func=labels.get)
    st.session_state.sel_exp = exp_id

    with st.expander("All experiments"):
        st.dataframe(
            pd.DataFrame([{
                "id": e["id"], "name": e["name"], "status": e["status"],
                "task": e["config"]["task"], "metric": e["config"]["metric"],
                "best model": (e["summary"] or {}).get("best_model"),
                "cv score": (e["summary"] or {}).get("cv_score"),
                "created": fmt_time(e["created_at"]),
            } for e in exps]),
            width="stretch", hide_index=True,
        )

    exp = call("GET", f"/experiments/{exp_id}").json()
    cfg, status = exp["config"], exp["status"]

    if status in ("queued", "running"):
        if status == "queued":
            st.info("Waiting in the queue; another experiment is training.")
            st.progress(0.0)
        else:
            budget, elapsed = cfg["time_budget"], exp["elapsed_s"]
            st.progress(min(elapsed / budget, 0.95), text=f"Searching for the best model... {elapsed:.0f}s of {budget}s")
        time.sleep(2)
        st.rerun()

    if status == "failed":
        st.error(f"Training failed: {exp['error']}")
        return

    # ---- completed ---------------------------------------------------
    meta, summary = exp["metadata"], exp["summary"]
    st.success(f"Finished in {exp['elapsed_s']:.0f}s - best model: **{summary['best_model']}**")

    top = st.columns(2 + len(summary["holdout_metrics"]))
    top[0].metric("Best model", summary["best_model"])
    top[1].metric(f"CV {summary['metric']}", f"{summary['cv_score']:.4f}")
    for slot, (k, v) in zip(top[2:], summary["holdout_metrics"].items()):
        slot.metric(f"Test {k}", f"{v:.4f}")

    st.subheader("Leaderboard")
    board = pd.DataFrame(call("GET", f"/experiments/{exp_id}/leaderboard").json())
    score_col = f"cv_{summary['metric']}"
    st.dataframe(board.drop(columns="best_config"), width="stretch", hide_index=True)
    st.bar_chart(board.set_index("model")[score_col])
    with st.expander("Best hyper-parameters per model"):
        for _, row in board.iterrows():
            st.write(f"**{row['model']}**")
            st.json(row["best_config"], expanded=False)

    with st.expander("Data and preprocessing used"):
        st.write("**Target:**", meta["target"], "| **Task:**", meta["task"], "| **Rows:**", meta["n_rows"])
        st.write("**Numeric features:**", ", ".join(meta["numeric_columns"]) or "none")
        st.write("**Categorical features:**", ", ".join(meta["categorical_columns"]) or "none")
        if meta["dropped_columns"]:
            st.write("**Ignored columns:**", meta["dropped_columns"])

    st.link_button("Download model (.joblib)", f"{st.session_state.api_url.rstrip('/')}/experiments/{exp_id}/model")
    if st.button("Use this model to predict", type="primary"):
        st.session_state.pred_exp = exp_id
        goto("Predict")


# ====================================================================== page: predict
def page_predict() -> None:
    st.header("Predict")
    done = [e for e in call("GET", "/experiments").json() if e["status"] == "completed"]
    if not done:
        st.info("No finished experiments yet.")
        return

    ids = [e["id"] for e in done]
    labels = {e["id"]: f"{e['name']}  ({e['summary']['best_model']}, {e['summary']['metric']} {e['summary']['cv_score']:.3f})"
              for e in done}
    current = st.session_state.get("pred_exp") or st.session_state.get("sel_exp")
    exp_id = st.selectbox("Model", ids, index=ids.index(current) if current in ids else 0, format_func=labels.get)
    st.session_state.pred_exp = exp_id
    meta = call("GET", f"/experiments/{exp_id}").json()["metadata"]
    is_clf = meta["task"] == "classification"

    tab_single, tab_batch = st.tabs(["Single prediction", "Batch (CSV)"])

    # ---- single ------------------------------------------------------
    with tab_single:
        st.caption("Fill in the feature values. Defaults are the training medians / most common categories.")
        record: dict = {}
        with st.form(f"single_{exp_id}"):
            cols = st.columns(3)
            for i, (feature, stats) in enumerate(meta["feature_stats"].items()):
                with cols[i % 3]:
                    if stats["type"] == "numeric":
                        record[feature] = st.number_input(feature, value=float(stats["median"]), format="%.4f",
                                                          help=f"training range: {stats['min']:.4g} to {stats['max']:.4g}")
                    else:
                        choice = st.selectbox(feature, ["(missing)"] + [str(v) for v in stats["values"]], index=1)
                        # send back the original value (it may be a number or bool, not a string)
                        lookup = {str(v): v for v in stats["values"]}
                        record[feature] = lookup.get(choice)
            submitted = st.form_submit_button("Predict", type="primary")

        if submitted:
            out = call("POST", f"/experiments/{exp_id}/predict",
                       json={"records": [record], "with_proba": is_clf}).json()["predictions"][0]
            st.metric("Prediction", str(out["prediction"]))
            proba = {k.removeprefix("proba_"): v for k, v in out.items() if k.startswith("proba_")}
            if proba:
                st.caption("Class probabilities")
                st.bar_chart(pd.Series(proba))

    # ---- batch -------------------------------------------------------
    with tab_batch:
        needed = list(meta["feature_stats"])
        st.caption(f"The CSV must contain these columns (extra columns are kept but ignored): {', '.join(needed)}")
        upload = st.file_uploader("CSV with new rows", type=["csv"], key=f"batch_{exp_id}")
        proba = st.checkbox("Include class probabilities", value=False, disabled=not is_clf)
        if upload is not None and st.button("Run batch prediction"):
            with st.spinner("Predicting..."):
                res = call("POST", f"/experiments/{exp_id}/predict/file", params={"with_proba": proba},
                           files={"file": (upload.name, upload.getvalue(), "text/csv")}).json()
            original = pd.read_csv(upload)
            original.columns = [str(c).strip() for c in original.columns]
            result = pd.concat([original, pd.DataFrame(res["predictions"])], axis=1)
            st.success(f"Predicted {res['n_rows']:,} rows")
            st.dataframe(result.head(200), width="stretch")
            st.download_button("Download predictions (CSV)", result.to_csv(index=False).encode(),
                               file_name="predictions.csv", mime="text/csv")


# ====================================================================== main
PAGES = {"Train": page_train, "Experiments": page_experiments, "Predict": page_predict}

st.session_state.setdefault("api_url", os.environ.get("AUTOML_API_URL", "http://localhost:8000"))
if "_goto" in st.session_state:                      # programmatic page switch (set before the radio is drawn)
    st.session_state["page"] = st.session_state.pop("_goto")

st.sidebar.title("Free AutoML")
page = st.sidebar.radio("Navigate", list(PAGES), key="page")
st.sidebar.divider()
st.sidebar.text_input("API URL", key="api_url")

try:
    requests.get(st.session_state.api_url.rstrip("/") + "/health", timeout=3).raise_for_status()
    st.sidebar.success("API connected")
except requests.RequestException:
    st.sidebar.error("API not reachable")
    st.error("The backend is not running. Start it from the project folder with:")
    st.code("uvicorn api.main:app --port 8000")
    st.stop()

try:
    PAGES[page]()
except ApiError as exc:
    st.error(str(exc))
