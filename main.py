"""Minimal Streamlit app: tinygrad LSTM stock-price forecasting.

Run with:  streamlit run main.py
"""

from __future__ import annotations

import io
import os
from datetime import date

# Must be set BEFORE tinygrad is imported — tinygrad reads its env-var config at
# import time. Streamlit can run script execution and background reruns on different
# threads, and tinygrad's on-disk kernel-compilation cache opens a SQLite connection
# that is thread-affine (sqlite3 objects can't cross threads). If JIT-triggered
# compilation later happens on a different thread than the one that first opened that
# connection, it raises `sqlite3.ProgrammingError: SQLite objects created in a thread
# can only be used in that same thread`. CACHELEVEL=0 tells tinygrad to skip the disk
# cache entirely (compile in memory only), which avoids the cross-thread SQLite access
# altogether. The only cost is that compiled kernels aren't persisted across process
# restarts — each fresh `streamlit run` recompiles once, which is a one-time cost per
# process, not per session.
os.environ.setdefault("CACHELEVEL", "0")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from tinygrad import Tensor, TinyJit, nn
from tinygrad.nn.optim import Adam
from tinygrad.nn.state import get_parameters

try:  # pragma: no cover - tinygrad API differs across versions
    from tinygrad.helpers import Context, TRAINING
except Exception:  # newer tinygrad
    Context = None
    TRAINING = None

st.set_page_config(page_title="LSTM stock forecaster", page_icon=":material/show_chart:", layout="wide")


# --------------------------------------------------------------------------------------
# tinygrad training-mode shim (older tinygrad uses Context(TRAINING=1), newer Tensor.train)
# --------------------------------------------------------------------------------------
def train_ctx():
    if hasattr(Tensor, "train"):
        return Tensor.train()
    return Context(TRAINING=1)


# --------------------------------------------------------------------------------------
# tinygrad LSTM model
# --------------------------------------------------------------------------------------
class LSTMCell:
    def __init__(self, input_size: int, hidden_size: int):
        self.hidden_size = hidden_size
        k = (1.0 / hidden_size) ** 0.5
        self.weight_ih = Tensor.uniform(4 * hidden_size, input_size, low=-k, high=k)
        self.weight_hh = Tensor.uniform(4 * hidden_size, hidden_size, low=-k, high=k)
        self.bias_ih = Tensor.zeros(4 * hidden_size)
        self.bias_hh = Tensor.zeros(4 * hidden_size)

    def __call__(self, x: Tensor, h: Tensor, c: Tensor) -> tuple[Tensor, Tensor]:
        gates = x @ self.weight_ih.T + self.bias_ih + h @ self.weight_hh.T + self.bias_hh
        i, f, g, o = gates.chunk(4, dim=-1)
        c = f.sigmoid() * c + i.sigmoid() * g.tanh()
        h = o.sigmoid() * c.tanh()
        return h, c


class LSTMRegressor:
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1):
        self.hidden_size = hidden_size
        self.cells = [
            LSTMCell(input_size if layer == 0 else hidden_size, hidden_size) for layer in range(num_layers)
        ]
        self.fc = nn.Linear(hidden_size, 1)

    def __call__(self, x: Tensor) -> Tensor:
        batch, seq_len, _ = x.shape
        hs = [Tensor.zeros(batch, self.hidden_size) for _ in self.cells]
        cs = [Tensor.zeros(batch, self.hidden_size) for _ in self.cells]
        for t in range(seq_len):
            inp = x[:, t, :]
            for layer, cell in enumerate(self.cells):
                hs[layer], cs[layer] = cell(inp, hs[layer], cs[layer])
                inp = hs[layer]
        return self.fc(inp).squeeze(-1)


# --------------------------------------------------------------------------------------
# Fixed feature set
# --------------------------------------------------------------------------------------
# The model ships with a fixed, automatic feature set — no user-selectable features.
# Close is the only price input; Month and Day-of-week are included as cyclical
# (sin/cos) encodings so the model sees, e.g., December and January as adjacent
# rather than as raw integers ~11 apart. Raw calendar integers (Year, Day-of-month)
# are intentionally left out: Year has no repeating pattern an LSTM can learn from
# and would dominate min-max scaling with a huge, non-cyclical range, and
# day-of-month has little repeatable relationship to next-day price.
DISPLAY_FEATURES = ["Close", "Month (sin/cos)", "Day of week (sin/cos)"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["Close"] = df["Close"]

    month = df.index.month.to_numpy()
    out["Month sin"] = np.sin(2 * np.pi * month / 12)
    out["Month cos"] = np.cos(2 * np.pi * month / 12)

    dow = df.index.dayofweek.to_numpy()  # Monday=0 ... Sunday=6
    out["DOW sin"] = np.sin(2 * np.pi * dow / 7)
    out["DOW cos"] = np.cos(2 * np.pi * dow / 7)

    # Target: next day's close (the value we want to predict).
    out["Next Close"] = df["Close"].shift(-1)
    return out.dropna()


# --------------------------------------------------------------------------------------
# Data / ML helpers
# --------------------------------------------------------------------------------------
@st.cache_data(ttl="1h", max_entries=50, show_spinner=False)
def download_stock(ticker: str, start: date, end: date) -> pd.DataFrame:
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw[["Open", "High", "Low", "Close", "Volume"]].dropna()
    raw.index = pd.to_datetime(raw.index).tz_localize(None)
    return raw


def make_sequences(features: np.ndarray, target: np.ndarray, seq_len: int):
    xs, ys, rows = [], [], []
    for t in range(seq_len - 1, len(features)):
        xs.append(features[t - seq_len + 1 : t + 1])
        ys.append(target[t])
        rows.append(t)
    return np.asarray(xs), np.asarray(ys), np.asarray(rows)


def minmax_fit(arr: np.ndarray):
    lo = arr.min(axis=0)
    hi = arr.max(axis=0)
    span = np.where(hi - lo == 0, 1.0, hi - lo)
    return lo, span


def minmax_apply(arr: np.ndarray, lo: np.ndarray, span: np.ndarray) -> np.ndarray:
    return (arr - lo) / span


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    mae = float(np.abs(y_true - y_pred).mean())
    rmse = float(np.sqrt(((y_true - y_pred) ** 2).mean()))
    safe = np.where(y_true == 0, np.nan, y_true)
    mape = float(np.nanmean(np.abs((y_true - y_pred) / safe)) * 100)
    return {"R²": r2, "MAE": mae, "RMSE": rmse, "MAPE (%)": mape}


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    hidden_size: int,
    num_layers: int,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    seed: int,
    progress_cb=None,
):
    Tensor.manual_seed(seed)
    model = LSTMRegressor(X_train.shape[2], hidden_size, num_layers)
    opt = Adam(get_parameters(model), lr=learning_rate)
    rng = np.random.default_rng(seed)
    history: list[float] = []

    # TinyJit captures the compute graph the first time it's called and replays the
    # compiled kernels on every subsequent call, instead of tinygrad re-tracing the
    # whole LSTM (a Python-level loop over every timestep and layer) from scratch on
    # every batch. This is the single biggest speedup available here — JIT requires
    # every call to see tensors of the same fixed shape, so batches must all be the
    # same size (see the drop of the ragged final batch below).
    @TinyJit
    def train_step(xb: Tensor, yb: Tensor) -> Tensor:
        opt.zero_grad()
        loss = ((model(xb) - yb) ** 2).mean()
        loss.backward()
        opt.step()
        return loss.realize()

    n_full_batches = len(X_train) // batch_size
    if n_full_batches == 0:
        n_full_batches = 1
        batch_size = len(X_train)

    with train_ctx():
        for epoch in range(epochs):
            order = rng.permutation(len(X_train))
            epoch_loss_sum = 0.0
            for b in range(n_full_batches):
                batch = order[b * batch_size : (b + 1) * batch_size]
                xb = Tensor(X_train[batch].astype(np.float32))
                yb = Tensor(y_train[batch].astype(np.float32))
                epoch_loss_sum += float(train_step(xb, yb).numpy())
            epoch_loss = epoch_loss_sum / n_full_batches
            history.append(epoch_loss)
            if progress_cb is not None:
                progress_cb(epoch + 1, epochs, epoch_loss)

    return model, history


def predict(model: LSTMRegressor, X: np.ndarray) -> np.ndarray:
    return model(Tensor(X.astype(np.float32))).numpy().ravel()


# --------------------------------------------------------------------------------------
# Linear regression baseline
# --------------------------------------------------------------------------------------
# Same train/test windows and same min-max scaling as the LSTM, so the comparison is
# apples-to-apples. Each (seq_len, n_features) window is flattened into one row so
# linear regression sees exactly the same information the LSTM does. A small L2 penalty
# (ridge) is added for numerical stability — with seq_len * n_features columns and a
# modest number of rows, plain OLS can be poorly conditioned or even underdetermined.
_RIDGE_ALPHA = 1e-3


def fit_linear_regression(X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    n = len(X_train)
    X_flat = X_train.reshape(n, -1)
    X_design = np.hstack([X_flat, np.ones((n, 1))])  # bias/intercept column
    n_cols = X_design.shape[1]
    # Ridge closed form: w = (XᵀX + αI)⁻¹Xᵀy, solved via lstsq on the augmented system
    # for better numerical stability than an explicit matrix inverse.
    reg = np.sqrt(_RIDGE_ALPHA) * np.eye(n_cols)
    reg[-1, -1] = 0.0  # don't regularize the intercept
    X_aug = np.vstack([X_design, reg])
    y_aug = np.concatenate([y_train, np.zeros(n_cols)])
    weights, *_ = np.linalg.lstsq(X_aug, y_aug, rcond=None)
    return weights


def predict_linear_regression(weights: np.ndarray, X: np.ndarray) -> np.ndarray:
    n = len(X)
    X_flat = X.reshape(n, -1)
    X_design = np.hstack([X_flat, np.ones((n, 1))])
    return X_design @ weights


# --------------------------------------------------------------------------------------
# Naive baseline: predict next day's close as simply today's close (no learning at all).
# --------------------------------------------------------------------------------------
# This isolates how much of a fancier model's apparent skill is just autocorrelation —
# daily closes move slowly day to day, so "tomorrow = today" is a surprisingly strong,
# zero-parameter predictor. If a model can't beat this, its added complexity (and any
# real "prediction") isn't earning its keep.
def predict_naive(X: np.ndarray, close_idx: int) -> np.ndarray:
    # Each window's last timestep is "today" — its Close value is the naive prediction
    # for tomorrow. Still in scaled space; caller unscales like every other model here.
    return X[:, -1, close_idx]


def to_excel_bytes(pred_df: pd.DataFrame, metrics: dict[str, float], raw_df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        pred_df.to_excel(writer, sheet_name="predictions", index=False)
        pd.DataFrame(
            {"metric": list(metrics.keys()), "value": list(metrics.values())}
        ).to_excel(writer, sheet_name="metrics", index=False)
        raw_df.to_excel(writer, sheet_name="raw data")
    return buffer.getvalue()


# --------------------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------------------
st.session_state.setdefault("raw_df", None)
st.session_state.setdefault("results", None)
st.session_state.setdefault("ticker", None)

# --------------------------------------------------------------------------------------
# Sidebar: data + model configuration
# --------------------------------------------------------------------------------------
with st.sidebar:
    st.header("Data")
    ticker = st.text_input("Ticker", value="AAPL").strip().upper()
    start_date = st.date_input("Start date", value=date(2024, 1, 1))
    end_date = st.date_input("End date", value=date(2025, 1, 1))
    download_clicked = st.button("Download data", icon=":material/download:", type="primary")

    st.divider()
    st.header("Model parameters")
    seq_len = st.slider("Sequence length (days)", 3, 60, 10)
    hidden_size = st.slider("Hidden size", 4, 128, 32, step=4)
    num_layers = st.slider("LSTM layers", 1, 3, 1)
    learning_rate = st.number_input("Learning rate", min_value=1e-5, max_value=1.0, value=1e-2, format="%.5f")
    epochs = st.slider("Epochs", 1, 200, 40)
    batch_size = st.slider("Batch size", 4, 256, 32, step=4)
    train_ratio = st.slider("Train split", 0.5, 0.95, 0.8, step=0.05)
    seed = st.number_input("Random seed", min_value=0, max_value=9999, value=42)
    train_clicked = st.button("Train model", icon=":material/play_arrow:")

if download_clicked:
    if not ticker:
        st.sidebar.error("Enter a ticker symbol.")
    elif start_date >= end_date:
        st.sidebar.error("Start date must be before end date.")
    else:
        with st.spinner(f"Downloading {ticker}..."):
            try:
                st.session_state.raw_df = download_stock(ticker, start_date, end_date)
                st.session_state.ticker = ticker
                st.session_state.results = None
            except Exception as exc:  # noqa: BLE001
                st.sidebar.error(f"Download failed: {exc}")

# --------------------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------------------
st.title("tinygrad LSTM stock forecaster")
st.caption("Download a stock, engineer features, train a tinygrad LSTM, then inspect and export the results.")

raw_df = st.session_state.raw_df
downloaded_ticker = st.session_state.ticker or ticker

if raw_df is None:
    st.info("Choose a ticker and date range in the sidebar, then click **Download data**.")
    st.stop()

if raw_df.empty:
    st.warning("No rows were returned for that ticker and date range.")
    st.stop()

with st.container(border=True):
    st.subheader(f"{downloaded_ticker} · {raw_df.index.min():%Y-%m-%d} → {raw_df.index.max():%Y-%m-%d}")
    with st.container(horizontal=True):
        st.metric("Rows", f"{len(raw_df):,}", border=True)
        st.metric("Last close", f"{raw_df['Close'].iloc[-1]:,.2f}", border=True)
        st.metric("Features used", f"{len(DISPLAY_FEATURES)}", border=True)
    st.dataframe(raw_df.tail(10))

# --------------------------------------------------------------------------------------
# Features (fixed, automatic — no user selection)
# --------------------------------------------------------------------------------------
st.subheader("Features")
st.caption(
    "Fixed feature set, built automatically: **Close**, plus **Month** and **Day-of-week** "
    "each encoded as sin/cos pairs so the model reads the calendar cyclically. "
    "Target is **next day's Close**."
)
with st.container(border=True):
    st.write(", ".join(DISPLAY_FEATURES))

# --------------------------------------------------------------------------------------
# Train
# --------------------------------------------------------------------------------------
if train_clicked:
    feature_df = build_features(raw_df)
    if len(feature_df) <= seq_len + 5:
        st.error("Not enough rows for the chosen date range and sequence length.")
        st.stop()

    feature_cols = [c for c in feature_df.columns if c != "Next Close"]
    X_raw = feature_df[feature_cols].to_numpy(dtype=float)
    y_raw = feature_df["Next Close"].to_numpy(dtype=float)

    split = int(len(feature_df) * train_ratio)

    feat_lo, feat_span = minmax_fit(X_raw[:split])
    targ_lo, targ_span = minmax_fit(y_raw[:split].reshape(-1, 1))

    X_scaled = minmax_apply(X_raw, feat_lo, feat_span)
    y_scaled = minmax_apply(y_raw.reshape(-1, 1), targ_lo, targ_span).ravel()

    X_seq, y_seq, rows = make_sequences(X_scaled, y_scaled, seq_len)
    train_mask = rows < split
    if train_mask.sum() == 0 or (~train_mask).sum() == 0:
        st.error("Train split leaves no sequences on one side. Adjust the split or sequence length.")
        st.stop()

    X_train, y_train = X_seq[train_mask], y_seq[train_mask]
    X_test, y_test = X_seq[~train_mask], y_seq[~train_mask]
    test_rows = rows[~train_mask]

    st.write(f"Training on {len(X_train)} windows · testing on {len(X_test)} windows")
    progress = st.progress(0.0, text="Training...")

    def _cb(epoch: int, total: int, loss: float):
        progress.progress(epoch / total, text=f"Epoch {epoch}/{total} · MSE {loss:.6f}")

    model, history = train_model(
        X_train,
        y_train,
        hidden_size=int(hidden_size),
        num_layers=int(num_layers),
        learning_rate=float(learning_rate),
        epochs=int(epochs),
        batch_size=int(batch_size),
        seed=int(seed),
        progress_cb=_cb,
    )
    progress.empty()

    y_pred_scaled = predict(model, X_test)
    y_pred = (y_pred_scaled.reshape(-1, 1) * targ_span + targ_lo).ravel()
    y_true = (y_test.reshape(-1, 1) * targ_span + targ_lo).ravel()

    metrics = regression_metrics(y_true, y_pred)
    pred_df = pd.DataFrame(
        {
            "Date": feature_df.index[test_rows],
            "Actual": y_true,
            "Predicted": y_pred,
            "Residual": y_true - y_pred,
        }
    )

    # Linear regression baseline — identical windows, split, and scaling as the LSTM.
    lr_weights = fit_linear_regression(X_train, y_train)
    lr_pred_scaled = predict_linear_regression(lr_weights, X_test)
    lr_pred = (lr_pred_scaled.reshape(-1, 1) * targ_span + targ_lo).ravel()

    lr_metrics = regression_metrics(y_true, lr_pred)
    lr_pred_df = pd.DataFrame(
        {
            "Date": feature_df.index[test_rows],
            "Actual": y_true,
            "Predicted": lr_pred,
            "Residual": y_true - lr_pred,
        }
    )

    # Naive baseline — identical windows/split/scaling; zero parameters, no training.
    close_idx = feature_cols.index("Close")
    naive_pred_scaled = predict_naive(X_test, close_idx)
    naive_pred = (naive_pred_scaled.reshape(-1, 1) * targ_span + targ_lo).ravel()

    naive_metrics = regression_metrics(y_true, naive_pred)
    naive_pred_df = pd.DataFrame(
        {
            "Date": feature_df.index[test_rows],
            "Actual": y_true,
            "Predicted": naive_pred,
            "Residual": y_true - naive_pred,
        }
    )

    st.session_state.results = {
        "lstm": {
            "metrics": metrics,
            "predictions": pred_df,
            "history": history,
        },
        "linear": {
            "metrics": lr_metrics,
            "predictions": lr_pred_df,
            "weights": lr_weights,
        },
        "naive": {
            "metrics": naive_metrics,
            "predictions": naive_pred_df,
        },
        "features": feature_cols,
        "seq_len": seq_len,
        "ticker": downloaded_ticker,
        "raw_rows": len(feature_df),
        "train_windows": len(X_train),
        "test_windows": len(X_test),
    }

# --------------------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------------------
results = st.session_state.results
if results is None:
    st.info("Configure the model in the sidebar, then click **Train model**.")
    st.stop()

st.subheader("Results")

MODEL_OPTIONS = {
    "LSTM": "lstm",
    "Linear regression (baseline)": "linear",
    "Naive (today's close, baseline)": "naive",
}

st.caption("Challenge: can your LSTM beat both baselines below?")
model_choice = st.radio(
    "Model",
    options=list(MODEL_OPTIONS),
    horizontal=True,
    label_visibility="collapsed",
)
selected = MODEL_OPTIONS[model_choice]
selected_results = results[selected]

metrics = selected_results["metrics"]
with st.container(horizontal=True):
    for name, value in metrics.items():
        st.metric(name, f"{value:,.4f}", border=True)
features_clause = (
    "using today's Close only, no training"
    if selected == "naive"
    else f"using features: {', '.join(results['features'])}"
)
st.caption(
    f"{results['ticker']} · {model_choice} · {features_clause} · "
    f"{results['train_windows']} train / {results['test_windows']} test windows"
)

if selected == "linear":
    st.markdown("**Fitted formula**")
    st.caption(
        "The model is a single linear combination of every scaled feature at every lag "
        "in the window (no nonlinearity anywhere) — fit in min-max-scaled space, the "
        "same space the model is trained in."
    )
    st.latex(
        r"\hat{y} = b + \sum_{t=1}^{T} \sum_{f=1}^{F} w_{t,f} \cdot x_{t,f}"
    )
    st.caption(
        f"where $T = {results['seq_len']}$ (lag days in the window), "
        f"$F = {len(results['features'])}$ (features: {', '.join(results['features'])}), "
        r"$x_{t,f}$ is feature $f$ at lag $t$, $w_{t,f}$ its fitted weight, and $b$ the intercept."
    )

    weights = selected_results["weights"]
    seq_len_ = results["seq_len"]
    feat_names = results["features"]
    n_feat_ = len(feat_names)
    bias = weights[-1]
    coef_grid = weights[:-1].reshape(seq_len_, n_feat_)
    lag_labels = [seq_len_ - t for t in range(seq_len_)]

    # Close is the dominant, most interpretable feature — show its per-lag coefficient
    # as a small readable table by default, rather than dumping all T*F weights on load.
    if "Close" in feat_names:
        close_idx = feat_names.index("Close")
        close_coefs = coef_grid[:, close_idx]
        coef_df = pd.DataFrame(
            {
                "Lag (days back)": lag_labels,
                "Weight on Close": close_coefs,
            }
        )
        with st.container(border=True):
            st.write(f"Intercept $b$ = `{bias:.5f}`")
            st.caption("Weight on the Close feature at each lag position (oldest day in the window first, most recent last):")
            st.dataframe(coef_df, hide_index=True)

    with st.expander("Show all feature weights (Close, Month, Day-of-week)"):
        st.caption(
            "Every fitted weight $w_{t,f}$, one row per lag day and one column per "
            "feature. All features are min-max scaled to [0, 1] before fitting, so "
            "weights are directly comparable to each other in magnitude — a feature "
            "the model leans on more will tend to have larger |weight| across its lags."
        )
        full_coef_df = pd.DataFrame(coef_grid, columns=feat_names)
        full_coef_df.insert(0, "Lag (days back)", lag_labels)
        st.dataframe(full_coef_df, hide_index=True)

        # Collapse T*F weights into one number per feature so students can see at a
        # glance whether the calendar features (Month/DOW) carry real weight or the
        # model is effectively ignoring them in favor of Close.
        importance = pd.DataFrame(
            {
                "Feature": feat_names,
                "Total |weight| across all lags": np.abs(coef_grid).sum(axis=0),
            }
        ).sort_values("Total |weight| across all lags", ascending=False)
        st.markdown("**Which features does the model actually lean on?**")
        st.caption(
            "Sum of |weight| across all lag days, per feature — a rough measure of "
            "how much total influence each feature has on the prediction."
        )
        chart_display = importance.set_index("Feature")
        st.bar_chart(chart_display)
        st.dataframe(importance, hide_index=True)



pred_df = selected_results["predictions"]

if selected == "linear":
    # A linear model's actual "linear-ness" doesn't show up in a predicted-vs-date line
    # chart — each point there comes from a different 10-day window sliding forward, so
    # the line zigzags with the input data even though the model itself is linear. The
    # honest way to see the linearity is predicted vs. actual: for a linear fit, points
    # cluster around the y = x line, and the tightness of that cluster *is* the fit
    # quality (this is what R² is measuring).
    st.markdown("**Predicted vs. actual (this is where the fit is genuinely linear)**")
    st.caption(
        "Each point is one prediction. A perfect model would place every point exactly "
        "on the dashed y = x line — this scatter shows the true linear relationship "
        "the model is fitting, unlike the price-over-time chart below."
    )
    lo = float(min(pred_df["Actual"].min(), pred_df["Predicted"].min()))
    hi = float(max(pred_df["Actual"].max(), pred_df["Predicted"].max()))
    scatter_fig = go.Figure()
    scatter_fig.add_trace(
        go.Scatter(
            x=pred_df["Actual"], y=pred_df["Predicted"], mode="markers", name="Predictions"
        )
    )
    scatter_fig.add_trace(
        go.Scatter(
            x=[lo, hi], y=[lo, hi], mode="lines", name="y = x (perfect fit)",
            line=dict(dash="dash", color="gray"),
        )
    )
    scatter_fig.update_layout(
        xaxis_title="Actual next-day close",
        yaxis_title="Predicted next-day close",
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(scatter_fig)

fig = go.Figure()
fig.add_trace(go.Scatter(x=pred_df["Date"], y=pred_df["Actual"], name="Actual", mode="lines"))
fig.add_trace(go.Scatter(x=pred_df["Date"], y=pred_df["Predicted"], name="Predicted", mode="lines"))
fig.update_layout(
    title=f"Actual vs predicted next-day close · {model_choice}",
    xaxis_title="Date",
    yaxis_title="Price",
    hovermode="x unified",
    margin=dict(l=10, r=10, t=50, b=10),
)
st.plotly_chart(fig)

if selected == "lstm":
    chart_col, loss_col = st.columns(2)
else:
    # Neither baseline has a training loss curve — linear regression is a closed-form
    # solve, and naive persistence has no parameters to train at all — so the residuals
    # panel gets the full row instead of splitting with an empty chart.
    chart_col = st.container()
    loss_col = None

with chart_col:
    with st.container(border=True):
        st.markdown("**Residuals**")
        residual_fig = go.Figure(
            go.Scatter(x=pred_df["Date"], y=pred_df["Residual"], mode="lines", name="Residual")
        )
        residual_fig.add_hline(y=0, line_dash="dash", line_color="gray")
        residual_fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), yaxis_title="Actual - predicted")
        st.plotly_chart(residual_fig)

if loss_col is not None:
    with loss_col:
        with st.container(border=True):
            st.markdown("**Training loss**")
            loss_fig = go.Figure(
                go.Scatter(y=results["lstm"]["history"], mode="lines", name="MSE")
            )
            loss_fig.update_layout(
                xaxis_title="Epoch", yaxis_title="MSE", margin=dict(l=10, r=10, t=10, b=10)
            )
            st.plotly_chart(loss_fig)

with st.container(border=True):
    st.subheader("Predictions")
    st.dataframe(pred_df, hide_index=True)
    st.download_button(
        "Export to Excel",
        data=to_excel_bytes(pred_df, metrics, raw_df),
        file_name=f"{results['ticker']}_{selected}_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        icon=":material/download:",
    )