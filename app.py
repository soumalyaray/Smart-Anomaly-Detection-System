import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import time
import random
from datetime import datetime, timedelta #somalaya fuck you 

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AnomalyWatch",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

:root {
    --bg: #080c14;
    --surface: #0d1422;
    --surface2: #111827;
    --border: #1e2d45;
    --accent: #00f5d4;
    --accent2: #ff3b6b;
    --accent3: #f5a623;
    --text: #e2e8f0;
    --muted: #64748b;
    --normal: #00f5d4;
    --anomaly: #ff3b6b;
    --warning: #f5a623;
}

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}

.stApp {
    background: var(--bg);
    background-image:
        radial-gradient(ellipse at 20% 0%, rgba(0,245,212,0.04) 0%, transparent 60%),
        radial-gradient(ellipse at 80% 100%, rgba(255,59,107,0.04) 0%, transparent 60%);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* Header */
.main-header {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 24px 0 8px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 24px;
}
.main-header h1 {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2rem;
    letter-spacing: -0.02em;
    margin: 0;
    background: linear-gradient(135deg, var(--accent), #7eeeff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.header-badge {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: var(--accent);
    border: 1px solid var(--accent);
    padding: 3px 8px;
    border-radius: 3px;
    letter-spacing: 0.08em;
    animation: pulse-border 2s ease-in-out infinite;
}
@keyframes pulse-border {
    0%, 100% { border-color: var(--accent); box-shadow: 0 0 0 0 rgba(0,245,212,0.3); }
    50% { border-color: var(--accent); box-shadow: 0 0 0 4px rgba(0,245,212,0); }
}

/* Metric Cards */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 20px;
}
.metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px 20px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.3s ease;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
}
.metric-card.normal::before { background: var(--normal); }
.metric-card.anomaly::before { background: var(--anomaly); }
.metric-card.warning::before { background: var(--warning); }
.metric-card.info::before { background: #7c8ef7; }

.metric-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem;
    color: var(--muted);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.5rem;
    font-weight: 700;
    line-height: 1;
}
.metric-value.normal { color: var(--normal); }
.metric-value.anomaly { color: var(--anomaly); }
.metric-value.warning { color: var(--warning); }
.metric-value.info { color: #7c8ef7; }
.metric-sub {
    font-size: 0.7rem;
    color: var(--muted);
    margin-top: 4px;
    font-family: 'Space Mono', monospace;
}

/* Alert Banner */
.alert-banner {
    padding: 14px 20px;
    border-radius: 6px;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 12px;
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
}
.alert-normal {
    background: rgba(0,245,212,0.07);
    border: 1px solid rgba(0,245,212,0.3);
    color: var(--normal);
}
.alert-anomaly {
    background: rgba(255,59,107,0.1);
    border: 1px solid rgba(255,59,107,0.4);
    color: var(--anomaly);
    animation: flicker 1.5s ease-in-out infinite;
}
.alert-warning {
    background: rgba(245,166,35,0.08);
    border: 1px solid rgba(245,166,35,0.3);
    color: var(--warning);
}
@keyframes flicker {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.75; }
}

/* Status Pill */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 6px 14px;
    border-radius: 100px;
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.06em;
}
.status-normal {
    background: rgba(0,245,212,0.1);
    color: var(--normal);
    border: 1px solid rgba(0,245,212,0.3);
}
.status-anomaly {
    background: rgba(255,59,107,0.12);
    color: var(--anomaly);
    border: 1px solid rgba(255,59,107,0.4);
}
.status-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    display: inline-block;
}
.status-normal .status-dot {
    background: var(--normal);
    box-shadow: 0 0 6px var(--normal);
    animation: blink 1.8s ease-in-out infinite;
}
.status-anomaly .status-dot {
    background: var(--anomaly);
    box-shadow: 0 0 8px var(--anomaly);
    animation: blink 0.6s ease-in-out infinite;
}
@keyframes blink {
    0%, 100% { opacity: 1; } 50% { opacity: 0.3; }
}

/* Log Table */
.log-entry {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 16px;
    border-bottom: 1px solid var(--border);
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
}
.log-entry:last-child { border-bottom: none; }
.log-time { color: var(--muted); min-width: 80px; }
.log-status-tag {
    padding: 2px 8px;
    border-radius: 3px;
    font-weight: 700;
    font-size: 0.6rem;
    letter-spacing: 0.08em;
    min-width: 80px;
    text-align: center;
}
.tag-normal { background: rgba(0,245,212,0.15); color: var(--normal); }
.tag-anomaly { background: rgba(255,59,107,0.15); color: var(--anomaly); }
.log-score { color: var(--muted); margin-left: auto; }

/* Section Labels */
.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 10px;
    padding-bottom: 6px;
    border-bottom: 1px solid var(--border);
}

/* Streamlit overrides */
div[data-testid="stSlider"] label { color: var(--muted) !important; font-size: 0.75rem !important; }
div.stButton > button {
    background: var(--accent) !important;
    color: #080c14 !important;
    border: none !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.05em !important;
    border-radius: 4px !important;
    padding: 8px 20px !important;
    transition: opacity 0.2s !important;
}
div.stButton > button:hover { opacity: 0.85 !important; }
div.stSelectbox label, div.stSlider label { color: var(--muted) !important; }
</style>
""", unsafe_allow_html=True)


# ─── Data Generation ────────────────────────────────────────────────────────
def generate_sensor_data(n=300, anomaly_frac=0.08, seed=42):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 4 * np.pi, n)

    signal = (
        10 * np.sin(t) + 5 * np.sin(2.5 * t) +
        rng.normal(0, 1.2, n)
    )

    n_anomalies = int(n * anomaly_frac)
    anomaly_idx = rng.choice(n, n_anomalies, replace=False)
    signal[anomaly_idx] += rng.choice([-1, 1], n_anomalies) * rng.uniform(14, 22, n_anomalies)

    timestamps = [datetime.now() - timedelta(seconds=(n - i) * 2) for i in range(n)]
    return pd.DataFrame({
        "timestamp": timestamps,
        "value": signal,
        "is_injected": np.isin(np.arange(n), anomaly_idx),
    })


def generate_multivariate_data(n=300, anomaly_frac=0.06, seed=42):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 4 * np.pi, n)

    cpu = 40 + 20 * np.sin(t) + rng.normal(0, 4, n)
    memory = 60 + 10 * np.cos(t * 0.7) + rng.normal(0, 3, n)
    latency = 50 + 15 * np.sin(t * 1.3 + 1) + rng.normal(0, 5, n)

    n_anomalies = int(n * anomaly_frac)
    anomaly_idx = rng.choice(n, n_anomalies, replace=False)
    cpu[anomaly_idx] += rng.uniform(30, 55, n_anomalies)
    memory[anomaly_idx] += rng.uniform(20, 35, n_anomalies)
    latency[anomaly_idx] += rng.uniform(80, 150, n_anomalies)

    timestamps = [datetime.now() - timedelta(seconds=(n - i) * 2) for i in range(n)]
    return pd.DataFrame({
        "timestamp": timestamps,
        "cpu": np.clip(cpu, 0, 100),
        "memory": np.clip(memory, 0, 100),
        "latency": np.clip(latency, 0, 400),
        "is_injected": np.isin(np.arange(n), anomaly_idx),
    })


def run_isolation_forest(features, contamination=0.08):
    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    clf = IsolationForest(contamination=contamination, random_state=42, n_estimators=120)
    clf.fit(X)
    scores = clf.decision_function(X)
    labels = clf.predict(X)
    return labels, scores


# ─── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙ Controls")
    st.markdown("---")

    data_mode = st.selectbox(
        "Data Source",
        ["Sensor Signal (1D)", "Server Metrics (3D)", "Upload CSV"],
    )
    n_points = st.slider("Data Points", 100, 600, 300, step=50)
    contamination = st.slider("Contamination Rate", 0.02, 0.20, 0.08, step=0.01,
                              help="Expected fraction of anomalies")
    seed = st.slider("Random Seed", 0, 99, 42)

    st.markdown("---")
    st.markdown("### 🎛 Visualization")
    show_bands = st.checkbox("Show Anomaly Bands", True)
    show_injected = st.checkbox("Show Injected Ground Truth", True)
    live_mode = st.checkbox("Live Simulation Mode", False)

    st.markdown("---")
    st.markdown(
        "<div style='font-family:Space Mono,monospace;font-size:0.6rem;color:#64748b;'>"
        "MODEL: Isolation Forest<br>"
        "ENGINE: scikit-learn 1.x<br>"
        "VIZ: Plotly"
        "</div>",
        unsafe_allow_html=True,
    )

# ─── Header ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>⚡ AnomalyWatch</h1>
  <div class="header-badge">LIVE MONITOR</div>
</div>
""", unsafe_allow_html=True)

# ─── Data Loading ────────────────────────────────────────────────────────────
uploaded_df = None
if data_mode == "Upload CSV":
    uploaded = st.file_uploader("Upload CSV (must have a numeric column)", type=["csv"])
    if uploaded:
        try:
            uploaded_df = pd.read_csv(uploaded)
            numeric_cols = uploaded_df.select_dtypes(include=np.number).columns.tolist()
            if not numeric_cols:
                st.error("No numeric columns found.")
                st.stop()
            sel_cols = st.multiselect("Select feature columns", numeric_cols, default=numeric_cols[:1])
            if not sel_cols:
                st.stop()
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            st.stop()
    else:
        st.info("👆 Upload a CSV file to get started, or switch to a built-in data source.")
        st.stop()

# ─── Generate / Prepare Data ────────────────────────────────────────────────
if data_mode == "Sensor Signal (1D)":
    df = generate_sensor_data(n=n_points, anomaly_frac=contamination, seed=seed)
    features = df[["value"]]
    signal_cols = ["value"]

elif data_mode == "Server Metrics (3D)":
    df = generate_multivariate_data(n=n_points, anomaly_frac=contamination, seed=seed)
    features = df[["cpu", "memory", "latency"]]
    signal_cols = ["cpu", "memory", "latency"]

else:  # Upload
    df = uploaded_df.copy()
    features = df[sel_cols]
    signal_cols = sel_cols
    if "timestamp" not in df.columns:
        df["timestamp"] = [datetime.now() - timedelta(seconds=(len(df) - i) * 2) for i in range(len(df))]
    df["is_injected"] = False

# ─── Run Model ───────────────────────────────────────────────────────────────
labels, scores = run_isolation_forest(features, contamination=contamination)
df["anomaly"] = labels == -1
df["score"] = scores
df["score_norm"] = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)

n_anomalies = df["anomaly"].sum()
n_total = len(df)
pct_anomaly = n_anomalies / n_total * 100
latest_status = "ANOMALY" if df["anomaly"].iloc[-1] else "NORMAL"
latest_score = df["score_norm"].iloc[-1]

# ─── Alert Banner ────────────────────────────────────────────────────────────
if latest_status == "ANOMALY":
    st.markdown(f"""
    <div class="alert-banner alert-anomaly">
        🚨 <strong>ANOMALY DETECTED</strong> — Latest reading flagged as suspicious.
        Confidence score: {1 - latest_score:.2%}
    </div>""", unsafe_allow_html=True)
elif pct_anomaly > 12:
    st.markdown(f"""
    <div class="alert-banner alert-warning">
        ⚠ <strong>ELEVATED ANOMALY RATE</strong> — {pct_anomaly:.1f}% of readings flagged above threshold.
    </div>""", unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="alert-banner alert-normal">
        ✓ <strong>ALL SYSTEMS NORMAL</strong> — No significant anomalies in recent window.
    </div>""", unsafe_allow_html=True)

# ─── Metric Cards ───────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    status_cls = "anomaly" if latest_status == "ANOMALY" else "normal"
    st.markdown(f"""
    <div class="metric-card {status_cls}">
        <div class="metric-label">Current Status</div>
        <div class="metric-value {status_cls}">{latest_status}</div>
        <div class="metric-sub">Latest reading</div>
    </div>""", unsafe_allow_html=True)

with c2:
    score_cls = "anomaly" if latest_score < 0.35 else "normal"
    st.markdown(f"""
    <div class="metric-card {score_cls}">
        <div class="metric-label">Anomaly Score</div>
        <div class="metric-value {score_cls}">{latest_score:.3f}</div>
        <div class="metric-sub">Isolation Forest</div>
    </div>""", unsafe_allow_html=True)

with c3:
    rate_cls = "anomaly" if pct_anomaly > 10 else ("warning" if pct_anomaly > 6 else "normal")
    st.markdown(f"""
    <div class="metric-card {rate_cls}">
        <div class="metric-label">Anomaly Rate</div>
        <div class="metric-value {rate_cls}">{pct_anomaly:.1f}%</div>
        <div class="metric-sub">{n_anomalies} / {n_total} points</div>
    </div>""", unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="metric-card info">
        <div class="metric-label">Data Points</div>
        <div class="metric-value info">{n_total}</div>
        <div class="metric-sub">{data_mode}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── Main Chart ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Signal Monitor</div>', unsafe_allow_html=True)

if data_mode == "Server Metrics (3D)":
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        subplot_titles=["CPU Usage (%)", "Memory Usage (%)", "Latency (ms)"],
        vertical_spacing=0.06,
    )
    cols_cfg = [
        ("cpu", "#00f5d4", 1),
        ("memory", "#7c8ef7", 2),
        ("latency", "#f5a623", 3),
    ]

    for col, color, row in cols_cfg:
        normal_mask = ~df["anomaly"]
        anomaly_mask = df["anomaly"]

        fig.add_trace(go.Scatter(
            x=df.loc[normal_mask, "timestamp"], y=df.loc[normal_mask, col],
            mode="lines", name=f"{col} (normal)",
            line=dict(color=color, width=1.5),
            showlegend=(row == 1),
        ), row=row, col=1)

        fig.add_trace(go.Scatter(
            x=df.loc[anomaly_mask, "timestamp"], y=df.loc[anomaly_mask, col],
            mode="markers", name="Anomaly",
            marker=dict(color="#ff3b6b", size=8, symbol="x", line=dict(width=2, color="#ff3b6b")),
            showlegend=(row == 1),
        ), row=row, col=1)

    fig.update_layout(
        height=480,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,20,34,0.6)",
        font=dict(family="Space Mono", color="#64748b", size=10),
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, bgcolor="rgba(0,0,0,0)"),
    )
    for i in range(1, 4):
        fig.update_xaxes(showgrid=True, gridcolor="#1e2d45", row=i, col=1)
        fig.update_yaxes(showgrid=True, gridcolor="#1e2d45", row=i, col=1)

else:
    col_name = signal_cols[0]
    normal_mask = ~df["anomaly"]
    anomaly_mask = df["anomaly"]

    fig = go.Figure()

    # Background anomaly shading
    if show_bands:
        for idx in df[anomaly_mask].index:
            fig.add_vrect(
                x0=df.loc[idx, "timestamp"] - timedelta(seconds=1.5),
                x1=df.loc[idx, "timestamp"] + timedelta(seconds=1.5),
                fillcolor="rgba(255,59,107,0.07)",
                line_width=0,
            )

    # Ground truth markers
    if show_injected and "is_injected" in df.columns and df["is_injected"].any():
        inj = df[df["is_injected"]]
        fig.add_trace(go.Scatter(
            x=inj["timestamp"], y=inj[col_name],
            mode="markers",
            name="Injected Anomaly",
            marker=dict(color="#f5a623", size=14, symbol="circle-open",
                        line=dict(width=2, color="#f5a623")),
        ))

    # Main signal
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df[col_name],
        mode="lines", name="Signal",
        line=dict(color="#00f5d4", width=1.8),
    ))

    # Normal points
    fig.add_trace(go.Scatter(
        x=df.loc[normal_mask, "timestamp"], y=df.loc[normal_mask, col_name],
        mode="markers", name="Normal",
        marker=dict(color="#00f5d4", size=4, opacity=0.5),
    ))

    # Anomaly markers
    fig.add_trace(go.Scatter(
        x=df.loc[anomaly_mask, "timestamp"], y=df.loc[anomaly_mask, col_name],
        mode="markers", name="Detected Anomaly",
        marker=dict(color="#ff3b6b", size=11, symbol="x-open",
                    line=dict(width=2.5, color="#ff3b6b")),
    ))

    fig.update_layout(
        height=350,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,20,34,0.6)",
        font=dict(family="Space Mono", color="#64748b", size=10),
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(showgrid=True, gridcolor="#1e2d45"),
        yaxis=dict(showgrid=True, gridcolor="#1e2d45"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, bgcolor="rgba(0,0,0,0)",
                    font=dict(size=10)),
        hovermode="x unified",
    )

st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ─── Score Distribution + Recent Log ────────────────────────────────────────
col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown('<div class="section-label">Score Distribution</div>', unsafe_allow_html=True)

    score_fig = go.Figure()
    score_fig.add_trace(go.Histogram(
        x=df.loc[~df["anomaly"], "score_norm"],
        name="Normal",
        marker_color="rgba(0,245,212,0.6)",
        nbinsx=30,
        xbins=dict(start=0, end=1, size=1/30),
    ))
    score_fig.add_trace(go.Histogram(
        x=df.loc[df["anomaly"], "score_norm"],
        name="Anomaly",
        marker_color="rgba(255,59,107,0.7)",
        nbinsx=30,
        xbins=dict(start=0, end=1, size=1/30),
    ))
    score_fig.add_vline(
        x=df["score_norm"].quantile(contamination),
        line_dash="dash", line_color="#f5a623",
        annotation_text="threshold",
        annotation_font=dict(family="Space Mono", size=9, color="#f5a623"),
    )
    score_fig.update_layout(
        height=240,
        barmode="overlay",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,20,34,0.6)",
        font=dict(family="Space Mono", color="#64748b", size=10),
        margin=dict(l=0, r=0, t=8, b=0),
        xaxis=dict(title="Anomaly Score", showgrid=True, gridcolor="#1e2d45"),
        yaxis=dict(title="Count", showgrid=True, gridcolor="#1e2d45"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=9)),
    )
    st.plotly_chart(score_fig, use_container_width=True, config={"displayModeBar": False})

with col_right:
    st.markdown('<div class="section-label">Recent Event Log</div>', unsafe_allow_html=True)

    recent = df.tail(12).iloc[::-1]
    log_html = ""
    for _, row in recent.iterrows():
        is_anom = row["anomaly"]
        tag_cls = "tag-anomaly" if is_anom else "tag-normal"
        tag_txt = "ANOMALY" if is_anom else "NORMAL"
        ts = row["timestamp"].strftime("%H:%M:%S")
        sc = f"{row['score_norm']:.3f}"
        log_html += f"""
        <div class="log-entry">
            <span class="log-time">{ts}</span>
            <span class="log-status-tag {tag_cls}">{tag_txt}</span>
            <span style="color:#94a3b8;font-size:0.68rem;">score</span>
            <span class="log-score">{sc}</span>
        </div>"""

    st.markdown(
        f'<div style="background:var(--surface);border:1px solid var(--border);border-radius:8px;overflow:hidden;">{log_html}</div>',
        unsafe_allow_html=True
    )

# ─── Live Simulation Mode ────────────────────────────────────────────────────
if live_mode:
    st.markdown("---")
    st.markdown('<div class="section-label">Live Simulation</div>', unsafe_allow_html=True)

    live_placeholder = st.empty()
    status_placeholder = st.empty()
    stop_btn = st.button("⏹ Stop Simulation")

    rng = np.random.default_rng()
    live_values = list(df["value"].values[-60:]) if "value" in df.columns else [0.0] * 60
    live_labels = [False] * 60
    live_scores = [0.5] * 60

    for step in range(200):
        if stop_btn:
            break

        new_val = 10 * np.sin(step * 0.2) + 5 * np.sin(step * 0.5) + rng.normal(0, 1.2)
        is_spike = rng.random() < contamination
        if is_spike:
            new_val += rng.choice([-1, 1]) * rng.uniform(15, 22)

        live_values.append(new_val)
        live_values = live_values[-80:]
        live_labels.append(is_spike)
        live_labels = live_labels[-80:]

        xs = list(range(len(live_values)))

        live_fig = go.Figure()
        norm_x = [x for x, a in zip(xs, live_labels) if not a]
        norm_y = [y for y, a in zip(live_values, live_labels) if not a]
        anom_x = [x for x, a in zip(xs, live_labels) if a]
        anom_y = [y for y, a in zip(live_values, live_labels) if a]

        live_fig.add_trace(go.Scatter(x=norm_x, y=norm_y, mode="lines+markers",
                                       marker=dict(color="#00f5d4", size=4, opacity=0.6),
                                       line=dict(color="#00f5d4", width=1.5), name="Normal"))
        live_fig.add_trace(go.Scatter(x=anom_x, y=anom_y, mode="markers",
                                       marker=dict(color="#ff3b6b", size=12, symbol="x",
                                                   line=dict(width=2.5)), name="Anomaly"))

        live_fig.update_layout(
            height=220,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(13,20,34,0.6)",
            font=dict(family="Space Mono", color="#64748b", size=10),
            margin=dict(l=0, r=0, t=8, b=0),
            xaxis=dict(showgrid=True, gridcolor="#1e2d45", showticklabels=False),
            yaxis=dict(showgrid=True, gridcolor="#1e2d45"),
            legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=1.02),
            showlegend=True,
        )
        live_placeholder.plotly_chart(live_fig, use_container_width=True, config={"displayModeBar": False})

        pill_cls = "status-anomaly" if is_spike else "status-normal"
        pill_label = "ANOMALY DETECTED" if is_spike else "NORMAL"
        status_placeholder.markdown(
            f'<div class="status-pill {pill_cls}"><span class="status-dot"></span>{pill_label}'
            f' &nbsp;|&nbsp; t={step} &nbsp;|&nbsp; val={new_val:.2f}</div>',
            unsafe_allow_html=True,
        )
        time.sleep(0.12)

# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;font-family:Space Mono,monospace;font-size:0.6rem;color:#334155;padding:8px 0;'>"
    "AnomalyWatch · Isolation Forest · Built with Streamlit + Plotly"
    "</div>",
    unsafe_allow_html=True,
)
