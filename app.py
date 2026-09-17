"""
Demand Forecast Agent — Streamlit Frontend
--------------------------------------------------
This is the UI layer. It expects the backend modules to expose the
functions listed in the "BACKEND CONTRACT" section below. Until your
teammate's real modules are ready, this file runs on lightweight mock
data so you can build and demo the UI independently.

BACKEND CONTRACT (share this with your teammate):

    data_loader.py
        load_data(filepath: str) -> pd.DataFrame
            Returns raw dataframe with at least: date, store_id, item_id, sales

    preprocessing.py
        get_series(df: pd.DataFrame, store_id: str, item_id: str) -> pd.DataFrame
            Returns a single filtered, resampled, date-indexed series
            with a "sales" column, sorted by date, no missing dates.

    eda.py
        summary_stats(series: pd.DataFrame) -> dict
            Returns {"total_sales": int, "avg_daily": float,
                     "min": float, "max": float, "std": float}

    trend_seasonality.py
        detect_trend(series: pd.DataFrame) -> dict
            Returns {"direction": "increasing"|"decreasing"|"stable",
                     "slope": float, "pct_change": float}
        detect_seasonality(series: pd.DataFrame) -> dict
            Returns {"has_weekly": bool, "has_yearly": bool,
                     "weekday_avg": dict, "monthly_avg": dict}

    anomaly.py
        detect_anomalies(series: pd.DataFrame) -> pd.DataFrame
            Returns the series with an added boolean column "is_anomaly"

    forecasting.py
        forecast(series: pd.DataFrame, horizon: int, confidence: float)
            -> pd.DataFrame
            Returns future dates with columns: date, forecast, lower, upper

    inventory.py
        recommend(series: pd.DataFrame, forecast_df: pd.DataFrame,
                   lead_time_days: int, service_level_z: float) -> dict
            Returns {"safety_stock": float, "reorder_point": float,
                     "reorder_qty": float, "avg_daily_demand": float}

    insights.py
        generate_insights(stats, trend, seasonality, anomalies, inventory)
            -> list[str]
            Returns plain-English bullet points grounded in the numbers above.

Swap the "MOCK BACKEND" block below for real imports once ready:
    from data_loader import load_data
    from preprocessing import get_series
    from eda import summary_stats
    from trend_seasonality import detect_trend, detect_seasonality
    from anomaly import detect_anomalies
    from forecasting import forecast
    from inventory import recommend
    from insights import generate_insights
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta

# ============================================================
# MOCK BACKEND (delete this block once real modules are ready)
# ============================================================
USE_MOCK_BACKEND = True

if USE_MOCK_BACKEND:
    def load_data(filepath):
        rng = pd.date_range("2019-01-01", "2023-12-31", freq="D")
        stores = [f"store_{i}" for i in range(1, 6)]
        items = [f"item_{i}" for i in range(1, 6)]
        rows = []
        for s in stores:
            for it in items:
                base = np.random.uniform(20, 60)
                trend = np.linspace(0, np.random.uniform(5, 20), len(rng))
                weekly = 8 * np.sin(2 * np.pi * rng.dayofweek / 7)
                yearly = 10 * np.sin(2 * np.pi * rng.dayofyear / 365)
                noise = np.random.normal(0, 4, len(rng))
                sales = np.clip(base + trend + weekly + yearly + noise, 0, None).round()
                rows.append(pd.DataFrame({
                    "date": rng, "store_id": s, "item_id": it, "sales": sales
                }))
        return pd.concat(rows, ignore_index=True)

    def get_series(df, store_id, item_id):
        sub = df[(df["store_id"] == store_id) & (df["item_id"] == item_id)].copy()
        sub = sub.sort_values("date").set_index("date")
        return sub[["sales"]]

    def summary_stats(series):
        s = series["sales"]
        return {"total_sales": int(s.sum()), "avg_daily": float(s.mean()),
                "min": float(s.min()), "max": float(s.max()), "std": float(s.std())}

    def detect_trend(series):
        s = series["sales"].values
        x = np.arange(len(s))
        slope = np.polyfit(x, s, 1)[0]
        pct_change = (s[-30:].mean() - s[:30].mean()) / max(s[:30].mean(), 1e-6) * 100
        direction = "increasing" if slope > 0.01 else "decreasing" if slope < -0.01 else "stable"
        return {"direction": direction, "slope": float(slope), "pct_change": float(pct_change)}

    def detect_seasonality(series):
        s = series.copy()
        s["weekday"] = s.index.dayofweek
        s["month"] = s.index.month
        return {
            "has_weekly": True, "has_yearly": True,
            "weekday_avg": s.groupby("weekday")["sales"].mean().to_dict(),
            "monthly_avg": s.groupby("month")["sales"].mean().to_dict(),
        }

    def detect_anomalies(series):
        s = series.copy()
        roll_mean = s["sales"].rolling(14, center=True, min_periods=1).mean()
        roll_std = s["sales"].rolling(14, center=True, min_periods=1).std().fillna(1)
        z = (s["sales"] - roll_mean) / roll_std.replace(0, 1)
        s["is_anomaly"] = z.abs() > 2.5
        return s

    def forecast(series, horizon, confidence):
        s = series["sales"]
        last_val = s.tail(30).mean()
        std = s.std()
        future_dates = pd.date_range(s.index[-1] + timedelta(days=1), periods=horizon, freq="D")
        z = 1.96 if confidence >= 0.95 else 1.64
        trend_bump = np.linspace(0, std * 0.1, horizon)
        weekly = 8 * np.sin(2 * np.pi * future_dates.dayofweek / 7)
        preds = last_val + trend_bump + weekly
        margin = z * std * np.sqrt(np.arange(1, horizon + 1)) / 5
        return pd.DataFrame({
            "date": future_dates, "forecast": preds,
            "lower": np.clip(preds - margin, 0, None), "upper": preds + margin
        })

    def recommend(series, forecast_df, lead_time_days, service_level_z):
        avg_daily = forecast_df["forecast"].mean()
        std_daily = series["sales"].std()
        safety_stock = service_level_z * std_daily * np.sqrt(lead_time_days)
        reorder_point = avg_daily * lead_time_days + safety_stock
        reorder_qty = avg_daily * lead_time_days * 1.5
        return {"safety_stock": safety_stock, "reorder_point": reorder_point,
                "reorder_qty": reorder_qty, "avg_daily_demand": avg_daily}

    def generate_insights(stats, trend, seasonality, anomalies, inventory):
        n_anom = int(anomalies["is_anomaly"].sum())
        top_weekday = max(seasonality["weekday_avg"], key=seasonality["weekday_avg"].get)
        weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        return [
            f"Demand is **{trend['direction']}**, changing by {trend['pct_change']:.1f}% "
            f"comparing the last 30 days to the first 30 days on record.",
            f"Average daily demand is **{stats['avg_daily']:.1f} units**, "
            f"with day-to-day variability (std dev) of {stats['std']:.1f}.",
            f"**{weekday_names[top_weekday]}** is consistently the strongest day of the week for sales.",
            f"**{n_anom} anomalous days** were flagged out of {len(anomalies)} — likely promotions or "
            f"stockout events worth reviewing.",
            f"To maintain service levels, keep a reorder point of **{inventory['reorder_point']:.0f} units** "
            f"and a safety stock buffer of **{inventory['safety_stock']:.0f} units**.",
        ]
# ============================================================
# END MOCK BACKEND
# ============================================================


# ---------------- PAGE CONFIG & STYLE ----------------
st.set_page_config(
    page_title="Demand Forecast Agent",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main > div { padding-top: 1.5rem; }
    .kpi-card {
        background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        text-align: left;
    }
    .kpi-label { font-size: 0.78rem; color: #9ca3af; text-transform: uppercase; letter-spacing: 0.04em; }
    .kpi-value { font-size: 1.6rem; font-weight: 700; color: #f9fafb; margin-top: 0.15rem; }
    .kpi-sub { font-size: 0.8rem; color: #6ee7b7; margin-top: 0.1rem; }
    .kpi-sub.negative { color: #fca5a5; }
    .insight-box {
        background: #0f172a10;
        border-left: 4px solid #4f46e5;
        padding: 0.7rem 1rem;
        border-radius: 6px;
        margin-bottom: 0.6rem;
    }
    h1, h2, h3 { font-family: 'Segoe UI', sans-serif; }
</style>
""", unsafe_allow_html=True)

st.title("📦 Demand Forecast Agent")
st.caption("Explainable demand forecasting & inventory recommendations — DATA → ANALYSIS → FORECAST → REASONING → RECOMMENDATION")

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("⚙️ Controls")

    data_path = st.text_input("Dataset path", value="data/retail_sales.csv",
                               help="Path to your CSV, e.g. data/retail_sales.csv")

    with st.spinner("Loading dataset..."):
        df = load_data(data_path)

    stores = sorted(df["store_id"].unique())
    items = sorted(df["item_id"].unique())

    store_id = st.selectbox("Store", stores, index=0)
    item_id = st.selectbox("Item", items, index=0)

    st.divider()
    st.subheader("Forecast settings")
    horizon = st.slider("Forecast horizon (days)", 7, 90, 30, step=7)
    confidence = st.select_slider("Confidence level", options=[0.80, 0.90, 0.95], value=0.95)

    st.divider()
    st.subheader("Inventory settings")
    lead_time = st.slider("Lead time (days)", 1, 30, 7)
    service_level = st.select_slider(
        "Service level (z-score)",
        options=[1.28, 1.64, 1.96, 2.33],
        value=1.64,
        format_func=lambda z: {1.28: "90%", 1.64: "95%", 1.96: "97.5%", 2.33: "99%"}[z],
    )

    st.divider()
    run = st.button("🚀 Run analysis", use_container_width=True, type="primary")

# ---------------- MAIN PIPELINE ----------------
if run or "series" not in st.session_state:
    series = get_series(df, store_id, item_id)
    stats = summary_stats(series)
    trend = detect_trend(series)
    seasonality = detect_seasonality(series)
    anomalies = detect_anomalies(series)
    forecast_df = forecast(series, horizon, confidence)
    inventory = recommend(series, forecast_df, lead_time, service_level)
    insights = generate_insights(stats, trend, seasonality, anomalies, inventory)

    st.session_state.update(dict(
        series=series, stats=stats, trend=trend, seasonality=seasonality,
        anomalies=anomalies, forecast_df=forecast_df, inventory=inventory, insights=insights
    ))

series = st.session_state["series"]
stats = st.session_state["stats"]
trend = st.session_state["trend"]
seasonality = st.session_state["seasonality"]
anomalies = st.session_state["anomalies"]
forecast_df = st.session_state["forecast_df"]
inventory = st.session_state["inventory"]
insights = st.session_state["insights"]

# ---------------- KPI ROW ----------------
k1, k2, k3, k4 = st.columns(4)
trend_arrow = "▲" if trend["direction"] == "increasing" else "▼" if trend["direction"] == "decreasing" else "→"
trend_class = "" if trend["direction"] == "increasing" else "negative" if trend["direction"] == "decreasing" else ""

with k1:
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-label">Avg Daily Demand</div>
        <div class="kpi-value">{stats['avg_daily']:.1f}</div>
        <div class="kpi-sub">units / day</div></div>""", unsafe_allow_html=True)
with k2:
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-label">Trend</div>
        <div class="kpi-value">{trend_arrow} {trend['direction'].title()}</div>
        <div class="kpi-sub {trend_class}">{trend['pct_change']:+.1f}% vs. start</div></div>""", unsafe_allow_html=True)
with k3:
    n_anom = int(anomalies["is_anomaly"].sum())
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-label">Anomalies Flagged</div>
        <div class="kpi-value">{n_anom}</div>
        <div class="kpi-sub">of {len(anomalies)} days</div></div>""", unsafe_allow_html=True)
with k4:
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-label">Reorder Point</div>
        <div class="kpi-value">{inventory['reorder_point']:.0f}</div>
        <div class="kpi-sub">units</div></div>""", unsafe_allow_html=True)

st.write("")

# ---------------- TABS ----------------
tab_overview, tab_trend, tab_anomaly, tab_forecast, tab_inventory, tab_insights = st.tabs(
    ["📊 Overview", "📈 Trend & Seasonality", "🚨 Anomalies", "🔮 Forecast", "📦 Inventory", "💡 AI Insights"]
)

with tab_overview:
    st.subheader(f"Historical sales — {store_id} / {item_id}")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series.index, y=series["sales"], mode="lines",
                              line=dict(color="#4f46e5", width=1.5), name="Sales"))
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10),
                       template="plotly_dark", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total sales", f"{stats['total_sales']:,}")
    c2.metric("Min day", f"{stats['min']:.0f}")
    c3.metric("Max day", f"{stats['max']:.0f}")
    c4.metric("Std dev", f"{stats['std']:.1f}")

with tab_trend:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Weekly seasonality")
        weekday_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        wk = seasonality["weekday_avg"]
        fig_w = go.Figure(go.Bar(
            x=[weekday_names[k] for k in sorted(wk)],
            y=[wk[k] for k in sorted(wk)],
            marker_color="#6366f1"))
        fig_w.update_layout(height=340, template="plotly_dark", margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_w, use_container_width=True)
    with col2:
        st.subheader("Yearly seasonality")
        mo = seasonality["monthly_avg"]
        month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        fig_m = go.Figure(go.Bar(
            x=[month_names[k-1] for k in sorted(mo)],
            y=[mo[k] for k in sorted(mo)],
            marker_color="#22d3ee"))
        fig_m.update_layout(height=340, template="plotly_dark", margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_m, use_container_width=True)

    st.info(f"**Trend slope:** {trend['slope']:.3f} units/day → classified as **{trend['direction']}**")

with tab_anomaly:
    st.subheader("Flagged anomalies")
    fig_a = go.Figure()
    fig_a.add_trace(go.Scatter(x=series.index, y=series["sales"], mode="lines",
                                line=dict(color="#4f46e5", width=1), name="Sales"))
    anom_pts = anomalies[anomalies["is_anomaly"]]
    fig_a.add_trace(go.Scatter(x=anom_pts.index, y=anom_pts["sales"], mode="markers",
                                marker=dict(color="#ef4444", size=8, symbol="x"), name="Anomaly"))
    fig_a.update_layout(height=420, template="plotly_dark", margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified")
    st.plotly_chart(fig_a, use_container_width=True)
    st.dataframe(anom_pts.reset_index().rename(columns={"date": "Date", "sales": "Sales"}),
                 use_container_width=True, height=220)

with tab_forecast:
    st.subheader(f"{horizon}-day forecast ({int(confidence*100)}% confidence interval)")
    fig_f = go.Figure()
    hist_tail = series.tail(90)
    fig_f.add_trace(go.Scatter(x=hist_tail.index, y=hist_tail["sales"], mode="lines",
                                line=dict(color="#94a3b8", width=1.5), name="Historical"))
    fig_f.add_trace(go.Scatter(x=forecast_df["date"], y=forecast_df["forecast"], mode="lines",
                                line=dict(color="#22c55e", width=2), name="Forecast"))
    fig_f.add_trace(go.Scatter(
        x=pd.concat([forecast_df["date"], forecast_df["date"][::-1]]),
        y=pd.concat([forecast_df["upper"], forecast_df["lower"][::-1]]),
        fill="toself", fillcolor="rgba(34,197,94,0.15)", line=dict(color="rgba(0,0,0,0)"),
        name=f"{int(confidence*100)}% interval"))
    fig_f.update_layout(height=440, template="plotly_dark", margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified")
    st.plotly_chart(fig_f, use_container_width=True)

with tab_inventory:
    st.subheader("Inventory recommendation")
    i1, i2, i3, i4 = st.columns(4)
    i1.metric("Avg daily demand (forecast)", f"{inventory['avg_daily_demand']:.1f}")
    i2.metric("Safety stock", f"{inventory['safety_stock']:.0f}")
    i3.metric("Reorder point", f"{inventory['reorder_point']:.0f}")
    i4.metric("Reorder quantity", f"{inventory['reorder_qty']:.0f}")

    with st.expander("How these numbers were calculated"):
        st.markdown(f"""
- **Safety stock** = z × σ(demand) × √(lead time) = {service_level} × {series['sales'].std():.1f} × √{lead_time} ≈ **{inventory['safety_stock']:.0f} units**
- **Reorder point** = avg daily demand × lead time + safety stock ≈ **{inventory['reorder_point']:.0f} units**
- **Reorder quantity** ≈ 1.5 × (avg daily demand × lead time) ≈ **{inventory['reorder_qty']:.0f} units**
- Lead time assumed: **{lead_time} days**, service level: **{ {1.28:'90%',1.64:'95%',1.96:'97.5%',2.33:'99%'}[service_level] }**
""")

with tab_insights:
    st.subheader("Auto-generated insights")
    st.caption("Generated from the actual computed numbers above — not hallucinated.")
    for point in insights:
        st.markdown(f'<div class="insight-box">{point}</div>', unsafe_allow_html=True)
