from __future__ import annotations

from pathlib import Path
import json
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title='AI Hedge Fund (Free)', layout='wide')
st.title('AI-Native Hedge Fund Prototype (Centralized + Auditable)')

out = Path('outputs')
metrics_path = out / 'metrics.csv'
equity_path = out / 'equity_curve.csv'
weights_path = out / 'weights.csv'
decision_path = out / 'last_decision.json'
audit_path = out / 'audit' / 'events.jsonl'
heartbeat_path = out / 'heartbeat.json'

if metrics_path.exists() and equity_path.exists() and weights_path.exists():
    metrics = pd.read_csv(metrics_path)
    equity = pd.read_csv(equity_path, parse_dates=[0])
    weights = pd.read_csv(weights_path, index_col=0, parse_dates=True)

    m = metrics.iloc[0].to_dict()
    cols = st.columns(5)
    cols[0].metric('CAGR', f"{m.get('cagr', 0.0):.2%}")
    cols[1].metric('Sharpe', f"{m.get('sharpe', 0.0):.2f}")
    cols[2].metric('Max Drawdown', f"{m.get('max_drawdown', 0.0):.2%}")
    cols[3].metric('Annual Vol', f"{m.get('annual_vol', 0.0):.2%}")
    cols[4].metric('Avg Turnover', f"{m.get('avg_daily_turnover', 0.0):.3f}")

    equity.columns = ['date', 'equity']
    fig_eq = px.line(equity, x='date', y='equity', title='Backtest Equity Curve')
    st.plotly_chart(fig_eq, use_container_width=True)

    latest_w = weights.iloc[-1].sort_values(ascending=False)
    fig_w = px.bar(latest_w, title='Backtest Latest Target Weights')
    st.plotly_chart(fig_w, use_container_width=True)

if decision_path.exists():
    st.subheader('Latest Live Decision')
    decision = json.loads(decision_path.read_text(encoding='utf-8'))
    st.json(decision)
else:
    st.info('No live decision yet. Run: python scripts/run_daily.py --dry-run')

if heartbeat_path.exists():
    st.subheader("Heartbeat")
    st.json(json.loads(heartbeat_path.read_text(encoding="utf-8")))

if audit_path.exists():
    st.subheader('Audit Trail (Tail)')
    lines = audit_path.read_text(encoding='utf-8').splitlines()[-30:]
    audit_rows = [json.loads(line) for line in lines if line.strip()]
    st.dataframe(pd.DataFrame(audit_rows))
else:
    st.info('No audit log yet. Decision cycles will create outputs/audit/events.jsonl')

# ============================================================
# AI INTELLIGENCE LAYER (new additive section)
# ============================================================
with st.expander(" AI Intelligence Layer", expanded=False):

    col1, col2 = st.columns(2)

    # Panel A: LLM Provider Status
    with col1:
        st.subheader("LLM Provider Status")
        try:
            from llm_router import get_provider_stats
            stats = get_provider_stats()
            if stats:
                rows = []
                for provider, data in stats.items():
                    failure_rate = (data["failures"] / data["calls"] * 100) if data["calls"] > 0 else 0
                    rows.append({"Provider": provider, "Calls": data["calls"],
                                 "Failures": data["failures"], "Avg Latency (ms)": round(data.get("avg_latency", 0)),
                                 "Failure Rate %": round(failure_rate, 1)})
                df_stats = pd.DataFrame(rows)
                st.dataframe(df_stats, use_container_width=True)
            else:
                st.info("No LLM calls made yet this session.")
        except Exception:
            st.info("LLM router not active.")

    # Panel B: AI Alpha Signals
    with col2:
        st.subheader("AI Alpha Signals")
        try:
            import json, os
            if os.path.exists("outputs/ai_alpha_latest.json"):
                with open("outputs/ai_alpha_latest.json") as f:
                    ai_alpha = json.load(f)
                tickers = list(ai_alpha.keys())
                combined = [ai_alpha[t].get("combined_alpha", 0) for t in tickers]
                df_alpha = pd.DataFrame({"Ticker": tickers, "AI Alpha Signal": combined})
                st.bar_chart(df_alpha.set_index("Ticker"))
            else:
                st.info("No AI alpha data yet. Enable ai_alpha.enabled in config.")
        except Exception:
            st.info("AI alpha data unavailable.")

    col3, col4 = st.columns(2)

    # Panel C: Bayesian Portfolio Weights
    with col3:
        st.subheader("Bayesian Portfolio Weights")
        try:
            if os.path.exists("outputs/bayesian_weights_latest.json"):
                with open("outputs/bayesian_weights_latest.json") as f:
                    bw = json.load(f)
                mean_w = bw.get("mean_weights", {})
                std_w = bw.get("std_weights", {})
                if mean_w:
                    df_bw = pd.DataFrame({
                        "Ticker": list(mean_w.keys()),
                        "Mean Weight": list(mean_w.values()),
                        "Std": [std_w.get(t, 0) for t in mean_w.keys()]
                    })
                    st.dataframe(df_bw, use_container_width=True)
            else:
                st.info("No Bayesian weights yet. Enable bayesian_optimizer.enabled in config.")
        except Exception:
            st.info("Bayesian weights unavailable.")

    # Panel D: Regime Intelligence
    with col4:
        st.subheader("Regime Intelligence")
        try:
            if os.path.exists("outputs/regime_latest.json"):
                with open("outputs/regime_latest.json") as f:
                    reg = json.load(f)
                st.metric("Current Regime", reg.get("most_likely", "Unknown"))
                st.metric("Bull Probability", f"{reg.get('bull', 0):.1%}")
                st.metric("Bear Probability", f"{reg.get('bear', 0):.1%}")
                st.metric("Crisis Probability", f"{reg.get('crisis', 0):.1%}")
            else:
                st.info("No regime data yet.")
        except Exception:
            st.info("Regime data unavailable.")
