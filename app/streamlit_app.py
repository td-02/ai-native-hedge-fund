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

if audit_path.exists():
    st.subheader('Audit Trail (Tail)')
    lines = audit_path.read_text(encoding='utf-8').splitlines()[-30:]
    audit_rows = [json.loads(line) for line in lines if line.strip()]
    st.dataframe(pd.DataFrame(audit_rows))
else:
    st.info('No audit log yet. Decision cycles will create outputs/audit/events.jsonl')
