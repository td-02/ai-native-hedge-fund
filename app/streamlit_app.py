from __future__ import annotations

from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title='AI Hedge Fund (Free)', layout='wide')
st.title('AI-Native Hedge Fund Prototype (Free Only)')

out = Path('outputs')
metrics_path = out / 'metrics.csv'
equity_path = out / 'equity_curve.csv'
weights_path = out / 'weights.csv'

if not metrics_path.exists() or not equity_path.exists() or not weights_path.exists():
    st.warning('No outputs found. Run: python scripts/run_backtest.py --config configs/default.yaml')
    st.stop()

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
fig_eq = px.line(equity, x='date', y='equity', title='Equity Curve')
st.plotly_chart(fig_eq, use_container_width=True)

latest_w = weights.iloc[-1].sort_values(ascending=False)
fig_w = px.bar(latest_w, title='Latest Target Weights')
st.plotly_chart(fig_w, use_container_width=True)

st.subheader('Latest Weights Table')
st.dataframe(latest_w.to_frame('weight'))
