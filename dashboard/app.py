import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

st.set_page_config(page_title="Trader Sentiment Dashboard", layout="wide")

# Resolve data path relative to this script's location, so it works
# regardless of which folder you launch `streamlit run` from
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'

@st.cache_data
def load_data():
    daily = pd.read_csv(DATA_DIR / 'daily_account_features.csv')
    account_summary = pd.read_csv(DATA_DIR / 'account_summary_segments.csv')
    daily['date'] = pd.to_datetime(daily['date'])
    return daily, account_summary

daily, account_summary = load_data()

st.title("Trader Performance vs Market Sentiment (Hyperliquid)")
st.caption("Primetrade.ai Assignment — Interactive exploration of Fear/Greed sentiment vs. trader behavior")

# Sidebar filters
st.sidebar.header("Filters")
sentiment_filter = st.sidebar.multiselect(
    "Sentiment regime",
    options=daily['sentiment_binary'].unique(),
    default=daily['sentiment_binary'].unique()
)

accounts = ['All'] + sorted(daily['Account'].unique().tolist())
account_filter = st.sidebar.selectbox("Account", accounts)

filtered = daily[daily['sentiment_binary'].isin(sentiment_filter)]
if account_filter != 'All':
    filtered = filtered[filtered['Account'] == account_filter]

# Top-level metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Trader-Days", len(filtered))
col2.metric("Median Daily PnL", f"${filtered['daily_pnl'].median():,.0f}")
col3.metric("Day-Level Win Rate", f"{filtered['is_winning_day'].mean():.1%}")
col4.metric("Avg Drawdown", f"${filtered['drawdown'].mean():,.0f}")

st.divider()

# Row 1: PnL and win rate by sentiment
c1, c2 = st.columns(2)
with c1:
    st.subheader("Daily PnL by Sentiment")
    fig, ax = plt.subplots()
    sns.boxplot(data=filtered, x='sentiment_binary', y='daily_pnl', showfliers=False, ax=ax)
    ax.axhline(0, color='red', linestyle='--', alpha=0.5)
    st.pyplot(fig)

with c2:
    st.subheader("Day-Level Win Rate by Sentiment")
    win_rates = filtered.groupby('sentiment_binary')['is_winning_day'].mean()
    fig, ax = plt.subplots()
    ax.bar(win_rates.index, win_rates.values, color=['#e74c3c', '#f39c12', '#27ae60'][:len(win_rates)])
    ax.set_ylim(0, 1)
    st.pyplot(fig)

st.divider()

# Row 2: Segmentation explorer
st.subheader("Trader Segmentation")
seg_choice = st.selectbox("Segment by:", ['exposure_segment', 'frequency_segment', 'consistency_segment'])

c3, c4 = st.columns(2)
with c3:
    fig, ax = plt.subplots()
    account_summary.groupby(seg_choice)['total_pnl'].median().plot(kind='bar', ax=ax, color='#3498db')
    ax.set_title(f'Median Total PnL by {seg_choice}')
    ax.set_ylabel('PnL ($)')
    plt.xticks(rotation=15)
    st.pyplot(fig)

with c4:
    fig, ax = plt.subplots()
    account_summary.groupby(seg_choice)['worst_drawdown'].median().plot(kind='bar', ax=ax, color='#e67e22')
    ax.set_title(f'Median Worst Drawdown by {seg_choice}')
    ax.set_ylabel('Drawdown ($)')
    plt.xticks(rotation=15)
    st.pyplot(fig)

st.divider()

# Raw data explorer
st.subheader("Raw Data")
st.dataframe(filtered.sort_values('date', ascending=False), use_container_width=True)

st.divider()
st.caption("Key finding: only 4 of 32 accounts show a genuine sentiment-reactive pattern "
           "(higher volume + worse drawdown in Fear), all from the Infrequent + Inconsistent segment. "
           "See README.md for full methodology.")
