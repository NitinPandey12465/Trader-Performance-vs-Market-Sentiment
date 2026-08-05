# Trader Performance vs Market Sentiment (Hyperliquid)

**Primetrade.ai — Data Science/Analytics Intern Round-0 Assignment**

An end-to-end analysis of how Bitcoin Fear/Greed sentiment relates to trader behavior,
risk, and profitability on Hyperliquid — covering data preparation, statistically
rigorous hypothesis testing, trader segmentation, unsupervised clustering, and
predictive modeling.

---

## TL;DR — Key Finding

Sentiment shows **statistically detectable but practically small** relationships with
trader drawdown and volume at the population level. Critically, these pooled effects
**do not generalize across individual traders** — only **4 of 32 accounts (12.5%)**
show a genuine sentiment-reactive pattern (higher volume + worse drawdown during Fear),
and all four belong to the same behavioral segment: **Infrequent, Inconsistent
traders**. This reframes the actionable takeaway from "sentiment drives the market" to
"a small, identifiable subset of traders needs sentiment-aware risk management."

---

## Setup

```bash
git clone <this-repo-url>
cd primetrade-sentiment
pip install -r requirements.txt
jupyter notebook notebooks/trader_sentiment_analysis.ipynb
```

## Repo Structure
```
data/          raw + processed CSVs (fear_greed_index.csv, historical_data.csv,
                daily_account_features.csv, account_summary_segments.csv)
notebooks/     trader_sentiment_analysis.ipynb — full analysis, run top to bottom
plots/         all saved PNG figures referenced below
dashboard/     app.py — Streamlit dashboard (bonus)
src/           reusable pipeline scripts
```

---

## Part A — Data Preparation

### Datasets
- **Hyperliquid trade log** (`historical_data.csv`): 211,224 rows x 16 columns, **32
  unique trader accounts**, 246 unique coins, 2023-05-01 to 2025-05-01. Zero nulls, zero
  duplicate rows.
- **Bitcoin Fear/Greed Index** (`fear_greed_index.csv`): 2,644 rows x 4 columns
  (timestamp, value 0-100, classification, date), 2018-02-01 to 2025-05-02. **5-class**
  classification (Extreme Fear, Fear, Neutral, Greed, Extreme Greed) — not binary, as
  the assignment brief implied.

### Key Data Limitation: No Leverage Field
The assignment brief references "leverage distribution," but the raw trade log has **no
margin, collateral, or leverage column**. `Start Position` is a token-quantity position
tracker, not notional/margin — leverage is not derivable from this data, even via
aggregation. We substitute **Size USD (notional trade value)** as an exposure proxy
throughout, and disclose this substitution rather than approximate a number we cannot
support.

### Temporal Alignment
Timestamps parsed from `Timestamp IST` (`DD-MM-YYYY HH:MM`), aggregated to daily grain.
One date (2024-10-26) was missing from the sentiment index; forward-filled from the
prior day (negligible — 1 of 480 trading days).

### Feature Engineering (-> `daily_account_features.csv`, 2,341 account-days)
Daily PnL, trade count, total/average notional volume, fees, long/short ratio,
cumulative PnL, running-peak drawdown proxy — all at `(Account, date)` grain.

**Two definitions of "win rate," reported separately:**
- **Trade-level win rate** (% of individual closing fills with positive PnL) — mean **84.8%**
- **Day-level win rate** (% of days the account was net profitable overall) — **62.7%**

The 22-point gap is a real finding: many accounts show mostly-winning individual fills
while still closing net negative on the day — consistent with cutting winners short and
letting losers run.

---

## Part B — Empirical Analysis

### Statistical Approach
Mann-Whitney U tests (non-parametric — PnL/volume/size are heavily right-skewed) across
6 metric comparisons, with **Bonferroni correction** (adjusted alpha = 0.0083) and
**rank-biserial effect sizes** reported alongside every p-value, since statistical
significance alone doesn't indicate practical importance at n=2,341.

### Results: Fear vs. Greed (binary)

| Metric | p-value | Sig. (a=0.05) | Sig. (Bonferroni) | Effect size |
|---|---|---|---|---|
| Daily PnL | 0.058 | No | No | 0.05 (negligible) |
| Drawdown | 0.002 | Yes | **Yes** | 0.07 (negligible) |
| Trade count | 0.037 | Yes | No | -0.06 (negligible) |
| Total volume | 0.007 | Yes | **Yes** | -0.07 (negligible) |
| Avg trade size | 0.465 | No | No | -0.02 (negligible) |
| Long/short ratio | 0.042 | Yes | No | -0.06 (negligible) |

All effect sizes are **negligible** by conventional thresholds (|r| < 0.1), including
the two Bonferroni survivors.

### Critical Check: Does This Hold at the Individual-Trader Level?

Pooled tests treat every trade as independent, but trades from the same account are
correlated (pseudo-replication risk). We checked the two Bonferroni-surviving findings
against each of the 31 accounts with data in both regimes:

- **Drawdown**: only **8 of 31 accounts (26%)** show worse drawdown in Fear. Median
  across-account difference is exactly 0.
- **Volume**: only **15 of 31 accounts (48%)** trade more in Fear — a coin flip.

**The pooled significance does not generalize across the trader population.** The same
check on long/short bias found an identical pattern (7 of 31 consistent).

### Reframing: Who Is Actually Sentiment-Reactive?

Accounts showing *both* worse drawdown *and* higher volume in Fear: **4 of 32 (12.5%)**,
all belonging to the **Infrequent + Inconsistent** segment (see below). This is the real,
targetable signal — not a population-wide effect.

### Continuous Validation (Spearman, sentiment_value 0-100)

| Metric | Spearman rho | p-value | Significant |
|---|---|---|---|
| Daily PnL | 0.066 | 0.001 | Yes |
| Drawdown | 0.100 | <0.001 | Yes |
| Trade count | -0.038 | 0.065 | No |
| Total volume | -0.058 | 0.005 | Yes |
| Long/short ratio | -0.044 | 0.054 | No |

Direction matches the binary tests, triangulating the finding across two independent
methods. Magnitudes remain small throughout.

### Segmentation (n=32 accounts, median splits)

| Segment | PnL comparison | Drawdown comparison |
|---|---|---|
| High vs Low Exposure | High: median $163K vs Low: $108K | High: ~2x deeper (-$34.5K vs -$16.9K) |
| Frequent vs Infrequent | Frequent: median $130K vs $91K | Comparable |
| Consistent vs Inconsistent Winner | Consistent: median $151K vs $70K | Consistent: deeper (-$21.5K vs -$10.6K) |

High Exposure and "Consistent Winner" both show a real risk/reward tradeoff — higher
typical PnL paired with meaningfully deeper typical drawdown, not a free lunch.

### Figures (in `/plots`)
- `pnl_winrate_by_sentiment.png` — PnL distribution + day-win-rate by regime
- `drawdown_behavior_by_sentiment.png` — drawdown frequency, trade count, volume by regime
- `segmentation_pnl_vs_risk.png` — PnL vs. drawdown across all three segments
- `drawdown_fear_vs_greed.png` — drawdown boxplot with outliers visible

---

## Part C — Actionable Strategy Rules

**Rule 1 — Targeted intervention, not a blanket sentiment rule.**
Sentiment does not broadly affect trader behavior or risk (see individual-trader check
above). Only ~12.5% of accounts show a genuine reactive pattern, and they share one
profile: Infrequent, Inconsistent traders. Recommendation: flag accounts in this segment
for automatic exposure caps or a trade-frequency cooldown specifically during
Fear/Extreme Fear regimes, rather than applying any restriction firm-wide.

**Rule 2 — Pair high exposure with drawdown-triggered de-risking, not a hard cap.**
High Exposure accounts show higher typical PnL (median $163K vs $108K) but ~2x deeper
typical drawdown. A blanket exposure cap would remove a segment that is, on balance,
performing better. Recommendation: auto-reduce position size after a position moves a
defined threshold (e.g. 15%) against peak equity, preserving upside while limiting the
downside tail — instead of capping exposure outright.

**Rule 3 — Don't conflate "consistent" with "safe."**
Consistent Winners (high day-win-rate) also carry deeper average drawdowns than
Inconsistent traders. Recommendation: risk monitoring should track drawdown
independently of win-rate-based labels — a high win rate is not evidence of low risk in
this dataset.

---

## Bonus 1 — Trader Archetype Clustering

K-Means with k selected via silhouette score (tested k=2-7; k=2 gave silhouette=0.357,
clearly the best — higher k values scored 0.19-0.26, indicating overfitting noise into
fake clusters at n=32). We chose the statistically supported k rather than a larger,
more "narratively convenient" k.

**Cluster 0 (n=28)**: typical accounts — 80 days active, $4,025 avg daily PnL, -$40,381
worst drawdown.
**Cluster 1 (n=4)**: high-volume power users — 251 median trades/day, $28,983 avg daily
PnL (7x higher), but also -$131,461 worst drawdown (~3x deeper). 3 of 4 are High
Exposure; all 4 are Consistent Winners.

This cluster is **structurally distinct** from the 4 sentiment-reactive accounts found
in Part B — clustering picked up overall risk magnitude, not sentiment-driven behavioral
change. These are two independent risk dimensions worth monitoring separately.

Figures: `clustering_k_selection.png`, `kmeans_clusters_pnl_vs_drawdown.png`

---

## Bonus 2 — Predictive Modeling

**Task**: predict next-day account profitability (binary: net win/loss) from same-day
sentiment + behavioral features. XGBoost and Random Forest, 80/20 stratified split
(n=1,558 after cleaning).

| Model | Accuracy | Baseline (majority class) | AUC-ROC |
|---|---|---|---|
| XGBoost | 71.2% | 70.8% | 0.650 |
| Random Forest | 72.8% | 70.8% | 0.652 |

**Honest conclusion: weak predictive power.** Both models barely beat the naive
baseline on accuracy (+0.3 to +2 points), and AUC = 0.65 indicates modest, not strong,
discriminative ability. More tellingly, both models struggle specifically at recalling
losing days (XGBoost: 19% recall on class 0; Random Forest: 12%) — the operationally
important case. Feature importance is diffuse across all 9 features (0.08-0.16 range,
no dominant predictor); `sentiment_value` contributes moderately (rank 4-5 of 9) but
does not stand out. This is consistent with Part B's small effect sizes: next-day
profitability is not strongly predictable from sentiment + simple behavioral aggregates
alone — likely driven more by per-trade market conditions and entry/exit skill than by
regime-level sentiment.

Figure: `feature_importance.png`

---

## Bonus 3 — Streamlit Dashboard

*(In progress — see `dashboard/app.py`)*

---

## Methodological Notes & Limitations

1. **Multiple comparisons**: 6 hypothesis tests were run; Bonferroni correction applied
   throughout rather than reporting raw p-values as if independent.
2. **Pseudo-replication**: daily account-level observations from the same trader are not
   independent; pooled tests can overstate significance relative to what holds across
   the actual 32-trader population. We explicitly checked and reported this rather than
   relying on pooled p-values alone.
3. **Small N for clustering/segmentation**: 32 accounts limits how many clusters can be
   reliably resolved; k=2 was chosen based on silhouette score rather than a
   narratively convenient higher k.
4. **Leverage unavailable**: substituted with notional exposure (Size USD); disclosed
   explicitly rather than approximated.
5. **Weak predictive signal**: reported honestly (AUC = 0.65) rather than overstated.

---

## Author's Note

This analysis prioritized **honest, defensible conclusions over impressive-sounding
ones**. Several early findings (a pooled long/short bias result, an initial drawdown
claim) did not survive individual-account or multiple-comparison scrutiny and were
revised or caveated rather than kept as headline claims. The final, more nuanced
conclusion — that sentiment-driven risk is concentrated in a small, identifiable trader
subset rather than distributed across the population — is a stronger and more
actionable result than the original broader claim would have been.
