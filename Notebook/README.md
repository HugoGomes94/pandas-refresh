# Telco Customer Churn — pandas Reproduction

**Problem:** Reproduce the headline figures from a widely-used public analysis of the Telco Customer Churn dataset, writing all code independently. The goal was not novel insight — it was to verify my own pandas output against a known-correct reference and investigate any discrepancy.

**Data:** [Telco Customer Churn (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — 7,043 rows, 21 columns, one row per customer.
Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` to `data/` before running. The data file is not committed.

**Approach:** Read the reference notebook's output cells only, recorded its reported figures, then wrote my own pandas from scratch and compared. Churn recoded to a 1/0 flag via `.map()`, rates computed with `groupby().mean()`, paired aggregations with named `.agg()`.

---

## Data quality assessment

| Check | Result |
| :---- | :---- |
| Row count | 7,043 |
| Duplicate rows (`duplicated()`) | 0 |
| `customerID` uniqueness | 7,043 distinct — one row per customer, clean denominators |
| True nulls (`isna()`) | 0 across all columns |
| **Blank strings** | **11 rows in `TotalCharges` contain `" "`, not a null** |

The `TotalCharges` finding is the one worth flagging. `isna()` reports zero missing values because an empty string is valid text — so the column silently loads with dtype `object` rather than numeric. Any arithmetic on it fails or produces wrong results until those rows are handled. All 11 correspond to customers with `tenure = 0`, i.e. new customers who have not yet been billed.

Also noted: several service columns carry a third category (`"No internet service"` / `"No phone service"`) alongside `Yes`/`No`. Treating these as a plain `No` would conflate "declined the add-on" with "not eligible for it."

---

## Results

All eleven figures matched the reference.

| Figure | Reference | Mine | Match |
| :---- | :---- | :---- | :---- |
| Overall churn rate | ~26.5% (1,869 / 7,043) | 26.5% | Yes |
| Churn — month-to-month | ~42.7% | 42.7% | Yes |
| Churn — one year | ~11.3% | 11.3% | Yes |
| Churn — two year | ~2.8% | 2.8% | Yes |
| Avg tenure — churned | ~18 months | 18 | Yes |
| Avg tenure — retained | ~38 months | 38 | Yes |
| Avg monthly charges — churned | ~$74 | $74 | Yes |
| Avg monthly charges — retained | ~$61 | $61 | Yes |
| Churn — fibre optic | ~42% | 42% | Yes |
| Churn — DSL | ~19% | 19% | Yes |
| Churn — no internet | ~7.4% | 7.4% | Yes |

### What the numbers say

Contract type is the dominant signal: month-to-month customers churn at roughly **fifteen times** the rate of two-year contract customers. Fibre optic customers churn at more than double the DSL rate despite being the premium product — and they also carry the higher monthly charge, so price sensitivity and service dissatisfaction are both plausible explanations that this dataset cannot separate.

Churned customers show shorter tenure (18 vs 38 months) and higher monthly charges ($74 vs $61). Both are descriptive, not causal — tenure in particular is partly definitional, since a customer who churns early cannot accumulate tenure.

---

## Stack

Python, pandas

## How to run

1. `pip install -r requirements.txt`
2. Download the dataset from the Kaggle link above into `data/`
3. Open `notebooks/01-telco-eda.ipynb` and run all cells

---

## Limitations

- **Single snapshot, no date column.** There is no way to observe *when* a customer churned, so time-based validation is impossible on this data. Any model trained on it would need a random split, which risks leakage for a time-dependent process.
- **`tenure` is a proxy, not a clean feature.** It is partly determined by the outcome it would be used to predict.
- **Class imbalance.** The 26.5% / 73.5% split means accuracy is a misleading metric here — a model predicting "no churn" for everyone scores 73.5%.
- **No causal claims.** Every figure above is a segment average. Confounding between contract type, tenure and service tier is untested.
- **All figures matched on first attempt**, so no discrepancy investigation was required. The reconciliation skill this project was meant to exercise remains untested.

---

*Phase 0 of a structured self-study path from data analyst to ML practitioner. This repo is a pandas fluency checkpoint, not a modelling project.*








