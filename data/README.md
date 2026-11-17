# Data

Raw data is not committed. Only this README is tracked.

| Field | Value |
|---|---|
| Kaggle slug | `thedevastator/online-retail-transaction-data` |
| License | "other" (as listed on the Kaggle page; not confirmed) |
| Declared origin | Invoice-line sales of a UK-based online gift retailer, Dec 2010 to Dec 2011 |
| Synthetic check | Not declared synthetic. Same dataset as [dynamic-pricing-elasticity](https://github.com/Patotricks15/dynamic-pricing-elasticity). |
| Read date | 2026-11-17 |
| Local file | `data/raw/online_retail.csv` (git-ignored) |

## Cost assumption

The data has no cost column. Unit cost is assumed to be a fixed share of each product's median weekly price. The base case uses **60%** (a 40% gross margin). The notebook shows 50% and 70%.

## Download

```bash
export KAGGLE_API_TOKEN=<your token>   # never commit or write the token to a file
bash scripts/download_data.sh
```
