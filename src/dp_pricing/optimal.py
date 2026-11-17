"""Profit-maximising price under constant elasticity, within an observed price band."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

BAND_QUANTILES = (0.05, 0.95)


@dataclass(frozen=True)
class Recommendation:
    price: float
    interior: bool
    clipped: bool


def optimal_price(elasticity: float, unit_cost: float, p_min: float, p_max: float) -> Recommendation:
    """Return the profit-maximising price for constant elasticity demand.

    With ``q = q0 (p/p0)^ε`` and ``ε < −1``, profit ``(p − c)·q`` is maximised
    at ``p* = ε/(1+ε)·c``. The result is clipped to ``[p_min, p_max]``.
    For ``ε ≥ −1`` there is no interior maximum, so profit rises with price
    until the band edge and the upper bound is returned.
    """
    if elasticity >= -1:
        return Recommendation(price=p_max, interior=False, clipped=True)
    p_star = elasticity / (1 + elasticity) * unit_cost
    price = min(max(p_star, p_min), p_max)
    return Recommendation(price=price, interior=True, clipped=price != p_star)


def expected_profit(price: float, unit_cost: float, p0: float, q0: float, elasticity: float) -> float:
    """Weekly profit at ``price`` under ``q = q0 (p/p0)^ε``."""
    quantity = q0 * (price / p0) ** elasticity
    return (price - unit_cost) * quantity


def product_bands(panel: pd.DataFrame, quantiles: tuple[float, float] = BAND_QUANTILES) -> pd.DataFrame:
    """Per-product reference price, reference quantity, and observed price band.

    ``p0`` is the median weekly price and ``q0`` the median weekly quantity.
    The band is the weekly price range between the given quantiles, so
    recommendations are not extrapolated beyond prices seen in the data.
    """
    grouped = panel.groupby("StockCode")
    low, high = quantiles
    return pd.DataFrame(
        {
            "p0": grouped["price"].median(),
            "q0": grouped["quantity"].median(),
            "p_min": grouped["price"].quantile(low),
            "p_max": grouped["price"].quantile(high),
        }
    )


def recommend(bands: pd.DataFrame, elasticity: pd.Series, cost_share: float) -> pd.DataFrame:
    """Recommend a price per product.

    ``elasticity`` is indexed by StockCode. Unit cost is assumed to be
    ``cost_share × p0``, because the data has no cost column.
    """
    if not 0 < cost_share < 1:
        raise ValueError("cost_share must be between 0 and 1")
    rows = []
    for code, band in bands.iterrows():
        eps = float(elasticity.loc[code])
        cost = cost_share * band["p0"]
        rec = optimal_price(eps, cost, band["p_min"], band["p_max"])
        now = expected_profit(band["p0"], cost, band["p0"], band["q0"], eps)
        after = expected_profit(rec.price, cost, band["p0"], band["q0"], eps)
        rows.append(
            {
                "StockCode": code,
                "elasticity": eps,
                "unit_cost": cost,
                "p0": band["p0"],
                "price": rec.price,
                "interior": rec.interior,
                "clipped": rec.clipped,
                "profit_now": now,
                "profit_rec": after,
                "uplift": after / now - 1,
            }
        )
    return pd.DataFrame(rows).set_index("StockCode")
