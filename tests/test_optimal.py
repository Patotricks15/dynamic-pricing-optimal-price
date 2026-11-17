import numpy as np
import pandas as pd
import pytest

from dp_pricing.optimal import expected_profit, optimal_price, recommend


def test_interior_optimum_matches_closed_form():
    rec = optimal_price(elasticity=-2.0, unit_cost=10.0, p_min=1.0, p_max=100.0)
    assert rec.price == pytest.approx(20.0)
    assert rec.interior and not rec.clipped


def test_optimum_is_clipped_to_observed_band():
    rec = optimal_price(elasticity=-2.0, unit_cost=10.0, p_min=1.0, p_max=15.0)
    assert rec.price == pytest.approx(15.0)
    assert rec.clipped


def test_inelastic_demand_has_no_interior_optimum():
    rec = optimal_price(elasticity=-0.5, unit_cost=10.0, p_min=1.0, p_max=30.0)
    assert not rec.interior
    assert rec.price == pytest.approx(30.0)


def test_closed_form_maximises_profit_over_a_grid():
    eps, cost, p0, q0 = -2.5, 6.0, 10.0, 100.0
    grid = np.linspace(cost + 0.01, 40.0, 4000)
    best = grid[np.argmax([expected_profit(p, cost, p0, q0, eps) for p in grid])]
    rec = optimal_price(eps, cost, p_min=cost, p_max=40.0)
    assert best == pytest.approx(rec.price, rel=5e-3)


def test_recommend_uses_product_elasticity_and_cost_share():
    bands = pd.DataFrame(
        {"p0": [10.0], "q0": [50.0], "p_min": [5.0], "p_max": [30.0]}, index=["A"]
    )
    out = recommend(bands, pd.Series({"A": -2.0}), cost_share=0.5)
    assert out.loc["A", "unit_cost"] == pytest.approx(5.0)
    assert out.loc["A", "price"] == pytest.approx(10.0)
    assert out.loc["A", "uplift"] == pytest.approx(0.0, abs=1e-9)
