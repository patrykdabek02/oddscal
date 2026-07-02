# oddscal

De-vigging, probability calibration metrics and Kelly staking for sports
betting research. Pure NumPy, no heavy dependencies.

Extracted from the experimental part of a bachelor's thesis on football
betting market efficiency ([market-efficiency-ml](https://github.com/patrykdabek02/market-efficiency-ml)).

## Install

```bash
pip install git+https://github.com/patrykdabek02/oddscal
```

## Quick start

```python
import oddscal as oc

# 1. Remove the bookmaker margin (Shin's method corrects
#    the favourite-longshot bias)
odds = [1.30, 5.50, 11.00]          # home / draw / away
p_fair = oc.implied_probabilities(odds, method="shin")
print(p_fair)                        # [0.751 0.169 0.080], sums to 1.0
print(oc.overround(odds))            # 1.042 -> ~4.2% margin

# 2. Score a probabilistic model (3-way football forecast)
#    random baseline: Brier 0.2222, log-loss ln(3)=1.0986
brier = oc.brier_score(y_true, proba)
ece   = oc.expected_calibration_error(y_true, proba, class_index=2)

# 3. Size the stake
f = oc.kelly_fraction(p=0.55, odds=2.10, fraction=0.25, cap=0.05)
```

## What's inside

| Module | Contents |
|---|---|
| `oddscal.devig` | `implied_probabilities` (multiplicative / Shin 1993 / power), `overround` |
| `oddscal.metrics` | multiclass `brier_score`, `log_loss`, `expected_calibration_error` |
| `oddscal.kelly` | `kelly_fraction` (fractional, capped), `log_growth` |
| `oddscal.plots` | `reliability_diagram` (optional, needs matplotlib) |

Works for any number of outcomes: 2 (tennis), 3 (football 1X2), n (outrights).

## References

- Shin, H. S. (1993). Measuring the incidence of insider trading in a market
  for state-contingent claims. *The Economic Journal*, 103(420).
- Kelly, J. L. (1956). A new interpretation of information rate.
  *Bell System Technical Journal*, 35(4).

## Disclaimer

Research tooling, not investment advice. The thesis this code comes from
concludes that top-league betting markets are too efficient to beat.

## License

MIT
