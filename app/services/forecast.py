"""Bottom-up MRR forecast, with a rolling-origin backtest against simple baselines.

Model (one step):   closing(t+1) = closing(t) x NRR + new_MRR(t+1)
    NRR      = average net revenue retention over the last `window` months
               (what share of existing MRR survives a month, incl. upgrades)
    new_MRR  = linear trend fitted on the last `window` months of new-customer MRR

Uncertainty: Monte Carlo. NRR and the new-MRR residual are resampled from the last
12 months of history, giving a P10-P90 range.

A forecast is only worth trusting if it beats a naive guess. `backtest` re-runs the
forecast from many past origins and scores it with MAPE against:
    naive   = "next months look like the last one"
    trend   = linear extrapolation of the last 6 closing MRR values
"""
import numpy as np
import pandas as pd


def _nrr_series(wf):
    w = wf[wf["opening"] > 0]
    return ((w["opening"] + w["expansion"] - w["contraction"] - w["churned"]) / w["opening"]).to_numpy()


def _new_trend(new, window):
    x = np.arange(window)
    slope, icpt = np.polyfit(x, new[-window:], 1)
    return slope, icpt


def forecast(wf, horizon=6, window=6, trend_window=12, n_sims=1000, seed=0):
    """Return DataFrame(month, forecast, p10, p90). p10/p90 are NaN if n_sims == 0."""
    nrr_hist = _nrr_series(wf)
    nrr = float(nrr_hist[-window:].mean())
    new = wf["new"].to_numpy(float)
    slope, icpt = _new_trend(new, trend_window)
    last_close = float(wf["closing"].iloc[-1])

    point, c = [], last_close
    for h in range(1, horizon + 1):
        c = c * nrr + max(0.0, slope * (trend_window - 1 + h) + icpt)
        point.append(c)

    out = pd.DataFrame(
        dict(
            month=[wf["month"].iloc[-1] + pd.DateOffset(months=h) for h in range(1, horizon + 1)],
            forecast=point,
        )
    )
    out["p10"], out["p90"] = np.nan, np.nan
    if n_sims:
        rng = np.random.default_rng(seed)
        n12 = nrr_hist[-12:]
        nrr_draws = n12 - n12.mean() + nrr  # same spread, centred on the recent average
        new12 = new[-12:]
        res = new12 - np.polyval(np.polyfit(np.arange(len(new12)), new12, 1), np.arange(len(new12)))
        sims = np.empty((n_sims, horizon))
        for i in range(n_sims):
            c = last_close
            for h in range(1, horizon + 1):
                n_h = max(0.0, slope * (trend_window - 1 + h) + icpt + rng.choice(res))
                c = c * rng.choice(nrr_draws) + n_h
                sims[i, h - 1] = c
        out["p10"] = np.percentile(sims, 10, axis=0)
        out["p90"] = np.percentile(sims, 90, axis=0)
    return out


def backtest(wf, horizon=6, min_history=12, **forecast_kwargs):
    """Rolling-origin evaluation. Returns dict(summary, by_horizon, n_origins)."""
    n = len(wf)
    methods = ("Bottom-up (this model)", "Naive (flat)", "Linear trend")
    errs = {m: [] for m in methods}  # each entry: array of APEs over horizon

    for origin in range(min_history - 1, n - horizon):
        hist = wf.iloc[: origin + 1]
        actual = wf["closing"].iloc[origin + 1 : origin + 1 + horizon].to_numpy(float)
        preds = {
            methods[0]: forecast(hist, horizon, n_sims=0, **forecast_kwargs)["forecast"].to_numpy(),
            methods[1]: np.full(horizon, float(hist["closing"].iloc[-1])),
        }
        b, a = np.polyfit(np.arange(6), hist["closing"].tail(6).to_numpy(float), 1)
        preds[methods[2]] = a + b * (5 + np.arange(1, horizon + 1))
        for m in methods:
            errs[m].append(np.abs(actual - preds[m]) / actual)

    by_h = pd.DataFrame({m: np.mean(errs[m], axis=0) * 100 for m in methods})
    by_h.index = range(1, horizon + 1)
    by_h.index.name = "months_ahead"
    summary = pd.DataFrame(
        dict(method=methods, mape=[float(np.mean(errs[m])) * 100 for m in methods])
    ).sort_values("mape").reset_index(drop=True)
    return dict(summary=summary, by_horizon=by_h, n_origins=len(errs[methods[0]]))
