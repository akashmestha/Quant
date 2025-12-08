# kalman.py
"""
Simple Kalman filter for dynamic linear regression:
Model:
    y_t = alpha_t + beta_t * x_t + eps_t,  eps_t ~ N(0, R)
    state_t = [alpha_t, beta_t]^T
    state_{t} = state_{t-1} + eta_t,   eta_t ~ N(0, Q)  (random walk)
Returns:
    - beta_t series
    - alpha_t series
    - residuals series (y - (alpha + beta*x))
    - optional state covariances (P_t)
Notes:
    - Q (process noise cov) and R (observation noise var) control adaptivity.
    - Larger Q -> faster beta adaptation. Larger R -> trusts model less.
"""

import numpy as np
import pandas as pd

def kalman_filter_regression(x, y, Q_diag=(1e-5, 1e-5), R=1e-2, init_state=None, init_P=None):
    """
    Run Kalman filter for regression y = alpha + beta*x

    Params
    ------
    x, y : 1-D arrays or pandas Series (same length)
    Q_diag : tuple of two floats -> process noise variances for [alpha, beta]
    R : float -> observation noise variance
    init_state : 2-array initial [alpha0, beta0] (defaults to zeros / small beta from OLS)
    init_P : 2x2 initial covariance matrix (defaults to large diag)

    Returns
    -------
    result : dict with
       - 'alpha' : pd.Series indexed as y.index
       - 'beta'  : pd.Series
       - 'residuals' : pd.Series
       - 'P' : list of 2x2 covariances (optional)
    """
    if isinstance(x, pd.Series):
        idx = x.index
        x = x.values
    elif isinstance(y, pd.Series):
        idx = y.index
        x = np.asarray(x)
    else:
        idx = None
        x = np.asarray(x)

    y = np.asarray(y).astype(float)
    n = len(y)
    if len(x) != n:
        raise ValueError("x and y must have same length")

    # Design matrix row for each t: [1, x_t]
    # State vector: [alpha, beta]

    # Initial state (alpha=0, beta from simple OLS slope if available)
    if init_state is None:
        # try robust initial beta estimate
        if np.nanstd(x) > 0:
            beta0 = np.cov(x, y)[0, 1] / (np.var(x) + 1e-12)
        else:
            beta0 = 0.0
        alpha0 = np.nanmean(y) - beta0 * np.nanmean(x)
        state = np.array([alpha0, beta0], dtype=float)
    else:
        state = np.asarray(init_state, dtype=float)

    # Initial covariance
    if init_P is None:
        P = np.diag([1.0, 1.0]) * 1.0  # moderately uncertain
    else:
        P = np.asarray(init_P, dtype=float)

    Q = np.diag(Q_diag)  # process noise cov (2x2)
    R = float(R)

    alphas = np.zeros(n)
    betas = np.zeros(n)
    residuals = np.zeros(n)
    Ps = [None] * n

    for t in range(n):
        xt = x[t]
        H = np.array([1.0, xt]).reshape(1, 2)  # measurement matrix

        # Predict step: state(t|t-1) = state(t-1|t-1) (random walk)
        state_pred = state.copy()
        P_pred = P + Q

        # Measurement prediction
        y_pred = H.dot(state_pred)[0]
        S = H.dot(P_pred).dot(H.T)[0, 0] + R  # innovation covariance (scalar)
        K = (P_pred.dot(H.T) / S).reshape(2,)  # Kalman gain (2x1) -> flattened

        # Update with observation y[t]
        innovation = y[t] - y_pred
        state = state_pred + K * innovation
        P = P_pred - np.outer(K, H).dot(P_pred)

        # Save
        alphas[t] = state[0]
        betas[t] = state[1]
        residuals[t] = innovation
        Ps[t] = P.copy()

    # Build pandas Series with original index if present
    if idx is not None:
        alpha_s = pd.Series(alphas, index=idx)
        beta_s = pd.Series(betas, index=idx)
        res_s = pd.Series(residuals, index=idx)
    else:
        alpha_s = pd.Series(alphas)
        beta_s = pd.Series(betas)
        res_s = pd.Series(residuals)

    return {
        "alpha": alpha_s,
        "beta": beta_s,
        "residuals": res_s,
        "P": Ps,
        "Q": Q,
        "R": R,
    }
