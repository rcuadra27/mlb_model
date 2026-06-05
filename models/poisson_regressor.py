"""Poisson regression for pitcher strikeout count models."""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize


class PoissonRegressor:
    """
    Poisson regression via scipy MLE.
    Equivalent to sklearn's PoissonRegressor but gives us full control
    over the lambda computation and PMF output.
    """

    def __init__(self, alpha=0.1):
        self.alpha = alpha  # L2 regularization
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        n, p = X.shape
        w0 = np.zeros(p + 1)  # [intercept, coef...]

        def neg_log_likelihood(w):
            intercept, coef = w[0], w[1:]
            eta = intercept + X @ coef
            lam = np.exp(np.clip(eta, -10, 10))
            nll = -(y * np.log(lam + 1e-10) - lam).sum()
            reg = 0.5 * self.alpha * (coef ** 2).sum()
            return nll + reg

        def grad(w):
            intercept, coef = w[0], w[1:]
            eta = intercept + X @ coef
            lam = np.exp(np.clip(eta, -10, 10))
            resid = lam - y
            g_intercept = resid.sum()
            g_coef = X.T @ resid + self.alpha * coef
            return np.concatenate([[g_intercept], g_coef])

        result = minimize(neg_log_likelihood, w0, jac=grad,
                          method='L-BFGS-B',
                          options={'maxiter': 1000, 'ftol': 1e-9})

        self.intercept_ = result.x[0]
        self.coef_ = result.x[1:]
        return self

    def predict_lambda(self, X):
        """Predict expected K count (lambda)."""
        eta = self.intercept_ + X @ self.coef_
        return np.exp(np.clip(eta, -10, 10))

    def predict_proba_k(self, X, k_values=None):
        """
        Return P(K=k) for each k in k_values, for each row.
        Returns DataFrame with columns p_k0, p_k1, ..., p_k10plus
        """
        if k_values is None:
            k_values = list(range(11))  # 0..10
        lam = self.predict_lambda(X)
        probs = {}
        for k in k_values:
            probs[f'p_k{k}'] = stats.poisson.pmf(k, lam)
        probs['p_k10plus'] = 1.0 - stats.poisson.cdf(9, lam)
        return pd.DataFrame(probs)

    def predict_over_k(self, X, thresholds=None):
        """
        Return P(K > threshold) for each threshold.
        e.g., threshold=4.5 → P(K >= 5) for DFS over/under lines.
        """
        if thresholds is None:
            thresholds = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]
        lam = self.predict_lambda(X)
        result = {}
        for t in thresholds:
            k_floor = int(t + 0.5)  # e.g., 4.5 → 5
            result[f'p_over_{str(t).replace(".", "_")}'] = (
                1.0 - stats.poisson.cdf(k_floor - 1, lam))
        return pd.DataFrame(result)
