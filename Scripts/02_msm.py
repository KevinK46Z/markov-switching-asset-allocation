"""
02_msm.py
=========
The estimator of the thesis: a two-state Markov-switching model whose
transition probabilities depend on observable covariates (Chapter 4).

This is a module, not a script.  It defines the likelihood, the Hamilton
filter, the multi-start estimator and the conditional moments the allocation
is built on.  03, 04, 05 and 07 load it, and 07c and 07d reach it through
07.  Running the file directly executes a self-test, at the bottom of
the file: it simulates data from known parameters, estimates the model on
them and checks that the estimates come back close to the true values.

Model
-----
On row t of the panel, the observation vector is
y_t = (r_obs_t, rb_x_t, re_x_t)', and r_t is the panel's `r` column, the bill
rate quoted at the close of month t-1:
    r_obs_t = mu_r(i) + rho(i) * r_t + eps1_t
    rb_x_t  = mu_b                   + eps2_t
    re_x_t  = mu_e                   + eps3_t      eps_t ~ N(0, Sigma(i))
Ang-Bekaert Model II: the regime enters through Sigma only, mu_b and mu_e are
common to both regimes.

Array conventions
-----------------
Z[t] is the covariate vector driving the transition into month t.  For M1
that is the short rate quoted at the close of month t-1, which is the
panel's `r` column, so Z[:, 0] and the autoregressive lag are the same array.
prepare() checks it.  In the code this lag is the array `rlag`.

Parameter vector
----------------
The optimiser works with one unconstrained vector theta of length 20 + 2k,
where k is the number of transition covariates:
    [0:2]       mu_r(1), mu_r(2)       short-rate intercepts
    [2:4]       rho(1), rho(2)         short-rate autoregressive coefficients
    [4:6]       mu_b, mu_e             mean excess returns, common to both
    [6:12]      Sigma(1)               covariance of regime 1, packed
    [12:18]     Sigma(2)               covariance of regime 2, packed
    [18:20]     a1, a2                 transition intercepts
    [20:20+k]   b1                     slopes of the equation for P
    [20+k:]     b2                     slopes of the equation for Q
The first 20 entries have the same layout for every k.  unpack() turns theta
into named parameters.

Each Sigma is stored as the six free elements of its lower-triangular
Cholesky factor L, row by row, with the three diagonal elements in logs.
Sigma = L L' is then positive definite for any real vector, so the optimiser
needs no constraints.  With y ordered (short rate, bond, equity), the six
elements correspond to R_rr, R_br, R_bb, R_er, R_eb and R_ee in Table 6 of
Ang and Bekaert (2002), under Lambda = R R' with R lower triangular
(Appendix A.2).
"""

import warnings
from math import exp, log
import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2

LOG2PI  = np.log(2.0 * np.pi)     # constant term of the normal log density
N_FIXED = 20                      # length of theta before the b-slopes; 
                                  # theta[:20] is the same layout for every k
                                  
# ==========================================================================
# Packing and unpacking
# ==========================================================================
def _chol_from_packed(v):
    """
    Rebuild the 3x3 Cholesky factor of a covariance matrix from the six
    numbers the optimiser searches over.

    Parameters
    ----------
    v : array_like, length 6
        (log l11, l21, log l22, l31, l32, log l33): the six free elements of
        the lower-triangular factor, row by row, with the three diagonal
        elements in logs.

    Returns
    -------
    np.ndarray
        The 3x3 lower-triangular factor L, with Sigma = L L'.

    Notes
    -----
    The diagonal elements are recovered by exponentiating, so they are
    strictly positive whatever the optimiser proposes.  Sigma = L L' is then
    positive definite for any real v, which lets the optimiser search
    without constraints (Section 4.4).
    """
    L = np.zeros((3, 3))
    L[0, 0] = np.exp(v[0])        # diagonal: exp, always > 0
    L[1, 0] = v[1]                # off-diagonal: free
    L[1, 1] = np.exp(v[2])
    L[2, 0] = v[3]
    L[2, 1] = v[4]
    L[2, 2] = np.exp(v[5])
    return L


def _packed_from_cov(S):
    """
    Turn a covariance matrix into the six numbers the optimiser searches
    over.

    Parameters
    ----------
    S : array_like
        3x3 positive-definite covariance matrix.

    Returns
    -------
    np.ndarray
        The six free elements of the Cholesky factor of S, row by row, with
        the three diagonal elements in logs: the layout _chol_from_packed()
        expects.

    Notes
    -----
    The reverse of _chol_from_packed() followed by Sigma = L L'.  Used by
    starting_values() to turn a sample covariance into a block of the
    starting parameter vector.  Raises an error if S is not positive
    definite, since it then has no Cholesky factor.
    """
    L = np.linalg.cholesky(S)                              # factorise
    return np.array([np.log(L[0, 0]), L[1, 0], np.log(L[1, 1]),
                     L[2, 0], L[2, 1], np.log(L[2, 2])])   # diagonals -> logs


def packed_from_chol_elements(R_rr, R_br, R_bb, R_er, R_eb, R_ee):
    """
    Turn the six Cholesky elements Ang and Bekaert publish into the six
    numbers the optimiser searches over.

    Parameters
    ----------
    R_rr, R_br, R_bb, R_er, R_eb, R_ee : float
        The elements of the lower-triangular factor as labelled in Table 6 of
        the working paper (2002), with the observation vector ordered 
        (short rate, bond, equity).  R_br, for example, is row bond, column 
        short rate.

    Returns
    -------
    np.ndarray
        The same six elements, row by row, with the three diagonal elements
        in logs: the layout _chol_from_packed() expects.

    Notes
    -----
    Used to build AB_MODEL_II_THETA, their published parameter vector, so it
    can be fed through the same filter and allocation as the fitted one.
    Unlike _packed_from_cov(), it starts from the factor itself rather than
    from a covariance matrix, because the factor is what Table 6 reports.
    """
    return np.array([np.log(R_rr), R_br, np.log(R_bb),
                     R_er, R_eb, np.log(R_ee)])            # diagonals -> logs


def unpack(theta, k):
    """
    Split the parameter vector the optimiser works with into named,
    readable parameters.

    Parameters
    ----------
    theta : array_like, length 20 + 2k
        Unconstrained parameter vector, layout in the module docstring.
    k : int
        Number of transition covariates, 0 for M0 up to 4 for M4.

    Returns
    -------
    dict
        mu_r, rho   intercept and autoregressive coefficient of the short-rate
                    equation, one entry per regime
        mu_b, mu_e  mean excess return on bond and equity, common to both
                    regimes
        L           the two Cholesky factors, rebuilt from their packed form
        Sigma       the two covariance matrices, Sigma = L L'
        a           the two transition intercepts, a1 and a2
        b           the two slope vectors, b1 and b2, each of length k
        k           the number of covariates, passed through so that later
                    functions need only the dictionary

    Notes
    -----
    Every entry is a copy or a new array, so changing an entry cannot change
    theta by accident.
    """
    theta = np.asarray(theta, dtype=float)
    # rebuild both Cholesky factors from their packed blocks
    L1, L2 = _chol_from_packed(theta[6:12]), _chol_from_packed(theta[12:18])
    return {
        "mu_r":  theta[0:2].copy(),           # short-rate intercept, per regime
        "rho":   theta[2:4].copy(),           # AR coefficient, per regime
        "mu_b":  theta[4],                    # bond mean, common
        "mu_e":  theta[5],                    # equity mean, common
        "L":     [L1, L2],                    # factors, used by _log_densities
        "Sigma": [L1 @ L1.T, L2 @ L2.T],      # level form, used for reporting
        "a":     theta[18:20].copy(),         # transition intercepts
        "b":     [theta[N_FIXED:N_FIXED + k].copy(),          # b1
                  theta[N_FIXED + k:N_FIXED + 2 * k].copy()], # b2
        "k":     k,
    }


def relabel(theta, k):
    """
    Order the two regimes so that regime 2 is always the more volatile one.

    The likelihood is the same whichever regime is called 1 and which 2, so
    the optimiser may return either ordering.  This function fixes the
    labels by one rule, so estimates are comparable across specifications
    and across the monthly fits of the backtest (Section 4.4).

    Parameters
    ----------
    theta : array_like, length 20 + 2k
        Estimated parameter vector, layout in the module docstring.
    k : int
        Number of transition covariates.

    Returns
    -------
    np.ndarray
        theta unchanged if the variance of the equity shock is already larger
        in regime 2.  Otherwise a copy with every regime-specific block
        swapped: mu_r, rho, the two covariance matrices, a1 with a2 and b1
        with b2.  mu_b and mu_e are common to both regimes and stay as they
        are.
    """
    theta = np.asarray(theta, dtype=float)
    p = unpack(theta, k)
    # equity variance in regime 1 <= regime 2: labels already correct
    if p["Sigma"][0][2, 2] <= p["Sigma"][1][2, 2]:
        return theta.copy()
    # otherwise regime 1 is the volatile one: swap every regime-specific block                               
    out = theta.copy()
    out[0:2]   = theta[[1, 0]]                            # mu_r
    out[2:4]   = theta[[3, 2]]                            # rho
    out[6:12]  = theta[12:18]                             # Sigma blocks
    out[12:18] = theta[6:12]
    out[18:20] = theta[[19, 18]]                          # a1 <-> a2
    if k:                                                 # b1 <-> b2
        out[N_FIXED:N_FIXED + k]         = theta[N_FIXED + k:N_FIXED + 2 * k]
        out[N_FIXED + k:N_FIXED + 2 * k] = theta[N_FIXED:N_FIXED + k]
    return out                                            # mu_b, mu_e untouched


# ==========================================================================
# Transition probabilities
# ==========================================================================
def _logistic(x):
    """
    Logistic function, which maps the linear index of a transition equation
    into a probability between 0 and 1 (Equation 4.7).

    Parameters
    ----------
    x : array_like or float
        Linear index a + b'z.

    Returns
    -------
    np.ndarray or float
        1 / (1 + exp(-x)).

    Notes
    -----
    The index is clipped to [-35, 35] first.  Beyond that range the logistic
    is within 1e-15 of 0 or 1, so the clip changes no result.  It only stops
    np.exp from overflowing when the optimiser tries extreme parameters.
    """
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35.0, 35.0)))


def transition_probs(params, Z):
    """
    Probabilities of staying in each regime, for every month of the sample.

    Evaluates the two logistic transition equations, P_t for regime 1 and
    Q_t for regime 2, at the covariates of every month (Equation 4.7).

    Parameters
    ----------
    params : dict
        Output of unpack().
    Z : np.ndarray, shape (T, k)
        Z[t] is the covariate vector driving the transition into month t.
        Shape (T, 0) for M0, which has no covariates.

    Returns
    -------
    P : np.ndarray, length T
        Probability of staying in regime 1 from month t-1 to month t.
    Q : np.ndarray, length T
        Probability of staying in regime 2 from month t-1 to month t.

    Notes
    -----
    Both are kept inside [1e-8, 1 - 1e-8].  A stay-probability of exactly one
    would make a regime absorbing, so any later switch out of it would have
    probability zero and the log-likelihood would be minus infinity.
    """
    # linear index of each transition equation, a + b'z_t
    idx1 = params["a"][0] + (Z @ params["b"][0] if params["k"] else 0.0)
    idx2 = params["a"][1] + (Z @ params["b"][1] if params["k"] else 0.0)
    T = Z.shape[0]
    # squash to probabilities
    # M0 has no covariates, so its index is one number for all months.
    # broadcast_to repeats it to length T. for k > 0 it changes nothing.
    # copy() turns the read-only view broadcast_to returns into a normal array.
    P = np.broadcast_to(_logistic(idx1), (T,)).copy()   # k = 0: scalar -> length T
    Q = np.broadcast_to(_logistic(idx2), (T,)).copy()
    eps = 1e-8                                          # keep off 0 and 1
    return np.clip(P, eps, 1 - eps), np.clip(Q, eps, 1 - eps)


def stay_probs_at(params, z):
    """
    Probabilities of staying in each regime for one month only, at a given
    covariate vector.

    The same calculation as transition_probs(), for a different use.
    transition_probs() serves the estimation: the filter needs P and Q for
    every month of the estimation window, from the window's own covariates.
    This function serves the allocation: at the decision date only the
    transition into the month about to be held matters.  Its covariate
    vector, observed at the close of the decision month, lies outside the
    window's covariate matrix, so it is passed in separately and two numbers
    come back.

    Parameters
    ----------
    params : dict
        Output of unpack().
    z : array_like, length k
        Covariate vector observed at the close of month t, the decision
        month.  Empty array for M0.

    Returns
    -------
    P : float
        Probability of staying in regime 1 from month t to month t+1.
    Q : float
        Probability of staying in regime 2 from month t to month t+1.

    Notes
    -----
    Used by conditional_moments() and predict_regime().  Both values are kept
    inside [1e-8, 1 - 1e-8], as in transition_probs().
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    # linear index of each transition equation, a + b'z
    idx1 = params["a"][0] + (z @ params["b"][0] if params["k"] else 0.0)
    idx2 = params["a"][1] + (z @ params["b"][1] if params["k"] else 0.0)
    eps = 1e-8                                          # keep off 0 and 1
    return (float(np.clip(_logistic(idx1), eps, 1 - eps)),
            float(np.clip(_logistic(idx2), eps, 1 - eps)))


def ergodic_prob(P1, Q1):
    """
    Long-run probability of regime 1 when the two stay-probabilities are held
    fixed (Appendix A.4).

    Parameters
    ----------
    P1, Q1 : float
        Probabilities of staying in regime 1 and in regime 2.

    Returns
    -------
    float
        (1 - Q1) / ((1 - P1) + (1 - Q1)), the share of time the chain spends
        in regime 1 in the long run.  It solves
        xi = P1 * xi + (1 - Q1) * (1 - xi).

    Notes
    -----
    Used by stationary_prob() for M0 and by 03 for the plug-in value at the
    mean short rate.  hamilton_filter() starts its recursion from the same
    formula at the first month's transition probabilities (Equation 4.17), but 
    writes it out inline rather than calling this function.
    """
    return (1.0 - Q1) / ((1.0 - P1) + (1.0 - Q1))


def stationary_prob(params, n=200_000, burn=2_000, seed=0):
    """
    Long-run probability of regime 1, by simulating the model forward.  This
    is the counterpart of the 0.7014 Ang and Bekaert report for Model II
    (Table 6.1, row "Pr(calm), simulated").  A diagnostic only: no estimate,
    filter or allocation uses it.

    With the short rate as the only covariate, the transition probabilities
    move with the rate, and the rate itself switches with the regime.  The
    long-run share of regime 1 therefore has no closed form.  The function
    simulates regime and short rate together for many months and counts the
    share spent in regime 1.

    Parameters
    ----------
    params : dict
        Output of unpack().
    n : int
        Simulated months that are counted.
    burn : int
        Months simulated first and discarded, so the result does not depend
        on the starting point.
    seed : int
        Seed of the random generator, so the result is reproducible.

    Returns
    -------
    float
        Share of the n counted months spent in regime 1.

    Notes
    -----
    M0 (k = 0) needs no simulation: the transition probabilities are
    constant, so ergodic_prob() gives the answer directly.  For k > 1 the
    function returns NaN, because the extra covariates have no model of
    their own and cannot be simulated forward.  Only M1 (k = 1) is
    simulated.
    """
    k = params["k"]
    if k == 0:
        # time-homogeneous chain, no simulation needed
        return ergodic_prob(_logistic(params["a"][0]), _logistic(params["a"][1]))
    if k != 1:
        return np.nan

    rng = np.random.default_rng(seed)
    mu_r, rho = params["mu_r"], params["rho"]
    sd = [float(np.sqrt(params["Sigma"][i][0, 0])) for i in (0, 1)]   # short-rate sd
    a = params["a"]
    b = [float(params["b"][0][0]), float(params["b"][1][0])]

    s = 0                                        # start in regime 1
    r = mu_r[0] / max(1.0 - rho[0], 1e-3)        # at its long-run level, unit-root guard
    n1 = 0
    for t in range(n + burn):
        # transition into this month is driven by last month's rate (Equation 4.7)
        stay = _logistic(a[0] + b[0] * r) if s == 0 else _logistic(a[1] + b[1] * r)
        s = s if rng.random() < stay else 1 - s                     # draw the regime
        r = mu_r[s] + rho[s] * r + sd[s] * rng.standard_normal()    # AR(1) step
        if t >= burn:
            n1 += (s == 0)                       # count months in regime 1
    return n1 / n


# ==========================================================================
# Conditional moments for the allocation
# ==========================================================================
def conditional_moments(params, z_now, regime=None, xi=None):
    """
    Mean and covariance of next month's two risky excess returns.

    Parameters
    ----------
    params : dict
        Output of unpack().
    z_now : array_like, length k
        Covariate vector observed at the decision date, the close of month t.
        Empty array for k = 0.
    regime : int, optional
        Hard rule (Ang and Bekaert).  The regime the current month has been
        classified as: 0 for regime 1 (calm), 1 for regime 2 (volatile).
        The mixing weight is then P for 0 and 1 - Q for 1.
    xi : float, optional
        Soft rule.  The filtered probability of regime 1 at the decision
        date, used as it is: the mixing weight is P * xi + (1 - Q) * (1 - xi).
        The hard rule is this with xi rounded to 0 or 1 first.

    Pass exactly one of regime and xi.  If both are given, xi is used.

    Returns
    -------
    mu : np.ndarray, length 2
        Expected excess return on (bond, equity), percent per month.
    Sigma : np.ndarray, shape (2, 2)
        Covariance, percent-squared per month.
    pi1 : float
        Probability assigned to regime one next month. Under the hard rule
        this is P(s_{t+1} = 1 | s_t = regime); under the soft rule it is the
        one-step-ahead probability given all information at t.
    P, Q : float
        Stay-probabilities at z_now.

    Raises
    ------
    ValueError
        Neither regime nor xi was given.

    Notes
    -----
    Equation 4.27 in the thesis. Sigma is the mixture of the two regime 
    covariances, weighted by pi1, not the covariance of one regime.  The jump 
    term from the difference of the regime means is computed but is exactly 
    zero, since mu_b and mu_e are common to both regimes.
    """
    P, Q = stay_probs_at(params, z_now)
    if xi is None:
        if regime is None:
            raise ValueError("pass either regime (hard) or xi (soft)")
        pi1 = P if regime == 0 else 1.0 - Q        # hard, Ang-Bekaert
    else:
        # soft: law of total probability over today's two regimes
        pi1 = P * xi + (1.0 - Q) * (1.0 - xi)

    # regime-conditional means of (bond, equity), common across regimes here
    # but written out so the code stays correct if the restriction is relaxed
    mu1 = np.array([params["mu_b"], params["mu_e"]])
    mu2 = np.array([params["mu_b"], params["mu_e"]])

    # bond/equity block of each covariance matrix
    Y1 = params["Sigma"][0][1:, 1:]
    Y2 = params["Sigma"][1][1:, 1:]

    mu = pi1 * mu1 + (1.0 - pi1) * mu2
    d = (mu1 - mu2).reshape(-1, 1)                    # jump term, zero here
    Sigma = pi1 * Y1 + (1.0 - pi1) * Y2 + pi1 * (1.0 - pi1) * (d @ d.T)
    return mu, Sigma, pi1, P, Q


# ==========================================================================
# Log densities
# ==========================================================================
def _log_densities(params, y, rlag):
    """
    Log density of each month's observation under each of the two regimes.

    For every month t and regime i this is log f(y_t | s_t = i): how likely
    the observed short rate, bond and equity returns of month t are if the
    month belongs to regime i.  The filter weighs the two regimes by these
    values (Appendix A.5).

    Parameters
    ----------
    params : dict
        Output of unpack().
    y : np.ndarray, shape (T, 3)
        Observation vectors (r_obs, rb_x, re_x), one row per month.
    rlag : np.ndarray, length T
        The panel's `r`: the bill rate quoted at the close of month t-1,
        which is the autoregressive lag of the short-rate equation.

    Returns
    -------
    np.ndarray, shape (T, 2)
        Column 0 holds log f(y_t | regime 1), column 1 holds
        log f(y_t | regime 2), one row per month.

    Notes
    -----
    The density is trivariate normal.  Its quadratic form is computed through
    the Cholesky factor L rather than by inverting Sigma, which is cheaper and
    numerically more stable.
    """
    T = y.shape[0]
    out = np.empty((T, 2))
    for i in (0, 1):
        # residual under regime i, only the first column is regime-dependent
        resid = y.copy()
        resid[:, 0] -= params["mu_r"][i] + params["rho"][i] * rlag
        resid[:, 1] -= params["mu_b"]
        resid[:, 2] -= params["mu_e"]
        L = params["L"][i]
        u = np.linalg.inv(L) @ resid.T             # (3, T), u'u = resid' Sigma^-1 resid
        quad = np.einsum("jt,jt->t", u, u)         # one quadratic form per date
        logdet = 2.0 * np.log(np.diag(L)).sum()    # log|Sigma| from the factor
        out[:, i] = -0.5 * (3.0 * LOG2PI + logdet + quad)
    return out


# ==========================================================================
# The filter
# ==========================================================================
def hamilton_filter(theta, y, rlag, Z, k):
    """
    Run the Hamilton filter over the estimation window at one parameter
    vector (Section 4.4).

    Goes through the window month by month.  Each month it first predicts
    the probability of regime 1 from last month's assessment and the
    transition probabilities, then updates that prediction with the month's
    observation by Bayes' rule.  The density of each observation under the
    prediction, D_t, is the month's contribution to the likelihood.

    Parameters
    ----------
    theta : array_like, length 20 + 2k
        Parameter vector, layout in the module docstring.
    y : np.ndarray, shape (T, 3)
        Observation vectors (r_obs, rb_x, re_x), one row per month.
    rlag : np.ndarray, length T
        The panel's `r`, the autoregressive lag of the short-rate equation.
    Z : np.ndarray, shape (T, k)
        Covariates driving the transition into each month.
    k : int
        Number of covariates.

    Returns
    -------
    loglik : float
Log-likelihood of the window, the sum of log D_t (Equation 4.23).
        Minus infinity if D_t underflows to zero in some
        month, which marks the parameter vector as unusable.
    xi_pred : np.ndarray, length T
        Predicted probability of regime 1 in month t, given the data through
        month t-1 (equation 4.19).
    xi_filt : np.ndarray, length T
        Filtered probability of regime 1 in month t, given the data through
        month t (equation 4.21).

    Notes
    -----
    Regime 1 is the calm state under the labelling of relabel().  xi_filt is
    what the allocation classifies on.  The recursion starts from the
    long-run probability of the first month's transition matrix (Equation 4.17)
    """
    params = unpack(theta, k)
    P, Q   = transition_probs(params, Z)        # transitions into each month
    logf   = _log_densities(params, y, rlag)    # densities under both regimes
    T = y.shape[0]

    # the recursion is sequential and cannot be vectorised, so it runs on plain
    # Python floats, converting once up front avoids numpy dispatch cost and
    # makes the likelihood about an order of magnitude faster
    Pl, Ql = P.tolist(), Q.tolist()
    f1l, f2l = logf[:, 0].tolist(), logf[:, 1].tolist()

    xi_pred = [0.0] * T
    xi_filt = [0.0] * T
    xi_prev = (1.0 - Ql[0]) / ((1.0 - Pl[0]) + (1.0 - Ql[0]))   # ergodic start
    loglik = 0.0

    for t in range(T):
        # PREDICT: regime 1 this month either by staying (P_t) or by switching
        # up from regime 2 (1 - Q_t)
        xp = Pl[t] * xi_prev + (1.0 - Ql[t]) * (1.0 - xi_prev)
        xi_pred[t] = xp

        # UPDATE: Bayes' rule once y_t arrives.  Both densities are rescaled by
        # the larger of the two before exponentiating.  The offset cancels in
        # the ratio and is added back to the log-likelihood, so this is exact
        # and only prevents overflow when one regime fits far better.
        f1, f2 = f1l[t], f2l[t]
        if f1 >= f2:
            c, d1, d2 = f1, 1.0, exp(f2 - f1)
        else:
            c, d1, d2 = f2, exp(f1 - f2), 1.0
        dens = d1 * xp + d2 * (1.0 - xp)          # D_t, rescaled
        if dens <= 0.0:                           # numerically dead parameter point
            return -np.inf, np.asarray(xi_pred), np.asarray(xi_filt)
        loglik += log(dens) + c                   # exact contribution of month t
        xi_prev = d1 * xp / dens                  # posterior, carried forward
        xi_filt[t] = xi_prev

    return loglik, np.asarray(xi_pred), np.asarray(xi_filt)


def neg_loglik(theta, y, rlag, Z, k):
    """
    The function the optimiser minimises: minus the log-likelihood of the
    estimation window at one parameter vector.

    The optimisers minimise, so maximising the likelihood means minimising
    its negative.  This wraps hamilton_filter() and returns that negative.

    Parameters
    ----------
    theta, y, rlag, Z, k
        As in hamilton_filter().

    Returns
    -------
    float
        Minus the log-likelihood.  At a parameter vector where the filter
        fails or returns no finite value, the fixed large number 1e10
        instead.

    Notes
    -----
    Returning 1e10 rather than an error or NaN lets the optimiser treat such
    a point as very bad and move away from it, instead of stopping.
    standard_errors() treats any value at or above 1e9 as such a failure, so
    the placeholder never enters a numerical derivative.
    """
    try:
        ll, _, _ = hamilton_filter(theta, y, rlag, Z, k)
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        return 1e10
    return 1e10 if not np.isfinite(ll) else -ll


# ==========================================================================
# Data preparation
# ==========================================================================
def prepare(df, zcols, strict=True):
    """
    Turn a panel window into the arrays the filter works on, and check it.

    Parameters
    ----------
    df : pd.DataFrame
        Estimation window from a panel of 01_build_data.py, one row per
        month.  Must carry r_obs, r, rb_x and re_x.
    zcols : list of str
        Transition covariate columns, empty for M0.  Their order fixes the
        order of the slopes in b1 and b2 and of the rows in the coefficient
        tables.
    strict : bool
        If True, also check that z_short equals the panel's `r`.  Pass False
        when the covariates have been standardised, since z_short then no
        longer equals `r`.

    Returns
    -------
    y : np.ndarray, shape (T, 3)
        Observation vectors (r_obs, rb_x, re_x), one row per month.
    rlag : np.ndarray, length T
        The panel's `r`, the bill rate quoted at the close of month t-1 and
        earned over month t.  The autoregressive lag of the short-rate
        equation.
    Z : np.ndarray, shape (T, k)
        The covariate columns.  Shape (T, 0) for M0.
    index : pd.Index
        The months of the window, passed through unchanged.

    Raises
    ------
    KeyError
        A required column is missing.
    ValueError
        The panel violates r_t == r_obs_{t-1}, or a row of the window
        contains a missing or infinite value.

    Notes
    -----
    No row is ever dropped.  A row that fails a check stops the run instead,
    so every month of the window enters the likelihood.
    """
    missing = [c for c in ("r_obs", "r", "rb_x", "re_x") if c not in df.columns]
    if missing:
        raise KeyError(f"panel is missing {missing}; rebuild it with the "
                       f"current 01_build_data.py")

    # extract the three observation columns, the AR lag, and the covariates
    y    = df[["r_obs", "rb_x", "re_x"]].to_numpy(dtype=float)
    rlag = df["r"].to_numpy(dtype=float)
    Z    = (df[list(zcols)].to_numpy(dtype=float) if zcols
            else np.zeros((len(df), 0)))          # (T, 0) placeholder for M0

    # the panel's defining invariant: r_t is r_obs_{t-1}
    if not np.allclose(rlag[1:], y[:-1, 0], atol=1e-10, equal_nan=False):
        raise ValueError("panel violates r_t == r_obs_{t-1}: the two datings "
                         "have come apart.  Re-run 01_build_data.py.")

    # for M1 the transition driver is the AR lag, so catch a stale panel here
    # rather than in the coefficient table
    if strict and "z_short" in zcols:
        j = list(zcols).index("z_short")
        if not np.allclose(Z[:, j], rlag, atol=1e-10):
            warnings.warn("z_short does not equal the AR lag.  Either the panel "
                          "is stale (old 01 lagged z_short a second time) or Z "
                          "was standardised before prepare(); pass strict=False "
                          "in the latter case.", RuntimeWarning)

    # no NaNs anywhere in the window
    bad = ~(np.isfinite(rlag) & np.isfinite(y).all(axis=1))
    if Z.shape[1]:
        bad |= ~np.isfinite(Z).all(axis=1)
    if bad.any():
        raise ValueError(f"{int(bad.sum())} non-finite row(s) in the estimation "
                         f"window, first at {df.index[bad][0]}")
    return y, rlag, Z, df.index


def standardise(Z, ref=None):
    """
    Rescale each covariate to zero mean and unit variance (Section 4.2).

    Parameters
    ----------
    Z : np.ndarray, shape (T, k)
        Covariates to transform.
    ref : np.ndarray, optional
        Array the mean and standard deviation are computed from.  Defaults to
        Z itself.  The scripts always use the default: in the full-sample fit
        Z is the whole panel, in the backtest it is the estimation window, so
        nothing dated after the decision enters the transform.

    Returns
    -------
    Zs : np.ndarray, shape (T, k)
        The standardised covariates.
    mu, sd : np.ndarray, length k
        Mean and standard deviation of each covariate over ref, one entry per
        column.  Not the model parameters mu_r, mu_b, mu_e.  07 applies them to
        the covariate vector of the decision date, so it is scaled exactly like
        the estimation window.

    Notes
    -----
    Not applied in the replication, where the slopes of M1 must stay in the
    raw units of Table 6.  A column with no variation is left unscaled
    rather than divided by zero.
    """
    if Z.shape[1] == 0:
        return Z, np.zeros(0), np.ones(0)         # nothing to scale for M0
    ref = Z if ref is None else ref
    mu, sd = ref.mean(axis=0), ref.std(axis=0, ddof=1)
    sd = np.where(sd > 0, sd, 1.0)                # constant column: leave alone
    return (Z - mu) / sd, mu, sd


# ==========================================================================
# Starting values
# ==========================================================================
def starting_values(y, rlag, Z, k, n_random=0, rng=None):
    """
    Build the parameter vectors the optimiser starts its searches from
    (Section 4.4, "Where the first candidate comes from").

    Parameters
    ----------
    y, rlag, Z : np.ndarray
        Output of prepare().  Z is not used, the number of covariates is
        taken from k.
    k : int
        Number of transition covariates.
    n_random : int
        Number of randomly shifted copies to add to the data-driven start.
    rng : np.random.Generator, optional
        Random generator for the shifts.  Defaults to a fixed seed, so the
        same starts come out on every run.

    Returns
    -------
    list of np.ndarray
        Each of length 20 + 2k.  The first is the data-driven start, the
        others are randomly shifted copies of it.

    Notes
    -----
    The data-driven start is built as follows:
        short-rate block   one least-squares AR(1) fit, copied into both
                           regimes and moved slightly apart
        risky means        the sample means of the window
        covariances        months are split at the median of the absolute
                           deviation of the equity return from its mean;
                           the quiet half gives one matrix, the loud half
                           the other
        intercepts         a1 = a2 = log(0.95/0.05) = 2.94, a stay-
                           probability of 0.95
        slopes             all zero, so the search starts from the
                           constant-transition model
    """
    rng = np.random.default_rng(0) if rng is None else rng

    # OLS on the short-rate equation, r_obs on a constant and the AR lag
    X = np.column_stack([np.ones_like(rlag), rlag])
    coef, *_ = np.linalg.lstsq(X, y[:, 0], rcond=None)

    # crude volatility split: months whose equity shock exceeds the median
    loud = np.abs(y[:, 2] - y[:, 2].mean()) > np.median(np.abs(y[:, 2] - y[:, 2].mean()))
    resid = np.column_stack([y[:, 0] - X @ coef, y[:, 1] - y[:, 1].mean(),
                             y[:, 2] - y[:, 2].mean()])
    S_calm = np.cov(resid[~loud].T)               # covariance of the quiet half
    S_loud = np.cov(resid[loud].T)                # covariance of the loud half

    # add 1e-8 to each variance (np.eye(3) is the identity matrix): on a short
    # window a half can give a nearly singular matrix, which the Cholesky
    # factorisation would reject
    S_calm += 1e-8 * np.eye(3)
    S_loud += 1e-8 * np.eye(3)

    # standard deviation of the AR(1) residual, used to set how far apart the
    # two starting values of mu_r are placed
    sd_r = float(np.std(resid[:, 0]))

    base = np.concatenate([
        [coef[0] - 0.5 * sd_r, coef[0] + 0.5 * sd_r],   # mu_r, nudged apart
        [coef[1], coef[1] * 0.95],               # rho,  nudged apart
        [y[:, 1].mean(), y[:, 2].mean()],        # mu_b, mu_e
        _packed_from_cov(S_calm),                # Sigma(1)
        _packed_from_cov(S_loud),                # Sigma(2)
        [np.log(0.95 / 0.05)] * 2,               # a1, a2: persistent regimes
        np.zeros(2 * k),                         # b1, b2: start at M0
    ])

    starts = [base]
    for _ in range(n_random):                 # randomly shifted copies of base
        jit = base.copy()
        # means and AR coefficients: noise relative to each value's size
        jit[0:6]   += rng.normal(0, 0.25, 6) * (np.abs(base[0:6]) + 0.1)
        # covariance blocks: absolute noise; on the log diagonals this is
        # roughly a +/-20 percent change in the standard deviations
        jit[6:18]  += rng.normal(0, 0.20, 12)
        jit[18:20] += rng.normal(0, 0.60, 2)     # transition intercepts
        if k:
            jit[N_FIXED:] += rng.normal(0, 0.30, 2 * k)   # slopes
        starts.append(jit)
    return starts


def nest_start(theta_lower, k_lower, k_upper):
    """
    Turn the estimate of a lower rung into a starting point for the rung
    above, with the slopes on the newly added covariates set to zero
    (Section 4.4).

    Parameters
    ----------
    theta_lower : array_like, length 20 + 2*k_lower
        Converged estimate of the lower rung.
    k_lower, k_upper : int
        Number of covariates of the lower and the upper rung, with
        k_upper > k_lower.

    Returns
    -------
    np.ndarray, length 20 + 2*k_upper
        The first 20 entries copied unchanged.  The old slopes of b1 and b2
        copied into the first k_lower places of the new b1 and b2.  The
        slopes on the new covariates set to zero.

    Notes
    -----
    At this point the upper rung is numerically identical to the lower one,
    so its search starts at a likelihood the lower rung already reached.
    Since the best start is kept, the upper rung cannot end with a lower
    likelihood.  If it does, the optimiser failed.
    """
    lo = np.asarray(theta_lower, dtype=float)
    new = np.zeros(N_FIXED + 2 * k_upper)                            # new slopes at 0
    new[:N_FIXED] = lo[:N_FIXED]                                     # shared block
    new[N_FIXED:N_FIXED + k_lower] = lo[N_FIXED:N_FIXED + k_lower]   # old b1
    new[N_FIXED + k_upper:N_FIXED + k_upper + k_lower] = \
        lo[N_FIXED + k_lower:N_FIXED + 2 * k_lower]                  # old b2
    return new


# ==========================================================================
# Estimation
# ==========================================================================
class MSResult:
    """
    Everything a fit produces, kept together in one object.

    fit() and evaluate() return one.  The scripts read the estimate, the
    likelihood and the regime probabilities from it, and pass `params` on to
    the allocation.

    Parameters
    ----------
    theta : np.ndarray, length 20 + 2k
        Estimated parameter vector, already passed through relabel().
    loglik : float
        Maximised log-likelihood of the estimation window.
    k : int
        Number of transition covariates.
    nobs : int
        Months in the estimation window.
    zcols : list of str
        Names of the covariate columns, in the order of the slopes.
    xi_pred, xi_filt : np.ndarray, length nobs
        Predicted and filtered probabilities of regime 1 for every month of
        the window, from the filter run at theta.
    index : pd.Index
        The months of the window.
    converged : bool
        Whether the optimiser reported success on the start that won.  True
        for evaluate(), which does not search.
    n_starts : int
        Number of starting points tried.  0 for evaluate().

    Attributes
    ----------
    params : dict
        unpack(theta, k), computed once when the object is created.
    npar : int
        Number of estimated parameters, len(theta).
    aic, bic : float
        Akaike and Bayesian information criteria, computed when read.
    """

    def __init__(self, theta, loglik, k, nobs, zcols, xi_pred, xi_filt,
                 index, converged, n_starts):
        # store everything the fit produced
        self.theta     = theta
        self.loglik    = loglik
        self.k         = k
        self.nobs      = nobs
        self.zcols     = list(zcols)        # own copy of the column names
        self.xi_pred   = xi_pred            # P(regime 1 | data through t-1)
        self.xi_filt   = xi_filt            # P(regime 1 | data through t)
        self.index     = index
        self.converged = converged
        self.n_starts  = n_starts
        # named parameters, computed once here and read by every later step
        self.params    = unpack(theta, k)
        self.npar      = len(theta)         # number of estimated parameters

    # -- information criteria ---------------------------------------------
    # @property lets these be read like attributes (res.aic, not res.aic()).
    # They are computed from loglik and npar each time they are read.
    @property
    def aic(self):
        """Akaike information criterion, -2 * loglik + 2 * npar."""
        return -2 * self.loglik + 2 * self.npar

    @property
    def bic(self):
        """Bayesian information criterion, -2 * loglik + log(nobs) * npar."""
        return -2 * self.loglik + np.log(self.nobs) * self.npar

    def summary(self, se=None):
        """
        The estimates as a readable table, the form 03 and 07 print.

        Shows the log-likelihood, AIC and BIC, then per regime the short-rate
        parameters, the standard deviations and the bond-equity correlation
        implied by each covariance matrix, then the two risky means and the
        transition intercepts and slopes.

        Parameters
        ----------
        se : np.ndarray, optional
            Output of standard_errors().  If given, the standard errors of the
            transition intercepts and slopes are added.  Those of the other
            parameters are left out: the covariance entries are estimated in
            their packed Cholesky form, so their standard errors do not refer
            to the standard deviations shown.

        Returns
        -------
        str
            The table as text, ready to print.
        """
        p = self.params
        S1, S2 = p["Sigma"][0], p["Sigma"][1]
        # bond/equity correlation implied by each covariance matrix
        c1 = S1[1, 2] / np.sqrt(S1[1, 1] * S1[2, 2])
        c2 = S2[1, 2] / np.sqrt(S2[1, 1] * S2[2, 2])
        lines = [f"log-likelihood {self.loglik:12.3f}   "
                 f"parameters {self.npar}   observations {self.nobs}",
                 f"AIC {self.aic:12.3f}   BIC {self.bic:12.3f}",
                 "-" * 62,
                 f"{'':16s}{'regime 1':>14s}{'regime 2':>14s}",
                 f"{'mu_r':16s}{p['mu_r'][0]:14.4f}{p['mu_r'][1]:14.4f}",
                 f"{'rho':16s}{p['rho'][0]:14.4f}{p['rho'][1]:14.4f}",
                 f"{'sd(short rate)':16s}"
                 f"{np.sqrt(S1[0, 0]):14.4f}{np.sqrt(S2[0, 0]):14.4f}",
                 f"{'sd(bond)':16s}"
                 f"{np.sqrt(S1[1, 1]):14.4f}{np.sqrt(S2[1, 1]):14.4f}",
                 f"{'sd(equity)':16s}"
                 f"{np.sqrt(S1[2, 2]):14.4f}{np.sqrt(S2[2, 2]):14.4f}",
                 f"{'corr(bond,eq)':16s}{c1:14.4f}{c2:14.4f}",
                 "-" * 62,
                 f"{'mu_b':16s}{p['mu_b']:14.4f}",
                 f"{'mu_e':16s}{p['mu_e']:14.4f}",
                 "-" * 62,
                 f"{'a (intercept)':16s}{p['a'][0]:14.4f}{p['a'][1]:14.4f}"]
        # one row per covariate, in the order of zcols
        for j, name in enumerate(self.zcols):
            lines.append(f"{'b: ' + name:16s}"
                         f"{p['b'][0][j]:14.4f}{p['b'][1][j]:14.4f}")
        if se is not None:                        # transition block only
            lines += ["-" * 62,
                      "standard errors on the transition block:",
                      f"{'a':16s}{se[N_FIXED - 2]:14.4f}{se[N_FIXED - 1]:14.4f}"]
            for j, name in enumerate(self.zcols):
                lines.append(f"{'b: ' + name:16s}"
                             f"{se[N_FIXED + j]:14.4f}"
                             f"{se[N_FIXED + self.k + j]:14.4f}")
        return "\n".join(lines)


def fit(df, zcols, n_random=6, theta0=None, maxiter=4000, polish=True,
        include_default_start=True, strict=True, verbose=False):
    """
    Estimate the model on one window by maximum likelihood, searching from
    several starting points and keeping the best (Section 4.4).

    Parameters
    ----------
    df : pd.DataFrame
        Estimation window.
    zcols : list of str
        Transition covariate columns, empty for M0.
    n_random : int
        Number of randomly shifted starts from starting_values().
    theta0 : array_like, optional
        One more starting point, tried first.  This is how the nested start
        of nest_start() and the previous month's estimate in the backtest
        enter.
    maxiter : int
        Maximum number of iterations for each optimiser run.
    polish : bool
        If True, follow the first L-BFGS-B search with a Nelder-Mead pass and
        a second L-BFGS-B search, and keep the result only if it improves.
    include_default_start : bool
        If True, also try the data-driven start from starting_values().
    strict : bool
        Passed to prepare().  False when the covariates are standardised.
    verbose : bool
        If True, print the log-likelihood reached from every start.

    Returns
    -------
    MSResult
        The best solution across all starts, relabelled by relabel(), with
        the filter run once more at the final estimate.

    Raises
    ------
    RuntimeError
        Every start failed.

    Notes
    -----
    Full-sample fits and the refit months of the backtest use the full
    search.  Every other backtest month uses the cheap one: polish=False,
    n_random=0, include_default_start=False, and the previous month's
    estimate as theta0, so a single search from that point.
    """
    # turn the window into arrays and check it
    y, rlag, Z, index = prepare(df, zcols, strict=strict)
    k = Z.shape[1]                                   # number of covariates

    # ---- starting points ----------------------------------------------------
    # theta0 first (nested start or previous month's estimate), then the
    # data-driven start and its randomly shifted copies
    starts = []
    if theta0 is not None:
        starts.append(np.asarray(theta0, dtype=float))
    pool = starting_values(y, rlag, Z, k, n_random=n_random)
    if not include_default_start:
        pool = pool[1:]                              # drop the data-driven start
    starts.extend(pool)
    # no theta0, no data-driven start and no random starts would leave nothing
    # to search from; fall back to the full pool
    if not starts:
        starts = starting_values(y, rlag, Z, k, n_random=n_random)

    # ---- one search per start, keep the best --------------------------------
    # nll is the negative log-likelihood, which the optimisers minimise
    args = (y, rlag, Z, k)
    best_theta, best_nll, best_ok = None, np.inf, False
    for s, th0 in enumerate(starts):
        th = np.asarray(th0, dtype=float)
        ok = False
        try:
            # L-BFGS-B from this start.  The gradient is found numerically,
            # which costs many likelihood evaluations per step, hence the
            # higher cap on function evaluations (maxfun)
            r = minimize(neg_loglik, th, args=args, method="L-BFGS-B",
                         options={"maxiter": maxiter, "maxfun": 20 * maxiter})
            th, nll, ok = r.x, r.fun, bool(r.success)
            if polish:
                th_pre, nll_pre, ok_pre = th, nll, ok
                # Nelder-Mead needs no gradient, so it can move along nearly
                # flat directions where L-BFGS-B stops; then L-BFGS-B again
                r2 = minimize(neg_loglik, th, args=args, method="Nelder-Mead",
                              options={"maxiter": maxiter, "maxfev": maxiter,
                                       "xatol": 1e-8, "fatol": 1e-10})
                r3 = minimize(neg_loglik, r2.x, args=args, method="L-BFGS-B",
                              options={"maxiter": maxiter,
                                       "maxfun": 20 * maxiter})
                # keep the refined result only if its likelihood is higher
                if np.isfinite(r3.fun) and r3.fun < nll_pre:
                    th, nll, ok = r3.x, r3.fun, bool(r3.success)
                else:
                    th, nll, ok = th_pre, nll_pre, ok_pre
        except Exception:                            # a start that fails is skipped
            continue
        if verbose:
            print(f"   start {s:2d}: loglik {-nll:12.4f}  ok={ok}")
        if np.isfinite(nll) and nll < best_nll:      # best so far
            best_theta, best_nll, best_ok = th, nll, ok

    if best_theta is None:
        raise RuntimeError("every optimisation start failed")

    # ---- final result -------------------------------------------------------
    theta = relabel(best_theta, k)                   # regime 2 = volatile
    # run the filter once more at the final estimate for the regime
    # probabilities of every month
    ll, xi_pred, xi_filt = hamilton_filter(theta, y, rlag, Z, k)
    return MSResult(theta, ll, k, len(y), zcols, xi_pred, xi_filt,
                    index, best_ok, len(starts))


def evaluate(df, zcols, theta, strict=True):
    """
    Run the filter at a given parameter vector, without estimating anything,
    and return the result in the same form as fit().

    Used in two ways: to restore an estimate saved earlier, so a long fit
    need not be repeated, and to apply a parameter vector from elsewhere,
    such as Ang and Bekaert's published AB_MODEL_II_THETA, to a panel.  The
    filter gives the same output every time for the same theta, so a
    restored fit is identical to the original.

    Parameters
    ----------
    df : pd.DataFrame
        Window to run the filter on.
    zcols : list of str
        Transition covariate columns, empty for M0.
    theta : array_like, length 20 + 2k
        The parameter vector to use, layout in the module docstring.
    strict : bool
        Passed to prepare().  False when the covariates are standardised.

    Returns
    -------
    MSResult
        The likelihood and regime probabilities at theta.  converged is True
        and n_starts is 0, since no search took place.

    Raises
    ------
    ValueError
        theta has the wrong length for the number of covariates.
    """
    y, rlag, Z, index = prepare(df, zcols, strict=strict)
    k = Z.shape[1]
    theta = np.asarray(theta, dtype=float)
    if len(theta) != N_FIXED + 2 * k:             # layout check before filtering
        raise ValueError(f"theta has length {len(theta)}, expected "
                         f"{N_FIXED + 2 * k} for k={k}")
    ll, xi_pred, xi_filt = hamilton_filter(theta, y, rlag, Z, k)
    return MSResult(theta, ll, k, len(y), zcols, xi_pred, xi_filt,
                    index, True, 0)


def standard_errors(res, df, zcols, step=1e-4, strict=True):
    """
    Standard errors of the estimates, from the curvature of the
    log-likelihood at its maximum (Section 4.4.5)

    Computes the matrix of second derivatives of minus the log-likelihood at
    the estimate numerically, by evaluating it at small changes of each pair
    of parameters.  The inverse of that matrix is the asymptotic covariance
    matrix of the estimates, and the square roots of its diagonal are the
    standard errors.

    Parameters
    ----------
    res : MSResult
        The fitted model.
    df : pd.DataFrame
        The window the model was fitted on.
    zcols : list of str
        Transition covariate columns, empty for M0.
    step : float
        Size of each change, relative to the parameter: the step for
        parameter j is step * (|theta_j| + 1).
    strict : bool
        Passed to prepare().  False when the covariates are standardised.

    Returns
    -------
    np.ndarray, length 20 + 2k
        One standard error per entry of theta, in the units of theta.  NaN
        for a parameter whose variance comes out zero or negative, which
        marks it as weakly identified.  NaN throughout if the matrix cannot
        be inverted or the likelihood fails at one of the evaluation points.

    Notes
    -----
    The transition intercepts and slopes are in their natural units, so their
    standard errors can be compared with Table 6 of Ang and Bekaert directly.
    The covariance entries are estimated in their packed Cholesky form, so
    their standard errors refer to that form and not to the standard
    deviations shown in the tables.
    """
    # turn the window into arrays and set up the matrix of second derivatives
    y, rlag, Z, _ = prepare(df, zcols, strict=strict)
    k = Z.shape[1]
    th = np.asarray(res.theta, dtype=float)
    n = len(th)
    H = np.zeros((n, n))                    # second derivatives, filled below
    h = step * (np.abs(th) + 1.0)           # step size for each parameter

    # A custom error used as a signal.  If the likelihood fails at any point
    # of the grid, the derivatives are meaningless, so the whole computation
    # is abandoned at once instead of continuing with a placeholder value.
    class _Dead(Exception):
        pass

    def f(t):
        """
        Minus the log-likelihood at t.  Raises _Dead instead of returning the
        1e10 placeholder of neg_loglik(), so that value never enters a
        derivative.
        """
        v = neg_loglik(t, y, rlag, Z, k)
        if not np.isfinite(v) or v >= 1e9:
            raise _Dead
        return v

    try:
        f0 = f(th)                          # value at the estimate itself
        # the matrix is symmetric, so only the upper triangle is computed
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    # second derivative in one parameter: step it up and
                    # down, (f(+h) - 2 f(0) + f(-h)) / h^2
                    tp, tm = th.copy(), th.copy()
                    tp[i] += h[i]; tm[i] -= h[i]
                    H[i, i] = (f(tp) - 2 * f0 + f(tm)) / h[i] ** 2
                else:
                    # cross derivative in two parameters: step both in the
                    # four combinations of up and down
                    a = th.copy(); a[i] += h[i]; a[j] += h[j]
                    b = th.copy(); b[i] += h[i]; b[j] -= h[j]
                    c = th.copy(); c[i] -= h[i]; c[j] += h[j]
                    d = th.copy(); d[i] -= h[i]; d[j] -= h[j]
                    H[i, j] = H[j, i] = ((f(a) - f(b) - f(c) + f(d))
                                         / (4 * h[i] * h[j]))
    except _Dead:                           # likelihood failed somewhere
        return np.full(n, np.nan)

    # The inverse of H is the asymptotic covariance matrix of the estimates;
    # its diagonal holds their variances.
    try:
        var = np.diag(np.linalg.inv(H))
    except np.linalg.LinAlgError:           # H cannot be inverted
        return np.full(n, np.nan)
    # standard error = square root of the variance; NaN where the variance is
    # zero or negative, which happens where the likelihood is flat
    return np.where(var > 0, np.sqrt(var), np.nan)


# ==========================================================================
# One-step-ahead regime forecast (reported, not used for classification)
# ==========================================================================
def predict_regime(res, z_next):
    """
    Probability that the month after the estimation window is a regime 1
    month, xi_{T+1|T}.

    The prediction step of the filter (Equation 4.19) run once past
    the last month of the window: the filtered probability of the last month,
    carried forward with the transition probabilities at z_next.

    Parameters
    ----------
    res : MSResult
        The fitted model.  Its last filtered probability is the starting
        point.
    z_next : array_like, length k
        Covariate vector observed at the close of month T, the decision month,
        which drives the transition into month T+1.  Empty array for M0.

    Returns
    -------
    float
        Probability of regime 1 in month T+1, given everything known at the
        close of month T.

    Notes
    -----
    The same number as the soft rule's mixing weight pi1 in
    conditional_moments(), so the `xi_pred` column of every backtest repeats
    `pi1_s`.  Kept as a separately named diagnostic.  The hard rule does not
    use it: it classifies on the filtered probability of the current month.
    """
    P, Q = stay_probs_at(res.params, np.atleast_1d(np.asarray(z_next, dtype=float)))
    xi_TT = res.xi_filt[-1]                       # Pr(s_T = 1 | I_T)
    return float(P * xi_TT + (1.0 - Q) * (1.0 - xi_TT))


def lr_test(res_restricted, res_full):
    """
    Likelihood-ratio test of a smaller model against a larger one that
    contains it (Equation 4.12).

    Tests whether the parameters the larger model adds improve the fit by
    more than chance would.  Under the null hypothesis that they are all zero,
    the statistic follows a chi-squared distribution with one degree of
    freedom per added parameter.

    Parameters
    ----------
    res_restricted, res_full : MSResult
        The two fitted models.  The first must be a special case of the
        second, for example M1 and M2.

    Returns
    -------
    stat : float
        2 * (loglik of the larger model - loglik of the smaller one).
    dof : int
        Number of added parameters: two per rung of the ladder, so 2 for M1
        against M2 and 8 for M0 against M4.
    pval : float
        Probability of a statistic at least this large under the null, the
        upper tail of the chi-squared distribution with dof degrees of
        freedom.

    Notes
    -----
    The returned statistic is left as computed.  Only for the p-value is it
    set to zero if negative.  A negative statistic means the optimiser of the
    larger model failed, since a model that contains another cannot fit
    worse.
    """
    stat = 2.0 * (res_full.loglik - res_restricted.loglik)
    dof = res_full.npar - res_restricted.npar
    return stat, dof, float(chi2.sf(max(stat, 0.0), dof))


# ==========================================================================
# Ang and Bekaert's published Model II estimates
# ==========================================================================
# Table 6 of the working paper (2002), page 24, Model II column.  All parameters are
# monthly and in percent except a and b, whose covariate is the monthly short
# rate in percent.  Assembled in this module's theta layout so 03 can compare
# element by element.
AB_MODEL_II_THETA = np.concatenate([
    [0.01, 0.04],                                     # mu_r(1), mu_r(2)
    [0.99, 0.94],                                     # rho(1),  rho(2)
    [0.07, 0.68],                                     # mu_b, mu_e
    packed_from_chol_elements(0.03,  0.11, 1.75,  0.31, 0.61, 3.33),   # regime 1
    packed_from_chol_elements(0.09, -0.26, 3.97, -1.07, 1.73, 5.16),   # regime 2
    [4.67, -0.83],                                    # a1, a2
    [-6.06], [3.43],                                  # b1, b2
])

# Standard errors as printed in Table 6, in the same natural units as the
# table, not in theta units.  For the Cholesky block they refer to R_rr, R_br,
# R_bb, R_er, R_eb, R_ee directly.
# The repeated pairs are in the source: R_er(1) and R_eb(1) are both 0.19, and
# R_er(2) and R_eb(2) are both 0.43.  Not a transcription slip.
AB_MODEL_II_SE = {
    "mu_r(1)": 0.00, "mu_r(2)": 0.02,
    "rho(1)":  0.01, "rho(2)":  0.03,
    "mu_b":    0.09, "mu_e":    0.16,
    "R_rr(1)": 0.00, "R_rr(2)": 0.01,
    "R_br(1)": 0.10, "R_br(2)": 0.31,
    "R_bb(1)": 0.09, "R_bb(2)": 0.25,
    "R_er(1)": 0.19, "R_er(2)": 0.43,
    "R_eb(1)": 0.19, "R_eb(2)": 0.43,
    "R_ee(1)": 0.13, "R_ee(2)": 0.33,
    "a1": 0.84, "a2": 1.24,
    "b1": 1.83, "b2": 2.04,
}


def natural_params(theta, k):
    """
    Express a parameter vector in the names and form of Ang and Bekaert's
    Table 6, so 03 can print the replication parameter by parameter next to
    their estimates.

    Unlike unpack(), the covariance block is given as the six Cholesky
    elements R_rr, R_br, ... that Table 6 reports, not as covariance
    matrices.

    Parameters
    ----------
    theta : array_like, length 20 + 2k
        Parameter vector, layout in the module docstring.
    k : int
        Number of transition covariates.

    Returns
    -------
    dict
        One entry per parameter of Table 6.  The keys match those of
        AB_MODEL_II_SE, so each estimate can be set against their standard
        error.  b1 and b2 are included only for k > 0, and only the slope on
        the first covariate, the short rate, since Table 6 has no
        counterpart for others.
    """
    p = unpack(theta, k)
    L1, L2 = p["L"][0], p["L"][1]                 # Cholesky elements, not sds
    out = {
        "mu_r(1)": p["mu_r"][0], "mu_r(2)": p["mu_r"][1],
        "rho(1)":  p["rho"][0],  "rho(2)":  p["rho"][1],
        "mu_b":    p["mu_b"],    "mu_e":    p["mu_e"],
        "R_rr(1)": L1[0, 0], "R_rr(2)": L2[0, 0],
        "R_br(1)": L1[1, 0], "R_br(2)": L2[1, 0],
        "R_bb(1)": L1[1, 1], "R_bb(2)": L2[1, 1],
        "R_er(1)": L1[2, 0], "R_er(2)": L2[2, 0],
        "R_eb(1)": L1[2, 1], "R_eb(2)": L2[2, 1],
        "R_ee(1)": L1[2, 2], "R_ee(2)": L2[2, 2],
        "a1": p["a"][0], "a2": p["a"][1],
    }
    if k:
        out["b1"] = p["b"][0][0]
        out["b2"] = p["b"][1][0]
    return out


def derived_moments(theta, k):
    """
    Standard deviations and the bond-equity correlation implied by a
    parameter vector, per regime.

    Table 6 of Ang and Bekaert (2002) reports Cholesky elements, while their text
    and the thesis quote standard deviations and correlations.  This function
    converts one into the other (Appendix A.2), so 03 can compare the
    replication with both.  natural_params() gives the Table 6 form.

    Parameters
    ----------
    theta : array_like, length 20 + 2k
        Parameter vector, layout in the module docstring.
    k : int
        Number of transition covariates.

    Returns
    -------
    dict
        For each regime: the standard deviation of the short-rate, bond and
        equity shocks in percent per month, and the correlation between the
        bond and the equity shock.
    """
    p = unpack(theta, k)
    out = {}
    for i, tag in ((0, "1"), (1, "2")):
        S = p["Sigma"][i]                         # level form, not the factor
        out[f"sd short rate ({tag})"] = np.sqrt(S[0, 0])
        out[f"sd bond ({tag})"]       = np.sqrt(S[1, 1])
        out[f"sd equity ({tag})"]     = np.sqrt(S[2, 2])
        out[f"corr bond-eq ({tag})"]  = S[1, 2] / np.sqrt(S[1, 1] * S[2, 2])
    return out


# ==========================================================================
# Self-test on simulated data
# ==========================================================================
if __name__ == "__main__":
    # Simulate from a known parameter vector and check the estimator finds it
    # back.  This is what confirms the filter, the packing and the labelling
    # convention are consistent with each other.
    import pandas as pd

    rng = np.random.default_rng(42)
    T = 1200

    # true parameters, in the neighbourhood of the published estimates
    mu_r, rho  = np.array([0.02, 0.09]), np.array([0.95, 0.85])
    mu_b, mu_e = 0.10, 0.60
    S = [np.diag([0.02, 1.75, 3.40]) ** 2,        # calm regime
         np.diag([0.09, 3.98, 5.55]) ** 2]        # volatile regime
    L = [np.linalg.cholesky(s) for s in S]

    # exogenous standardised AR(1) covariate, what a recursively standardised
    # macro driver looks like.  The coefficients keep both regimes off the
    # absorbing boundary: P runs over 0.49 to 0.99 and Q over 0.51 to 0.99.
    # If one regime did become absorbing the sample would contain essentially
    # one state and the transition parameters would not be identified.
    true_a  = np.array([2.5, 2.5])
    true_b  = [np.array([-0.9]), np.array([0.9])]
    z = np.empty(T)
    z[0] = 0.0
    for t in range(1, T):                          # unit-variance AR(1)
        z[t] = 0.85 * z[t - 1] + np.sqrt(1 - 0.85 ** 2) * rng.standard_normal()

    r = np.empty(T); rb = np.empty(T); re = np.empty(T); state = np.empty(T, int)
    r[0], s_t = 0.45, 0
    for t in range(1, T):
        # z[t] drives the move from s_{t-1} to s_t, the same convention the
        # filter assumes.  Here it is exogenous, so the test does not depend on
        # the data build.
        P = _logistic(true_a[0] + true_b[0][0] * z[t])
        Q = _logistic(true_a[1] + true_b[1][0] * z[t])
        stay = P if s_t == 0 else Q
        s_t = s_t if rng.random() < stay else 1 - s_t      # draw the regime
        e = L[s_t] @ rng.standard_normal(3)                # correlated shock
        r[t]  = mu_r[s_t] + rho[s_t] * r[t - 1] + e[0]
        rb[t] = mu_b + e[1]
        re[t] = mu_e + e[2]
        state[t] = s_t

    # shift into panel form: r_obs is this month's rate, r last month's
    sim = pd.DataFrame({"r_obs": r[1:], "r": r[:-1],
                        "rb_x": rb[1:], "re_x": re[1:], "z_exog": z[1:]},
                       index=pd.period_range("1900-01", periods=T - 1, freq="M"))

    print("SELF-TEST: recovering known parameters from simulated data")
    print(f"months in the volatile regime: {state[1:].mean():.1%}\n")
    res = fit(sim, ["z_exog"], n_random=3, verbose=True)
    print("\n" + res.summary())

    print("\nTRUE VALUES")
    print(f"{'mu_r':16s}{mu_r[0]:14.4f}{mu_r[1]:14.4f}")
    print(f"{'rho':16s}{rho[0]:14.4f}{rho[1]:14.4f}")
    print(f"{'sd(short rate)':16s}{np.sqrt(S[0][0,0]):14.4f}{np.sqrt(S[1][0,0]):14.4f}")
    print(f"{'sd(bond)':16s}{np.sqrt(S[0][1,1]):14.4f}{np.sqrt(S[1][1,1]):14.4f}")
    print(f"{'sd(equity)':16s}{np.sqrt(S[0][2,2]):14.4f}{np.sqrt(S[1][2,2]):14.4f}")
    print(f"{'mu_b':16s}{mu_b:14.4f}")
    print(f"{'mu_e':16s}{mu_e:14.4f}")
    print(f"{'a':16s}{true_a[0]:14.4f}{true_a[1]:14.4f}")
    print(f"{'b: z_exog':16s}{true_b[0][0]:14.4f}{true_b[1][0]:14.4f}")

    # the filter should also recover the state path it was simulated from.
    # sim starts at the second simulated month, and prepare() drops nothing,
    # so the comparison starts at state[1].
    hit = ((res.xi_filt > 0.5) == (state[1:] == 0)).mean()
    print(f"\nfiltered regime classified correctly in {hit:.1%} of months")

    # ---- conditional moments: check the blend does what it should ---------
    # At the published theta the weights move continuously with the short rate.
    # Raw Sigma(i) would give a flat line.
    print("\nCONDITIONAL MOMENTS CHECK (gamma = 5, published Model II theta)")
    pub = unpack(AB_MODEL_II_THETA, 1)
    print(f"{'r_ann %':>9s}{'P':>8s}{'Q':>8s}"
          f"{'reg1 bond':>12s}{'reg1 eq':>10s}{'reg2 bond':>12s}{'reg2 eq':>10s}")
    for ra in (1, 4, 8, 15):
        rm = ((1 + ra / 100) ** (1 / 12) - 1) * 100      # annualised -> monthly %
        z = np.array([rm])
        P, Q = stay_probs_at(pub, z)                     # same for both regimes
        ws = []
        for regime in (0, 1):
            mu_c, Sig_c, _, _, _ = conditional_moments(pub, z, regime)
            ws.append((100.0 / 5.0) * np.linalg.solve(Sig_c, mu_c))   # w = Sigma^-1 mu / gamma
        print(f"{ra:9d}{P:8.3f}{Q:8.3f}"
              f"{ws[0][0]:12.3f}{ws[0][1]:10.3f}{ws[1][0]:12.3f}{ws[1][1]:10.3f}")
    print("   expected: reg1 equity 1.15 -> 0.50, reg2 equity 0.76 -> 0.48")

    print(f"\nsimulated long-run P(regime 1) at published theta: "
          f"{stationary_prob(pub, n=50_000):.4f}   (AB quote 0.7014; Table 6")
    print("   rounds mu_r(1) to two decimals and rho(1) to 0.99, and the")
    print("   long-run level mu_r/(1-rho) is very sensitive to both)")