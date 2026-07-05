"""Core simulation primitives for the misspecified-rho cost-of-precision experiment.

Ports the mechanics of "The Price of Precision" (post 5): a free cheap screen
builds a shortlist, then a fixed budget buys extra cheap and/or precise reads
per shortlisted compound. Cheap and precise reads from the SAME instrument
share that instrument's systematic bias (parameterised by rho); the private
component of the noise averages away with more reads, the systematic
component does not.

This module adds one new axis: the price-aware split's *allocation* decision
can be made using an assumed rho_hat that differs from the true rho used to
generate the world. Everything downstream of the allocation (interpreting
the resulting reads into a posterior mean) uses the true rho -- i.e. the
mistake being isolated is "I priced precision using the wrong rho", not
"I also misread my own instruments afterwards". See the post's "What This
Is Not" section for why that split is the honest scope for this note.
"""
import numpy as np

N = 4000
H = 80
M = 500
L = 100
# NOTE: post 5 used BUDGET=4000 (8 units/compound). At r=32 that only affords
# a quarter of a precise read's worth of averaging (KE_MAX = 8/32 = 0.25) --
# too thin a margin to be worth diverting from cheap reads except when
# rho_hat is already large, so optimal_split collapses to a near-corner
# solution (ke near 0 or pinned at the 0.25 ceiling) for most rho_hat.
# Raising the budget to 16000 (32 units/compound, KE_MAX = 32/32 = 1.0) gives
# the split real interior room to trade cheap reads against a genuine
# fraction of a precise read, which is what makes "the wrong rho" a decision
# that can actually go wrong. Everything else (N, H, M, L, sigma_c, sigma_e,
# r) is unchanged from post 5.
BUDGET = 16000
SIGMA_C = 1.0
SIGMA_E = 0.35
# NOTE: the crossing r* ~= 8.16 is, by construction, the price at which
# all-cheap and all-precise tie -- i.e. the point of *maximum* indifference
# to allocation. Every allocation performs similarly there, so a wrong
# rho_hat has almost nothing to be wrong about. The real risk from
# misspecification shows up away from the crossing, where the correct
# allocation is a strong, rho-dependent skew and a wrong rho_hat can pick
# the wrong skew entirely. r=32 matches the far side already explored in
# post 5's own crossing figure (where all-precise collapsed to 0.43).
R = 32.0
NREPS = 80

PER_COMPOUND_BUDGET = BUDGET / M  # = 32.0
KE_MAX = PER_COMPOUND_BUDGET / R  # = 1.0 at r=32 (the old BUDGET=4000/r=32 setup gave 0.25)


def var_c(rho, k_total):
    """Variance of the cheap instrument's aggregate over k_total reads (incl. screen)."""
    return SIGMA_C**2 * (rho + (1 - rho) / k_total)


def var_e(rho, k):
    """Variance of the precise instrument's aggregate over k reads. inf if k<=0."""
    if k <= 0:
        return np.inf
    return SIGMA_E**2 * (rho + (1 - rho) / k)


def optimal_split(rho_hat, b=PER_COMPOUND_BUDGET, r=R, grid=2001):
    """Return (kc_extra, ke) maximising assumed posterior precision under rho_hat.

    kc_extra: extra cheap reads bought beyond the free screen (cost 1 each).
    ke:       precise reads bought (cost r each).
    kc_extra + ke*r == b (budget spent in full).
    """
    kes = np.linspace(0.0, b / r, grid)
    kc_extras = b - kes * r
    kc_totals = 1.0 + kc_extras

    vc = var_c(rho_hat, kc_totals)
    prec = 1.0 / vc

    with np.errstate(divide="ignore"):
        ve = np.where(kes > 0, SIGMA_E**2 * (rho_hat + (1 - rho_hat) / np.maximum(kes, 1e-12)), np.inf)
    prec_e = np.where(kes > 0, 1.0 / ve, 0.0)

    total_prec = prec + prec_e
    # Tiny tie-break toward larger ke: when rho_hat pushes both instruments
    # to their bias floor (rho_hat -> 1), many allocations tie exactly in
    # assumed precision. The continuous limit from rho_hat < 1 always picks
    # the largest-ke member of that tied set, so nudge the argmax the same
    # way rather than let it fall on an arbitrary grid point.
    total_prec = total_prec + 1e-9 * kes
    best = np.argmax(total_prec)
    return float(kc_extras[best]), float(kes[best])


def generate_world(rng, rho_true):
    """Draw one replicate's hidden qualities, instrument biases, and the free screen."""
    q = rng.standard_normal(N)
    hits = frozenset(np.argsort(q)[-H:].tolist())
    z_c = rng.standard_normal(N)
    z_e = rng.standard_normal(N)

    eta_screen = rng.standard_normal(N)
    screen = q + np.sqrt(rho_true) * SIGMA_C * z_c + np.sqrt(1 - rho_true) * SIGMA_C * eta_screen

    shortlist_idx = np.argsort(screen)[-M:]
    return q, z_c, z_e, hits, shortlist_idx


def evaluate_strategy(q_s, z_c_s, z_e_s, xi_c, xi_e, rho_true, kc_extra, ke, hits, shortlist_idx):
    """Compute recall@L for one strategy's allocation, on one replicate's shortlist.

    Downstream interpretation of the reads uses rho_true (see module docstring):
    only the allocation (kc_extra, ke) may be based on a misspecified rho_hat.
    """
    kc_total = 1.0 + kc_extra

    agg_c = q_s + np.sqrt(rho_true) * SIGMA_C * z_c_s + np.sqrt(1 - rho_true) * SIGMA_C / np.sqrt(kc_total) * xi_c
    vc_true = var_c(rho_true, kc_total)
    precision = 1.0 + 1.0 / vc_true
    weighted = agg_c / vc_true

    if ke > 0:
        agg_e = q_s + np.sqrt(rho_true) * SIGMA_E * z_e_s + np.sqrt(1 - rho_true) * SIGMA_E / np.sqrt(ke) * xi_e
        ve_true = var_e(rho_true, ke)
        precision += 1.0 / ve_true
        weighted += agg_e / ve_true

    posterior_mean = weighted / precision

    order = np.argsort(posterior_mean)[::-1]
    top_l = shortlist_idx[order[:L]]
    return len(set(int(i) for i in top_l) & hits) / H
