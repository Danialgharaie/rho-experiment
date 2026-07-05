"""Paired comparison: how costly is confidently assuming rho_hat=0, by true rho.

Uses common random numbers (same replicate draws) so the reported delta is a
within-replicate comparison, not two independently noisy means -- same
discipline as the lambda-experiment's paired.py.
"""
import numpy as np

from sim import generate_world, evaluate_strategy, optimal_split, M, NREPS

RHO_TRUE_GRID = np.round(np.linspace(0.0, 1.0, 11), 2)


def paired_loss_at_rho_hat_zero(nreps=NREPS, seed=7):
    rng = np.random.default_rng(seed)
    zero_kc, zero_ke = optimal_split(0.0)

    out = {}
    for rho_true in RHO_TRUE_GRID:
        oracle_kc, oracle_ke = optimal_split(rho_true)
        deltas = []
        for _ in range(nreps):
            q, z_c, z_e, hits, shortlist_idx = generate_world(rng, rho_true)
            q_s, z_c_s, z_e_s = q[shortlist_idx], z_c[shortlist_idx], z_e[shortlist_idx]
            xi_c = rng.standard_normal(M)
            xi_e = rng.standard_normal(M)

            r_oracle = evaluate_strategy(q_s, z_c_s, z_e_s, xi_c, xi_e, rho_true,
                                          oracle_kc, oracle_ke, hits, shortlist_idx)
            r_zero = evaluate_strategy(q_s, z_c_s, z_e_s, xi_c, xi_e, rho_true,
                                        zero_kc, zero_ke, hits, shortlist_idx)
            deltas.append(r_zero - r_oracle)

        deltas = np.array(deltas)
        lo, med, hi = np.percentile(deltas, 16), np.median(deltas), np.percentile(deltas, 84)
        out[rho_true] = (med, lo, hi)
        print(f"rho_true={rho_true:.1f}  paired delta (rho_hat=0 vs oracle): "
              f"med={med:+.4f} [16-84: {lo:+.4f}, {hi:+.4f}]")

    return out


if __name__ == "__main__":
    paired_loss_at_rho_hat_zero()
