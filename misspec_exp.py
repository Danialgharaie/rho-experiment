"""
The price of a wrong prior.

Sweep an assumed rho_hat (used only to decide the price-aware split's
cheap/precise allocation) against a true rho (used to generate the world)
at a fixed cost ratio r=32 -- away from the r*~=8.16 crossing, where the
correct allocation is a genuine, rho-dependent skew rather than a near-tie.
Compare against:
  - oracle split   (allocates using the true rho)
  - all-cheap      (never buys precise reads)
  - all-precise    (spends the whole per-compound budget on precise reads)

Two figures:
  1. heatmap: recall(misspecified) - recall(oracle) over (rho_hat, rho_true)
  2. lines:   recall vs rho_hat at a few true-rho slices, with oracle/
              all-cheap/all-precise reference lines
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sim import (
    generate_world, evaluate_strategy, optimal_split,
    N, H, M, L, R, NREPS, KE_MAX,
)

FIGURES = Path(__file__).parent / "figures"
FIGURES.mkdir(exist_ok=True)

RHO_GRID = np.round(np.linspace(0.0, 1.0, 11), 2)  # 0.0, 0.1, ..., 1.0

STYLE = {
    "font.family": "monospace",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
}


def run_grid(nreps=NREPS, seed=42):
    """Returns:
      recalls[rho_true]['oracle' | 'all_cheap' | 'all_precise'] -> array[nreps]
      recalls[rho_true]['misspec'][rho_hat] -> array[nreps]
    """
    rng = np.random.default_rng(seed)
    out = {}

    # Pure-strategy allocations are fixed by construction, independent of rho.
    from sim import PER_COMPOUND_BUDGET
    kc_extra_all_cheap = PER_COMPOUND_BUDGET  # spend the whole budget on cheap reads
    ke_all_precise = KE_MAX                   # spend the whole budget on precise reads

    for rho_true in RHO_GRID:
        print(f"  rho_true={rho_true:.1f} ...")
        oracle_kc, oracle_ke = optimal_split(rho_true)

        cell = {
            "oracle": [],
            "all_cheap": [],
            "all_precise": [],
            "misspec": {rh: [] for rh in RHO_GRID},
        }

        for rep in range(nreps):
            q, z_c, z_e, hits, shortlist_idx = generate_world(rng, rho_true)
            q_s = q[shortlist_idx]
            z_c_s = z_c[shortlist_idx]
            z_e_s = z_e[shortlist_idx]
            xi_c = rng.standard_normal(M)
            xi_e = rng.standard_normal(M)

            cell["oracle"].append(
                evaluate_strategy(q_s, z_c_s, z_e_s, xi_c, xi_e, rho_true,
                                   oracle_kc, oracle_ke, hits, shortlist_idx)
            )
            cell["all_cheap"].append(
                evaluate_strategy(q_s, z_c_s, z_e_s, xi_c, xi_e, rho_true,
                                   kc_extra_all_cheap, 0.0, hits, shortlist_idx)
            )
            cell["all_precise"].append(
                evaluate_strategy(q_s, z_c_s, z_e_s, xi_c, xi_e, rho_true,
                                   0.0, ke_all_precise, hits, shortlist_idx)
            )
            for rho_hat in RHO_GRID:
                mis_kc, mis_ke = optimal_split(rho_hat)
                cell["misspec"][rho_hat].append(
                    evaluate_strategy(q_s, z_c_s, z_e_s, xi_c, xi_e, rho_true,
                                       mis_kc, mis_ke, hits, shortlist_idx)
                )

        out[rho_true] = {
            "oracle": np.array(cell["oracle"]),
            "all_cheap": np.array(cell["all_cheap"]),
            "all_precise": np.array(cell["all_precise"]),
            "misspec": {rh: np.array(v) for rh, v in cell["misspec"].items()},
        }

    return out


def sanity_checks(results):
    print("\n--- sanity checks ---")
    for rho_true in RHO_GRID:
        diag = results[rho_true]["misspec"][rho_true].mean() - results[rho_true]["oracle"].mean()
        print(f"  rho_true={rho_true:.1f}  diag(rho_hat=rho_true) - oracle = {diag:+.4f} (should be ~0)")

    r0 = results[0.0]
    print(f"\n  rho_true=0.0: all_cheap={r0['all_cheap'].mean():.4f}  "
          f"all_precise={r0['all_precise'].mean():.4f}  (all-cheap should win clearly, r={R:.0f} is above the crossing)")


def plot_heatmap(results, path):
    diff = np.zeros((len(RHO_GRID), len(RHO_GRID)))
    for i, rho_true in enumerate(RHO_GRID):
        oracle_mean = results[rho_true]["oracle"].mean()
        for j, rho_hat in enumerate(RHO_GRID):
            mis_mean = results[rho_true]["misspec"][rho_hat].mean()
            diff[i, j] = mis_mean - oracle_mean

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(diff, origin="lower", cmap="Reds_r", vmin=diff.min(), vmax=0.0,
                        extent=[RHO_GRID[0], RHO_GRID[-1], RHO_GRID[0], RHO_GRID[-1]],
                        aspect="auto")
        ax.set_xlabel(r"assumed $\hat\rho$")
        ax.set_ylabel(r"true $\rho$")
        ax.set_title(f"Recall(misspecified) - Recall(oracle), r={R:.0f}")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("recall difference")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        print(f"  saved {path.name}")


def plot_lines(results, path, slices=(0.0, 0.3, 0.6, 1.0)):
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, len(slices), figsize=(4 * len(slices), 4.5), sharey=True)
        cmap = plt.cm.viridis(np.linspace(0.2, 0.85, len(RHO_GRID)))

        for ax, rho_true in zip(axes, slices):
            rho_true = min(RHO_GRID, key=lambda r: abs(r - rho_true))
            meds = [results[rho_true]["misspec"][rh].mean() for rh in RHO_GRID]
            ses = [results[rho_true]["misspec"][rh].std(ddof=1) / np.sqrt(NREPS) for rh in RHO_GRID]
            ax.errorbar(RHO_GRID, meds, yerr=ses, color="#1f77b4", lw=1.5, marker="o", ms=3,
                        label="misspecified split")

            oracle_mean = results[rho_true]["oracle"].mean()
            all_cheap_mean = results[rho_true]["all_cheap"].mean()
            all_precise_mean = results[rho_true]["all_precise"].mean()
            ax.axhline(oracle_mean, color="black", lw=1.2, linestyle="--", label="oracle split")
            ax.axhline(all_cheap_mean, color="#2ca02c", lw=1, linestyle=":", label="all-cheap")
            ax.axhline(all_precise_mean, color="#d62728", lw=1, linestyle=":", label="all-precise")
            ax.axvline(rho_true, color="gray", lw=0.8, alpha=0.5)

            ax.set_title(rf"true $\rho$={rho_true:.1f}")
            ax.set_xlabel(r"assumed $\hat\rho$")

        axes[0].set_ylabel(f"Recall@{L}")
        axes[-1].legend(fontsize=7.5, loc="lower left")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        print(f"  saved {path.name}")


def main():
    print(f"Misspecified-rho experiment: N={N} H={H} M={M} L={L} r={R:.0f} nreps={NREPS}")
    results = run_grid()
    sanity_checks(results)
    plot_heatmap(results, FIGURES / "rho-crossing-heatmap.png")
    plot_lines(results, FIGURES / "rho-crossing-lines.png")


if __name__ == "__main__":
    main()
