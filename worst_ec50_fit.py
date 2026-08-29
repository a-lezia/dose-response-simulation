"""Find the single simulated trial (across all noise levels) whose fitted
EC50 deviated the most from the true EC50, then plot that trial's noisy
data alongside the true curve and its fitted curve, marking the EC50 point
on each curve.
"""

import numpy as np
import matplotlib.pyplot as plt

from generate_data import (
    generate_dose_response,
    four_param_logistic,
    CONCENTRATIONS,
    N_REPLICATES,
    TOP,
    BOTTOM,
    EC50 as TRUE_EC50,
    HILL_SLOPE,
)
from fit_and_plot import fit_curve
from noise_sensitivity_curves import NOISE_MULTIPLIERS, TRIALS_PER_LEVEL, X_SMOOTH


def find_worst_trial():
    """Re-run every (noise level, trial) combination and keep the one whose
    fitted EC50 was furthest from the true EC50."""
    worst = None

    for level_idx, noise_multiplier in enumerate(NOISE_MULTIPLIERS):
        base_seed = level_idx * 10_000
        for trial in range(TRIALS_PER_LEVEL):
            seed = base_seed + trial
            df = generate_dose_response(
                CONCENTRATIONS, N_REPLICATES, seed=seed, noise_multiplier=noise_multiplier
            )
            concentrations = df["concentration_nM"].to_numpy()
            responses = df["response"].to_numpy()

            try:
                params, _ = fit_curve(concentrations, responses)
            except RuntimeError:
                continue

            ec50_error = params[2] - TRUE_EC50
            if worst is None or abs(ec50_error) > abs(worst["ec50_error"]):
                worst = {
                    "noise_multiplier": noise_multiplier,
                    "seed": seed,
                    "params": params,
                    "ec50_error": ec50_error,
                    "concentrations": concentrations,
                    "responses": responses,
                }

    return worst


def main():
    worst = find_worst_trial()

    top, bottom, ec50, hill_slope = worst["params"]
    print(
        f"Worst trial: noise x{worst['noise_multiplier']}, seed={worst['seed']}\n"
        f"  Fitted EC50 = {ec50:.3f} nM vs True EC50 = {TRUE_EC50:.3f} nM "
        f"(error = {worst['ec50_error']:+.3f} nM)"
    )

    y_true = four_param_logistic(X_SMOOTH, TOP, BOTTOM, TRUE_EC50, HILL_SLOPE)
    y_fit = four_param_logistic(X_SMOOTH, *worst["params"])

    # The response at x = EC50 is always the curve's midpoint, (top+bottom)/2,
    # by definition of EC50 in the 4PL equation.
    true_ec50_response = (TOP + BOTTOM) / 2.0
    fit_ec50_response = (top + bottom) / 2.0

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(
        worst["concentrations"],
        worst["responses"],
        color="tab:gray",
        alpha=0.7,
        label=f"Noisy data (noise x{worst['noise_multiplier']})",
        zorder=3,
    )

    ax.plot(X_SMOOTH, y_true, color="black", linestyle="--", linewidth=2, label="True curve")
    ax.plot(X_SMOOTH, y_fit, color="tab:red", linewidth=2, label="Worst fitted curve")

    ax.scatter(
        [TRUE_EC50],
        [true_ec50_response],
        color="black",
        marker="o",
        s=110,
        zorder=5,
        label=f"True EC50 = {TRUE_EC50:.2f} nM",
    )
    ax.scatter(
        [ec50],
        [fit_ec50_response],
        color="tab:red",
        marker="X",
        s=150,
        zorder=5,
        label=f"Fitted EC50 = {ec50:.2f} nM",
    )

    ax.set_xscale("log")
    ax.set_xlabel("Concentration (nM)")
    ax.set_ylabel("Response (%)")
    ax.set_title(
        f"Worst-Case EC50 Fit vs. True Curve (error = {worst['ec50_error']:+.2f} nM)"
    )
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig("worst_ec50_fit.png", dpi=150)
    print("Saved plot to worst_ec50_fit.png")
    plt.show()


if __name__ == "__main__":
    main()
