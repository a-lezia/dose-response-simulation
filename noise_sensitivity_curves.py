"""Explore how measurement noise affects the fitted dose-response curve.

For each of several noise multipliers, run many simulate-and-fit trials
(fresh noisy data -> 4PL fit) and overlay every trial's fitted curve on one
plot as a thin, semi-transparent line. Curves from noisier conditions form a
wider, hazier band; curves from cleaner conditions hug the true curve
tightly. Also reports the bias and spread of the fitted EC50 at each noise
level.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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

NOISE_MULTIPLIERS = [0.5, 1.0, 2.0, 4.0]
TRIALS_PER_LEVEL = 40
COLORS = ["tab:green", "tab:blue", "tab:orange", "tab:red"]
CURVE_ALPHA = 0.15

X_SMOOTH = np.logspace(
    np.log10(CONCENTRATIONS.min()) - 0.5,
    np.log10(CONCENTRATIONS.max()) + 0.5,
    300,
)


def run_noise_level(noise_multiplier, n_trials, base_seed):
    """Fit n_trials fresh noisy datasets at a given noise level.

    Returns the array of fitted EC50s and a list of (top, bottom, ec50,
    hill_slope) parameter tuples, one per successful fit.
    """
    fitted_ec50s = []
    all_params = []
    failed = 0

    for trial in range(n_trials):
        seed = base_seed + trial
        df = generate_dose_response(
            CONCENTRATIONS, N_REPLICATES, seed=seed, noise_multiplier=noise_multiplier
        )
        concentrations = df["concentration_nM"].to_numpy()
        responses = df["response"].to_numpy()

        try:
            params, _ = fit_curve(concentrations, responses)
        except RuntimeError:
            failed += 1
            continue

        all_params.append(params)
        fitted_ec50s.append(params[2])

    return np.array(fitted_ec50s), all_params, failed


def main():
    fig, ax = plt.subplots(figsize=(8, 6))

    legend_handles = []

    for level_idx, (noise_multiplier, color) in enumerate(zip(NOISE_MULTIPLIERS, COLORS)):
        # Distinct, non-overlapping seed ranges so each noise level uses its
        # own independent set of "experiments".
        base_seed = level_idx * 10_000
        fitted_ec50s, all_params, failed = run_noise_level(
            noise_multiplier, TRIALS_PER_LEVEL, base_seed
        )

        for params in all_params:
            y_smooth = four_param_logistic(X_SMOOTH, *params)
            ax.plot(X_SMOOTH, y_smooth, color=color, alpha=CURVE_ALPHA, linewidth=1)

        errors = fitted_ec50s - TRUE_EC50
        bias = errors.mean()
        std = errors.std(ddof=1)
        n_ok = len(fitted_ec50s)

        print(
            f"Noise x{noise_multiplier:<4} | {n_ok}/{TRIALS_PER_LEVEL} fits converged "
            f"| EC50 bias = {bias:+.3f} nM | EC50 std = {std:.3f} nM"
        )

        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                alpha=1.0,
                linewidth=2,
                label=f"noise x{noise_multiplier} (EC50 std={std:.2f} nM)",
            )
        )

    # True underlying curve, drawn on top and bold for reference.
    y_true = four_param_logistic(X_SMOOTH, TOP, BOTTOM, TRUE_EC50, HILL_SLOPE)
    true_line = ax.plot(
        X_SMOOTH, y_true, color="black", linestyle="--", linewidth=2, label="True curve"
    )[0]
    legend_handles.append(true_line)

    ax.set_xscale("log")
    ax.set_xlabel("Concentration (nM)")
    ax.set_ylabel("Response (%)")
    ax.set_title(
        f"Fitted 4PL Curves Across Noise Levels ({TRIALS_PER_LEVEL} trials each)"
    )
    ax.legend(handles=legend_handles, loc="upper left")
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig("noise_sensitivity_curves.png", dpi=150)
    print("Saved plot to noise_sensitivity_curves.png")
    plt.show()


if __name__ == "__main__":
    main()
