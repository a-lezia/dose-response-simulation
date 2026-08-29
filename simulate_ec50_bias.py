"""Repeat the simulate-and-fit pipeline many times to check how well the
4PL fit recovers the true EC50 from noisy replicate data.

Each trial generates a fresh noisy dataset (same true parameters, different
random noise), fits a 4PL curve, and records the fitted EC50. The spread and
mean offset of (fitted - true) across trials show the estimator's precision
and bias.
"""

import numpy as np
import matplotlib.pyplot as plt

from generate_data import generate_dose_response, CONCENTRATIONS, N_REPLICATES, EC50 as TRUE_EC50
from fit_and_plot import fit_curve

N_TRIALS = 200


def run_simulation(n_trials=N_TRIALS):
    fitted_ec50s = []
    failed = 0

    for trial in range(n_trials):
        # A different seed each trial gives fresh noise; the true underlying
        # curve (TOP/BOTTOM/EC50/HILL_SLOPE in generate_data.py) is unchanged.
        df = generate_dose_response(CONCENTRATIONS, N_REPLICATES, seed=trial)
        concentrations = df["concentration_nM"].to_numpy()
        responses = df["response"].to_numpy()

        try:
            params, _ = fit_curve(concentrations, responses)
        except RuntimeError:
            failed += 1
            continue

        _, _, ec50, _ = params
        fitted_ec50s.append(ec50)

    return np.array(fitted_ec50s), failed


def main():
    fitted_ec50s, failed = run_simulation()
    errors = fitted_ec50s - TRUE_EC50

    mean_bias = errors.mean()
    std_error = errors.std(ddof=1)

    print(f"Ran {N_TRIALS} trials ({failed} fits failed to converge)")
    print(f"True EC50:              {TRUE_EC50:.3f} nM")
    print(f"Mean fitted EC50:       {fitted_ec50s.mean():.3f} nM")
    print(f"Mean bias (fit - true): {mean_bias:+.3f} nM")
    print(f"Std dev of error:       {std_error:.3f} nM")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(errors, bins=20, color="tab:blue", edgecolor="black", alpha=0.8)
    ax.axvline(0, color="black", linestyle="--", linewidth=1, label="Zero error")
    ax.axvline(
        mean_bias,
        color="tab:red",
        linewidth=2,
        label=f"Mean bias = {mean_bias:+.3f} nM",
    )
    ax.set_xlabel("Fitted EC50 - True EC50 (nM)")
    ax.set_ylabel("Count")
    ax.set_title(f"EC50 Estimation Error over {len(fitted_ec50s)} Simulated Trials")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig("ec50_error_histogram.png", dpi=150)
    print("Saved histogram to ec50_error_histogram.png")
    plt.show()


if __name__ == "__main__":
    main()
