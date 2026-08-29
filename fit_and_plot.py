"""Load dose-response CSV, fit a 4-parameter logistic curve, and plot it."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

CSV_PATH = "dose_response_data.csv"


def four_param_logistic(conc, top, bottom, ec50, hill_slope):
    return bottom + (top - bottom) / (1.0 + (ec50 / conc) ** hill_slope)


def fit_curve(concentrations, responses):
    # Reasonable initial guesses based on the data itself.
    p0 = [responses.max(), responses.min(), np.median(concentrations), 1.0]
    bounds = ([-np.inf, -np.inf, 0, 0], [np.inf, np.inf, np.inf, np.inf])
    params, covariance = curve_fit(
        four_param_logistic, concentrations, responses, p0=p0, bounds=bounds
    )
    return params, covariance


def main():
    df = pd.read_csv(CSV_PATH)

    concentrations = df["concentration_nM"].to_numpy()
    responses = df["response"].to_numpy()

    params, covariance = fit_curve(concentrations, responses)
    top, bottom, ec50, hill_slope = params
    param_errors = np.sqrt(np.diag(covariance))

    print("Fitted 4-parameter logistic curve:")
    print(f"  Top        = {top:.3f} +/- {param_errors[0]:.3f}")
    print(f"  Bottom     = {bottom:.3f} +/- {param_errors[1]:.3f}")
    print(f"  EC50 (nM)  = {ec50:.3f} +/- {param_errors[2]:.3f}")
    print(f"  Hill slope = {hill_slope:.3f} +/- {param_errors[3]:.3f}")

    # Summary stats per concentration for error bars.
    summary = df.groupby("concentration_nM")["response"].agg(["mean", "std"]).reset_index()

    # Smooth curve for the fitted line, on a log-spaced x-axis.
    x_smooth = np.logspace(
        np.log10(concentrations.min()) - 0.5,
        np.log10(concentrations.max()) + 0.5,
        300,
    )
    y_smooth = four_param_logistic(x_smooth, *params)

    fig, ax = plt.subplots(figsize=(7, 5))

    # Raw replicate points.
    ax.scatter(
        concentrations, responses, alpha=0.5, color="tab:blue", label="Replicates"
    )
    # Mean +/- std at each concentration.
    ax.errorbar(
        summary["concentration_nM"],
        summary["mean"],
        yerr=summary["std"],
        fmt="o",
        color="tab:orange",
        capsize=4,
        label="Mean ± SD",
    )
    # Fitted curve.
    ax.plot(x_smooth, y_smooth, color="tab:red", label="4PL fit")

    ax.set_xscale("log")
    ax.set_xlabel("Concentration (nM)")
    ax.set_ylabel("Response (%)")
    ax.set_title("Dose-Response Curve")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig("dose_response_plot.png", dpi=150)
    print("Saved plot to dose_response_plot.png")
    plt.show()


if __name__ == "__main__":
    main()
