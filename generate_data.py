"""Generate synthetic dose-response data and save it to CSV.

Simulates a sigmoidal (four-parameter logistic) dose-response curve
across 5 drug concentrations with 6 replicates each, plus realistic
multiplicative + additive noise.
"""

import numpy as np
import pandas as pd

RNG_SEED = 42
N_REPLICATES = 6
CONCENTRATIONS = np.array([0.1, 1.0, 10.0, 100.0, 1000.0])  # nM

# "True" four-parameter logistic parameters used to simulate the data
TOP = 100.0       # maximum response (%)
BOTTOM = 0.0       # minimum response (%)
EC50 = 15.0        # concentration producing half-maximal response (nM)
HILL_SLOPE = 1.2   # steepness of the curve


def four_param_logistic(conc, top, bottom, ec50, hill_slope):
    return bottom + (top - bottom) / (1.0 + (ec50 / conc) ** hill_slope)


def generate_dose_response(concentrations, n_replicates, seed=RNG_SEED, noise_multiplier=1.0):
    rng = np.random.default_rng(seed)

    rows = []
    for conc in concentrations:
        true_response = four_param_logistic(conc, TOP, BOTTOM, EC50, HILL_SLOPE)
        for replicate in range(1, n_replicates + 1):
            # Realistic noise: proportional (measurement scales with signal)
            # plus a small fixed baseline noise floor. noise_multiplier scales
            # both, so callers can explore how estimation quality degrades
            # as measurement noise grows.
            proportional_noise = rng.normal(0, noise_multiplier * 0.06 * true_response)
            baseline_noise = rng.normal(0, noise_multiplier * 1.5)
            response = true_response + proportional_noise + baseline_noise
            rows.append(
                {
                    "concentration_nM": conc,
                    "replicate": replicate,
                    "response": response,
                }
            )

    return pd.DataFrame(rows)


def main():
    df = generate_dose_response(CONCENTRATIONS, N_REPLICATES)
    df.to_csv("dose_response_data.csv", index=False)
    print(f"Saved {len(df)} rows to dose_response_data.csv")
    print(df.groupby("concentration_nM")["response"].agg(["mean", "std"]))


if __name__ == "__main__":
    main()
