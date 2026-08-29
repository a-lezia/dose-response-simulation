# Dose-Response Simulation

Simulates a synthetic pharmacology dose-response experiment, fits a
four-parameter logistic (4PL) curve to recover the drug's potency (EC50),
and studies how measurement noise affects the reliability of that estimate.

## What's here

| Script | What it does |
|---|---|
| [`generate_data.py`](generate_data.py) | Generates synthetic dose-response data for 5 concentrations x 6 replicates, with realistic noise, and saves it to `dose_response_data.csv`. |
| [`fit_and_plot.py`](fit_and_plot.py) | Loads the CSV, fits a 4PL curve with `scipy.optimize.curve_fit`, and plots the replicates, mean ± SD, and fitted curve (`dose_response_plot.png`). |
| [`simulate_ec50_bias.py`](simulate_ec50_bias.py) | Repeats the simulate-and-fit pipeline 200 times to check whether the fitted EC50 is biased, and plots a histogram of (fitted - true) EC50 errors. |
| [`noise_sensitivity_curves.py`](noise_sensitivity_curves.py) | Runs the simulate-and-fit pipeline at several noise multipliers (0.5x-4x) and overlays every fitted curve on one plot, showing how the spread of fits widens as noise increases. |
| [`worst_ec50_fit.py`](worst_ec50_fit.py) | Finds the single worst-case trial (largest EC50 error) across all noise levels and plots it against the true curve, marking the EC50 point on each. |

The underlying model is the standard 4-parameter logistic dose-response
equation:

```
response = bottom + (top - bottom) / (1 + (EC50 / concentration) ** hill_slope)
```

`generate_data.py` uses this equation (with fixed "true" parameters) to
generate data, and the other scripts fit the same equation back to noisy
data to see how well the true parameters can be recovered.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

Run in order, from the project root with the virtual environment active:

```bash
python generate_data.py            # creates dose_response_data.csv
python fit_and_plot.py             # fits + plots a single dataset
python simulate_ec50_bias.py       # 200-trial bias/variance check
python noise_sensitivity_curves.py # fitted curves across noise levels
python worst_ec50_fit.py           # worst-case fit vs. true curve
```

Each script prints a short numeric summary to the terminal and saves its
plot as a PNG in the project root. Generated CSV/PNG files are not tracked
in git (see `.gitignore`) since they're reproducible by rerunning the
scripts.
