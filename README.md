# Feature Drift Detection on Real-World Sensor Data

A lightweight statistical pipeline for detecting feature distribution shift in time-series data, applied to the UCI Air Quality dataset.

---

## Motivation

In production ML systems, models are trained on historical data — but the real world changes. **Feature drift** (covariate shift) occurs when the statistical distribution of model inputs changes over time, causing predictions to silently degrade without any obvious error signal.

This project demonstrates how two complementary statistical methods can be combined in a sliding-window framework to detect such drift automatically, without requiring labelled change-point data.

---

## Dataset

**UCI Air Quality Dataset**  
- 9,358 hourly sensor readings from a multisensor device deployed in an Italian city  
- Date range: March 2004 – April 2005  
- Source: [https://archive.ics.uci.edu/dataset/360/air+quality](https://archive.ics.uci.edu/dataset/360/air+quality)  
- Reference: S. De Vito et al., *Sensors and Actuators B: Chemical*, 2008  

**Feature monitored:** `C6H6(GT)` — ground-truth benzene concentration (μg/m³)

Benzene is highly correlated with CO levels (the prediction target in many air quality models) and exhibits natural **seasonal drift**: higher concentrations in winter due to increased combustion activity and reduced atmospheric mixing, lower in summer. This makes it an ideal real-world test case for a drift detection pipeline.

---

## Approach

### Data Preprocessing
- Date and time columns are combined and parsed into a `datetime` index
- Sentinel missing values (`-200`) are replaced with `NaN` and dropped

### Exploratory Analysis
- 30-day rolling mean plotted alongside monthly concentration averages to visualise the seasonal trend
- Histograms of the first 3 months (Mar–Apr 2004) vs the last 3 months (Jan–Apr 2005) confirm a clear distributional shift between periods

### Drift Detection — Sliding Window Framework

At each test point `t` (sampled every 50 observations, starting after a burn-in period):

| Window | Observations | Duration (approx.) |
|---|---|---|
| **Test window** | `[t - 50, t)` | ~2 days |
| **Baseline window** | `[t - 200, t - 50)` | ~6 days |

Two metrics are computed per window pair:

#### 1. Kolmogorov-Smirnov Two-Sample Test
A non-parametric test of whether the test and baseline windows could plausibly come from the same distribution.

- **H₀:** The two samples are drawn from the same continuous distribution  
- **Alert condition:** p-value < α = 1e-8 (conservative to minimise false positives)  
- **Output:** Binary flag per test point

#### 2. Wasserstein Distance (Earth Mover's Distance)
Measures the minimum "work" required to transform one empirical distribution into the other — providing a continuous, magnitude-aware measure of drift severity.

- **Threshold:** 99th percentile of all computed distances (data-driven, avoids arbitrary constants)  
- **Output:** Continuous score + binary flag above threshold

Both methods are computed in a single loop over the test points to avoid redundant window slicing.

---

## Key Parameters

| Parameter | Value | Rationale |
|---|---|---|
| `intervals` | 50 | ~2 days of hourly data; granular enough to catch seasonal transitions |
| `lookback_to_test_size_ratio` | 3 | Baseline = 3× test window — statistically stable without being too stale |
| `start_point` | 200 | Burn-in: ensures a full baseline window exists before the first test |
| `alpha` | 1e-8 | Conservative KS threshold for large sample sizes |
| `threshold_percentile` | 99 | Top 1% Wasserstein distances flagged as drift |

---

## Results

Both methods independently flagged the same seasonal transition periods — the spring-to-summer drop and the autumn recovery in benzene concentration. This convergence from two different statistical approaches strengthens confidence that alerts reflect genuine distributional change rather than noise.

**Method comparison:**

| | KS Test | Wasserstein Distance |
|---|---|---|
| **Output** | p-value → binary flag | Continuous distance score |
| **Sensitivity** | Mean + shape changes | Full distribution shape |
| **Threshold** | Fixed α = 1e-8 | Data-driven (99th percentile) |
| **Strength** | Strong statistical grounding | Magnitude-aware, interpretable |

---

## Limitations

- **Batch processing only:** The current implementation is post-hoc. Streaming deployment would require an incremental rolling buffer.
- **Single feature:** Only `C6H6(GT)` is monitored. A production system would apply the pipeline per feature and aggregate drift scores.
- **Autocorrelation:** The KS test assumes independent observations; autocorrelated time-series data can inflate Type I error rates.

---

## Requirements

```
pandas
numpy
scipy
matplotlib
seaborn
ucimlrepo
```

Install with:
```bash
pip install pandas numpy scipy matplotlib seaborn ucimlrepo
```

---

## Usage

Run the notebook end-to-end in any Jupyter environment. The dataset is fetched automatically via `ucimlrepo` — no manual download required.

```python
from ucimlrepo import fetch_ucirepo
air_quality = fetch_ucirepo(id=360)
```

---

## File Structure

```
.
├── feature_drift_detection.ipynb   # Main notebook
└── README.md                       # This file
```
