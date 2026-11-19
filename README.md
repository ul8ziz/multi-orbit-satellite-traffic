# Statistical Analysis and Forecasting of Network Traffic Profiles for Dynamic Resource Management in Multi-Orbital Satellite Access

Research codebase for a thesis on traffic profiling, forecasting, and queueing analysis of LEO/MEO satellite access using the Leontyeva retrial-queue CTMC model.

## Dataset

Real mobile traffic dataset from a Portuguese operator (provided by Prof. Luis M. Correia, IST/INESC-ID).

| Property | Value |
|----------|-------|
| Raw records | 44,009,843 |
| Cleaned records | 36,862,187 |
| Unique subscribers | 94,871 |
| Application classes | 16 |
| Time period | 2018-02-10 to 2018-03-04 (23 days) |
| Granularity | 1 hour |
| Columns | `START_HOUR`, `MASKED_MSISDN`, `APP_CLASS`, `UPLOAD`, `DOWNLOAD` |

The raw dataset is not included in this repository. Scripts that read it expect `data/Applications_data_sample.csv`; the computed tables and figures in `outputs/` are committed, so the results can be reviewed without the data. The three-orbit model (`run_multi_orbit.py`) and the tests run without the dataset.

## 3GPP Slice Mapping

Traffic is classified into three 5G/6G network slices:

| Slice | Applications | Records |
|-------|-------------|---------|
| **eMBB** (Enhanced Mobile Broadband) | Web Applications, Streaming, File Transfer, Music Streaming, Mail, P2P | ~14.9M |
| **URLLC** (Ultra-Reliable Low Latency) | Instant Messaging, Games, VoIP, DB Transactions | ~12.8M |
| **mMTC** (Massive IoT) | Others, Security, Terminals, Network Operation, File Systems, Legacy Protocols | ~9.2M |

## Pipeline Stages

### Stage A — Traffic Data Analysis

Chunked two-pass processing of the raw CSV (~3.6 GB) without loading it entirely into RAM:

1. **Pass 1:** Collect all `total_mb` values via memmap for Tukey IQR outlier bounds (factor = 3.0)
2. **Pass 2:** Filter, add `slice_id`, write cleaned Parquet + partial aggregates
3. **Post-processing:** Hourly/slice aggregation, user profiling (Heavy/Medium/Light), synthetic spatial placement (500x500 km grid), demand matrices, statistical profiling

**Key outputs:**
- `data/processed/cleaned_traffic.parquet` — cleaned dataset
- `outputs/tables/artifact1_slice_aggregated.csv` — hourly traffic by slice
- `outputs/tables/daily_profile.csv` — average daily pattern per slice
- `outputs/tables/burstiness_papr.csv` — Peak-to-Average Power Ratio per slice
- `outputs/tables/slice_correlation_matrix.csv` — inter-slice correlation
- `outputs/tables/demand_matrix_long.csv` — spatial demand matrix
- `outputs/figures/` — time series, heatmaps, histograms, correlation plots

### Forecasting

Per-slice hourly time series forecasting:

- **SARIMAX** with seasonal period = 24h (baseline)
- **LSTM** (PyTorch) or **MLP** (sklearn fallback) with MinMax scaling and sliding window

Metrics: RMSE, MAE, spike-weighted RMSE/MAE.

**Horizon:** `--horizon one_step` (default) predicts each test hour from all observations
before it (Ŷ_{t+1|t}, as in the article); `--horizon multi_step` makes one forecast of the
whole test segment from the end of training (the earlier behaviour).

**Missing hours.** The dataset has no records from 2018-02-22 14:00 to 2018-03-01 15:00.
These hours are kept as `NaN` (earlier versions filled them with 0, which made the models learn
and be scored on a fake 7-day outage):

- the train/test split takes the last 20 % of **observed** hours as test (77 hours, all after the gap);
- SARIMAX skips `NaN` in the Kalman filter;
- LSTM/MLP trains only on fully observed 24-hour windows; missing *input* hours are filled with
  the training hour-of-day mean (never used as targets or in metrics);
- RMSE/MAE/spike metrics use observed hours only (`n_test` = observed test hours).

The earlier zero-filled results are not kept in the repository.

**Key outputs:**
- `outputs/tables/forecast_model_comparison.csv`
- `outputs/figures/forecast_vs_actual_{slice}.png`

### Stage B — Leontyeva LEO/MEO CTMC

Retrial queueing system per Leontyeva (ШТМО 2026). State space `(n, m)` where `n` = active sessions (0..N), `m` = requests in retrial orbit (0..M).

**Generator matrix Q transitions:**

```
Q[(n,m), (n+1,m)] = λ           n < N        (arrival served)
Q[(n,m), (n,m+1)] = λ           n = N, m < M (arrival → orbit)
Q[(n,m), (n-1,m)] = μ(n)        n > 0        (service completion)
Q[(n,m), (n+1,m-1)] = m·γ       n < N, m > 0 (retrial from orbit)
```

Service rate: `μ(n) = min(n, C_LEO)·μ_LEO + max(0, n − C_LEO)·μ_MEO`

**Stationary distribution:** `πQ = 0`, `Σπ = 1`

**QoS metrics (per Leontyeva paper):**

| Metric | Description |
|--------|-------------|
| `P_loss = π(N, M)` | Loss probability (all channels AND orbit full) |
| `P(n=N)` | Probability all channels busy |
| `E[n]`, `E[m]` | Expected active sessions / orbit queue length |
| `util_LEO`, `util_MEO` | Channel utilization per orbital level |

### Integration

Maps empirical traffic to CTMC arrival rate:

```
λ_t = α · total_mb_t / 3600
```

Two scenarios per run: **mean-load** (`λ = mean(λ_t)`) and **peak-load** (`λ = max(λ_t)`).

Calibration grid search over `α, N, M, C_LEO, μ_LEO, μ_MEO, γ` to find parameter combinations yielding interpretable blocking and utilization levels.

### Stage C — Three-orbit LEO/MEO/GEO extension (`src/markov/multi_orbit.py`)

Extends the Leontyeva model with a third orbit (GEO) for 3GPP TR 22.887 use case 5.20
(multi-orbit connectivity to a mobile base station relay). Two formulations:

1. **Aggregated** — Leontyeva state `(n, m)` unchanged; K-segment service rate
   `μ(n) = Σ_j min(max(0, n − C_1 − … − C_{j−1}), C_j)·μ_j`. For K = 2 it is identical to
   Stage B (tested against the article's Table 6). It implicitly re-packs sessions onto the
   faster orbit whenever a channel frees up.
2. **Exact, multi-class** — state `(k_LEO, k_MEO, k_GEO, m_1, …, m_C)`: busy channels per
   orbit plus one retrial orbit per traffic class. Each class has its own arrival rate and
   orbit traversal order, so the model covers both single-class order comparison
   (LEO→MEO→GEO vs MEO→LEO→GEO …) and the service-based routing of §5.20
   (URLLC: LEO→MEO, eMBB: MEO→LEO→GEO, mMTC: GEO→MEO).

Metrics per class: `P_block`, `P_loss`, `E[m]`, retrial time `W = E[m]/λ_acc` (Little),
orbit shares of admitted sessions and mean round-trip propagation delay; per orbit: utilisation.

With GEO present, the aggregated formula noticeably misstates the load split (default
parameters, λ = 6: util_GEO 0.026 aggregated vs 0.154 exact; P_block 0.0124 vs 0.0175),
so the exact model should be used for three orbits.

Outputs (`outputs/multi_orbit/`): `order_comparison.csv/.png`, `scenario_5_20.csv/.png`,
`aggregated_vs_exact.csv`. Orbit delays (30/130/550 ms RTT) are illustrative defaults.

### Journal article figures (`src/visualization/article_figures.py`)

`python run_article_figures.py` regenerates Figures 1–12 and Table 3 of
`docs/Dabwan_2026_Journal_Article.docx` into `outputs/article_figures/`
(`--only 5 11 12` for a subset, `--lang ru` for Fig. 4, `--skip-records` to skip the parquet sample).

| Fig. | File | Source |
|------|------|--------|
| 1 | `fig01_daily_profile.png` | `daily_profile.csv` |
| 2, 3 | `fig02_hourly_pdf.png`, `fig03_hourly_cdf.png` (as printed: hour-of-day totals, Normal fit) | `artifact1_slice_aggregated.csv` |
| 2b, 3b, Table 3 | `fig02b_record_pdf.png`, `fig03b_record_cdf.png`, `table3_distribution_fits.csv` (per-record Normal/LogNormal/Gamma + KS, 1 % sample) | `cleaned_traffic.parquet` |
| 4 | `fig04_pipeline.png` | — |
| 5 | `fig05_service_rate.png` | `default_params()` |
| 6 | `fig06_state_diagram.png` (N=3, M=2) | — |
| 7 | `fig07_forecast_vs_actual_eMBB.png` (zero-filled hours shaded) | `forecast_predictions_*.csv` |
| 8–10 | `fig08…fig10_error_hist_*.png` (`--drop-missing-hours` excludes zero-filled hours) | `forecast_predictions_*.csv` |
| 11 | `fig11_sensitivity.png/.csv` (P_loss < 1e-14 hidden as solver round-off) | CTMC |
| 12 | `fig12_quasi_stationary.png/.csv` (α so that mean λ = 3.0; `--alpha` to override) | `daily_profile.csv` + CTMC |

Data gap: the hourly series have no records from 2018-02-22 14:00 to 2018-03-01 15:00
(170 of 551 hours; 381 hours are observed). See **Forecasting → Missing hours** above.

## Project Layout

```
project/
├── data/                        # Applications_data_sample.csv (~3.6 GB, not in the repo)
├── data/processed/              # cleaned Parquet, hourly aggregates, metrics JSON
├── docs/                        # thesis documents, Leontyeva paper (not in the repo)
├── outputs/
│   ├── tables/                  # CSV tables (slices, stats, forecasts, demand)
│   ├── figures/                 # PNG plots
│   ├── reports/                 # text summaries, DOCX reports
│   ├── markov/                  # stationary distribution, metrics
│   ├── multi_orbit/             # LEO/MEO/GEO order comparison, scenario 5.20
│   ├── article_figures/         # journal article Fig. 1–12, Table 3
│   └── integration/             # scenario metrics, calibration grid, plots
├── src/
│   ├── preprocessing/           # chunked cleaning, loaders, dataset path
│   ├── analysis/                # slice mapping, statistics, spatial, users
│   ├── forecasting/             # SARIMAX, LSTM/MLP, timeseries
│   ├── markov/                  # Leontyeva params, CTMC generator, stationary solver,
│   │                            #   multi_orbit.py (LEO/MEO/GEO, multi-class)
│   ├── integration/             # lambda derivation, scenarios, calibration
│   ├── visualization/           # stage A and forecast figures
│   ├── metrics/                 # forecast accuracy helpers
│   └── reports/                 # DOCX content (Russian academic text)
├── tests/                       # pytest suite
├── run_stage_a.py               # Stage A entry point
├── run_forecasting.py           # Forecasting entry point
├── run_stage_b.py               # Leontyeva CTMC entry point
├── run_integration.py           # Integration entry point
├── run_pipeline.py              # Full pipeline (A → forecast → B → integration)
├── run_calibration.py           # Calibration grid search
├── run_multi_orbit.py           # Three-orbit model entry point
├── run_article_figures.py       # Journal article Figures 1–12 + Table 3
├── build_final_report.py        # Academic DOCX report builder
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

Optional: install `torch>=2.0` for LSTM forecasting (falls back to sklearn MLP if missing).

## Running

```bash
python run_stage_a.py           # Stage A — traffic analysis (~5 min on 44M rows)
python run_forecasting.py       # Per-slice SARIMAX + LSTM/MLP forecasting
python run_stage_b.py           # Leontyeva CTMC with default parameters
python run_integration.py       # λ from data → mean/peak CTMC scenarios
python run_pipeline.py          # All stages sequentially + DOCX report
python run_calibration.py       # Full α/capacity/service calibration grid
python run_multi_orbit.py       # LEO/MEO/GEO: traversal orders + TR 22.887 §5.20 routing
python run_article_figures.py    # Figures 1–12 + Table 3 of the journal article
```

## Tests

```bash
pytest tests/
```

26 tests covering: missing-hour handling in forecasting (`tests/test_forecast_gaps.py`), article figures on synthetic data (`tests/test_article_figures.py`), package imports, slice mapping, Stage A pipeline, forecast metrics, integration scenarios, Leontyeva CTMC verification (`|πQ| < 10⁻⁶`, `P_loss ≤ P(n=N)`), and the three-orbit model (`tests/test_multi_orbit.py`: K = 2 equals Stage B and reproduces Table 6, exact = aggregated for one orbit, §5.20 routing).
