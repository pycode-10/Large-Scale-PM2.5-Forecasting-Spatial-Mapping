# CS771 Mini-Project 1: Next-Day PM2.5 Forecasting and Mapping

This repository documents our CS771 Mini-Project 1 work on cleaning low-cost sensor data, forecasting next-day PM2.5 across Bihar, and evaluating forecasts with lead-time and spatial-map analyses.

All PM2.5 error metrics are in **µg/m³**. Bihar timestamps are local time (IST). The December 2024 test data are withheld; no December test score is reported here.

## Project question and forecast target

The main task is a day-ahead forecast. At **12:00 IST on day d**, predict PM2.5 at **12:00 IST on day d + 1** for each of 468 Bihar stations. A feature must use readings available at or before the issue time. No external weather, satellite, or other data are used, and neural networks are not used.

The chronological split is by target timestamp:

- **Model-development training targets:** Jan 1-Aug 31, 2024.
- **Validation targets:** Sep 1-Nov 30, 2024, all 91 noon targets at all 468 stations (42,588 station-time forecasts).
- **Final Part 5 refit:** all eligible target dates through Nov 30, 2024. This includes validation-period observations and is for the hidden December evaluation, not for reporting validation results.

The supplied Bihar hourly data were prepared and gap-filled upstream using neighboring stations. The provided values do not identify which readings were filled, so that remains a limitation.

## Main results

The final Part 2 model is a 50/50 blend of a Ridge model and Extra-trees model, both trained to predict the change from today’s noon reading. It has the lowest overall validation RMSE and MAE among the listed models. Its daily RMSE is lower than persistence on 73 of 91 validation days, lower than Part 1 Ridge on 60 days, and lower than Part 2 level Ridge on 55 days. Since validation was used to choose features and models, these comparisons are optimistic and should not be interpreted as a separate test result.

### Day-ahead validation metrics

| Model | Overall RMSE | Overall MAE | Sep RMSE / MAE | Oct RMSE / MAE | Nov RMSE / MAE |
|---|---:|---:|---:|---:|---:|
| Persistence | 32.745 | 17.467 | 23.328 / 8.914 | 20.217 / 11.966 | 47.811 / 31.702 |
| Part 1 Ridge, 29 features | 29.584 | 18.156 | 20.338 / 10.348 | 19.360 / 13.464 | 43.057 / 30.813 |
| Part 2 Ridge, 39 selected features | 29.078 | 16.572 | 19.490 / 8.066 | 17.922 / 11.506 | 43.046 / 30.313 |
| Final 50/50 blend | **28.773** | **16.131** | **19.263 / 7.724** | **17.785 / 11.296** | **42.583 / 29.533** |

### Part 3: validation RMSE / MAE by lead

The target remains 12:00 IST on d + 1. The issue time changes, and learned models are refit for each lead.

| Lead (issue time) | Persistence | Part 1 Ridge | Part 2 level Ridge | Final blend |
|---|---:|---:|---:|---:|
| 1 h (11:00 on d+1) | 23.954 / 10.363 | 18.951 / 8.672 | 18.831 / 8.633 | **18.685 / 7.879** |
| 3 h (09:00 on d+1) | 51.427 / 28.280 | 28.370 / 16.550 | **25.973 / 13.940** | 27.087 / 13.890 |
| 6 h (06:00 on d+1) | 66.825 / 41.280 | 33.725 / 22.800 | **29.831 / 17.600** | 31.139 / 17.810 |
| 12 h (00:00 on d+1) | 72.735 / 46.740 | 31.698 / 21.000 | **29.850 / 18.180** | 30.470 / 17.680 |
| 24 h (12:00 on d) | 32.745 / 17.467 | 29.584 / 18.156 | 29.078 / 16.572 | **28.773 / 16.131** |

Persistence performs poorly at 3-12 hours because those issue times fall in the high nighttime or morning part of the daily pollution cycle, while the target is at noon. At 24 hours, issue and target times match, so much of the daily-cycle difference cancels. The 24-hour lead is the public-warning choice because it gives people time to plan; compared with the 1-hour nowcast, the final model’s RMSE increases by about 10.1 µg/m³.

### Part 4: map evaluation

Station readings were interpolated with inverse-distance weighting (power 2) to the 3,396-point Bihar grid at 0.05° spacing. The same weights were used for observations and forecasts.

| Method | Mean daily station RMSE | Pooled station RMSE | Mean daily map RMSE | Mean daily pattern correlation |
|---|---:|---:|---:|---:|
| Persistence | 28.568 | 32.745 | 15.369 | 0.580 |
| Part 1 Ridge | 26.001 | 29.584 | 15.419 | 0.605 |
| Final Part 2 blend | **25.000** | **28.773** | **14.310** | **0.616** |

Map RMSE is smaller partly because interpolation smooths station-level differences. The observed map is itself an interpolation, not ground truth. Station RMSE is the more direct forecast-accuracy measure.

## What each part does

### Part 0: Kanpur data cleaning

The raw Kanpur file covers 76 sensors from July-August 2026 on a 15-minute UTC grid. The cleaning workflow reshapes readings to an hourly sensor table, flags problems rather than silently deleting them, preserves missing hours, checks sensor agreement, evaluates a July calibration on August data, and summarizes the daily cycle in IST.

- The 10-digit sensors report hourly at :00 with integer values; the 8-character sensors report hourly at :00 with decimal values; one 12-character sensor reports every 15 minutes.
- No sensor moves. The hourly mean changes values only for the 15-minute sensor: mean absolute difference from its :00 reading is 2.61 µg/m³ over 1,212 paired hours.
- Flags include 136 error-code readings (999), 813 exact zeros, 929 stuck readings, 600 otherwise-good readings relabeled for two dropped sensors, and 30,957 missing sensor-hours. The network outage indicator marks 117 hours when fewer than 10% of sensors report.
- The output has 113,088 rows (76 sensors × 1,488 hours). It retains missing rows and does not gap-fill.
- The July-fitted cross-family correction does not generalize to August: August MAE increases from 12.33 to 15.27 µg/m³. No correction is applied.
- The cleaned network daily cycle has a morning peak around 06:30 IST, an afternoon minimum around 15:30, and an evening peak around 19:30.

The Part 0 cleaning notebook and kanpur_hourly_clean.csv are separate artifacts. They are **not present in the supplied IML_Submission directory snapshot**; add them to the repository if they are intended to be part of the GitHub submission.

### Part 1: required baselines

The persistence forecast copies each station’s reading at the issue time. The required Ridge baseline uses 29 raw, unscaled features: 24 station readings from t through t-23 hours; the mean of the 10 nearest other stations at t; the mean for those neighbors over the 24-hour history; the network mean at t; and annual sine/cosine terms. It uses scikit-learn Ridge with alpha=1.0, pooled station-time training examples, and clips negative forecasts to zero.

The hourly package begins at Jan 1, 2024 00:00. A complete 24-hour history is not available for Jan 1 or Jan 2 target forecasts, so the first eligible training target is Jan 3. Training uses 113,256 rows (242 dates × 468 stations); validation uses 42,588 rows.

Ridge has lower RMSE overall and in all three validation months. Persistence has lower overall MAE and lower MAE in September and October; Ridge is slightly better on November MAE. This difference is expected because squared error gives large misses more influence than absolute error.

### Part 2: feature development and final model

The 57 candidate features are organized into eight groups: 24-hour station lags (24); multi-day history (10); nearest-neighbor statistics (3); radius-neighbor statistics at 15, 30, and 60 km (6); network summaries (4); trends (3); seasonal and training-only station climatology features (5); and day-of-week terms (2).

Forward selection retained 39 Ridge features: station 24-hour lags, network summaries, radius-neighbor summaries, trends, and day-of-week terms. Validation RMSE improved from 29.863 with lags alone to 29.078 with the selected groups. Removing the station-lag group increased RMSE by 7.033; removing the other retained groups increased RMSE by 0.044-0.209 each. The groups overlap in information, especially spatial and network means.

Tree models trained to predict the PM2.5 level fit the training period better but validated worse than Ridge, indicating overfitting and difficulty extrapolating into the autumn pollution increase. The final model instead predicts tomorrow’s change from today:

1. Standardized Ridge on the selected 39 features, alpha=316.
2. Extra-trees on all 57 features, five ensembles of 60 trees, leaf size 10.
3. Equal 0.5 / 0.5 blend of the two change forecasts, added to today’s reading and clipped at zero.

The 0.5 blend weight was retained because validation scores were relatively flat around that value. The final model’s November bias is -4.60 µg/m³, so it slightly under-forecasts the late-season rise.

### Part 5: final refit and forecast interface

The final model was refit through target date Nov 30, using 327 target dates from Jan 9-Nov 30 and 153,036 station-time rows. The station climatology values were recomputed on these refit rows.

forecast.py defines a function with this interface:

~~~python
forecast(history: pandas.DataFrame, issue_time: pandas.Timestamp) -> pandas.Series
~~~

It returns a nonnegative forecast for every station, indexed by string sensor IDs, for issue_time + 24 hours. It loads the fitted model and station coordinates from files alongside the script. It truncates history at issue_time, builds features, and predicts without refitting model parameters. If history contains gaps, the implementation reindexes to hourly timestamps and fills from the available history.

**Verification status in the attached notebook:** the notebook confirms that forecast.py matches the refitted model on two issue times, returns the same output when later rows are appended, and runs in 0.06 seconds at its slowest checked call. The official starter/check_submission.py was not found in that Colab session, so this notebook does not establish an official checker pass. Run the course-provided checker before submission. The checker’s November result is a sanity check, not a December test score; after the final refit, November is in the training data.

## Repository contents

The supplied IML_Submission directory contains:

| File or folder | Purpose |
|---|---|
| IML_Part_1_to_5.ipynb | Reproducible analysis, experiments, tables, figures, model refit, and submission generation |
| forecast.py | Part 5 forecasting entry point |
| pm25_features.py | Shared feature and station-neighborhood calculations |
| model.pkl | Fitted Ridge and Extra-trees model components and configuration (about 67.6 MB) |
| bihar_stations.csv | Station IDs and coordinates required by the forecaster |
| report.pdf | Six-page report covering Parts 0-5 |
| outputs/ | Exported validation metrics, ablation results, lead-time and map metrics, and figures |

The notebook also creates a submission/ folder containing forecast.py, pm25_features.py, model.pkl, and bihar_stations.csv.

### Exported outputs

| Output | Contents |
|---|---|
| outputs/part1_validation_metrics.csv | Persistence and Part 1 Ridge RMSE/MAE, overall and by month |
| outputs/part1_validation_forecasts.csv | Observed values and both baseline predictions for each validation station-time row |
| outputs/part2_ablation.csv | Validation RMSE after removing each retained feature group |
| outputs/part2_validation_metrics.csv | Part 2 selected-feature Ridge validation metrics |
| outputs/part2_blend_validation_metrics.csv | Validation metrics for change-target Ridge, Extra-trees, and their blend |
| outputs/part3_lead_time.csv | RMSE and MAE for each method at 1, 3, 6, 12, and 24 hour leads |
| outputs/part4_daily_map_metrics.csv | Per-day station/map RMSE and pattern correlations |
| outputs/part4_map_summary.csv | Average daily and pooled map/station metrics by method |
| outputs/figures/ | Lead-time plot, daily map-metric plot, and selected-day forecast maps |

The combined notebook requires these assignment inputs to reproduce Parts 1-5:

- bihar_pm25_hourly_jan_nov_2024.csv
- bihar_stations.csv
- bihar_grid_005deg.csv
- bihar_boundary.csv

The complete assignment dataset is not included in the supplied IML_Submission directory snapshot. The submission does include bihar_stations.csv for Part 5. To rerun Parts 1-4, place the hourly Bihar table, station coordinates, grid, and boundary CSVs together in /content, /content/data, a data/ directory next to the notebook, or the notebook’s working directory; the notebook checks those locations. Part 0 additionally needs the raw Kanpur CSV and its separate cleaning notebook.

## Run the analysis in Google Colab

1. Upload the Bihar input CSVs listed above into /content (or /content/data).
2. Open IML_Part_1_to_5.ipynb in Colab and run all cells from top to bottom.
3. The notebook writes result tables and figures under outputs/, and writes the Part 5 package under submission/.
4. Download the generated files from Colab and retain them with the notebook.

The notebook uses Python with NumPy, pandas, Matplotlib, scikit-learn, SciPy, and joblib. In Colab, if a dependency is missing, install it with:

~~~python
!pip install numpy pandas matplotlib scikit-learn scipy joblib
~~~

## Run the Part 5 checker

The official checker is supplied in the assignment package under starter/check_submission.py. With the original package layout intact, run:

~~~bash
python starter/check_submission.py path/to/IML_Submission
~~~

The package’s data/ folder must contain bihar_pm25_hourly_jan_nov_2024.csv, because the checker reads its November validation examples relative to the checker script. The submission directory passed to the checker must contain forecast.py, pm25_features.py, model.pkl, and bihar_stations.csv. Each call must finish within 10 seconds and return a finite value for every station.

The November RMSE printed by the checker is not a hidden-test result. The hidden evaluation uses December issue times and withheld December targets.

## Limitations and interpretation

- Validation was used to select features, models, and hyperparameters; reported validation scores are likely optimistic.
- The validation window has only 91 days and November contributes the largest errors.
- Bihar readings were gap-filled upstream without flags.
- No meteorological or emissions data are available.
- The final blend is tuned for the 24-hour forecast; Part 3 shows the Part 2 Ridge is better at 3-12 hours.
- Map metrics compare interpolated fields, not independent ground truth.
- The final model is pooled across stations and the fitted model file is large.


