# Multi-Scale Temporal Feature Design for 48-Hour Soil Moisture Forecasting Using ISMN Station Data

## 1. Abstract

Soil moisture forecasting is useful for agriculture, irrigation planning, drought monitoring, and environmental studies. Many existing studies focus mainly on comparing different machine learning or deep learning models, but they do not always study an equally important question: which parts of past history are actually most useful for future prediction. This project focuses on that question for 48-hour-ahead soil moisture forecasting using ISMN station data. The main idea is not to propose a new complex architecture, but to design a structured multi-scale temporal feature framework that uses short-term, medium-term, and weekly historical information in a more meaningful way.

To support this goal, the data pipeline was rebuilt from scratch. The station data was expanded onto a true hourly grid so that lag features and prediction targets represent real time gaps instead of simple row offsets. After that, a forecasting dataset was prepared using current environmental variables, calendar signals, lag features, and rolling summary statistics. The main novelty experiment was an XGBoost-based lag ablation study. The best saved ablation result came from the `full_multiscale` feature group with an RMSE of `0.029122`, compared with `0.029588` for the simpler `baseline_limited` setup. This shows a modest but consistent improvement of about `1.57%`.

Additional benchmark experiments were also carried out using CatBoost, LightGBM, KNN, Prophet, ARIMA, LSTM, GRU, and stacking. On the strict common-subset benchmark among selected strong models, XGBoost remained the best final predictor. Overall, the main contribution of this project is a simple and defensible study showing that structured multi-scale temporal context can improve 48-hour soil moisture forecasting for a strong tree-based model.

## 2. Keywords

- Soil moisture forecasting
- Time-series prediction
- Multi-scale temporal features
- Lag ablation
- XGBoost
- ISMN

## 3. Introduction

Soil moisture plays an important role in agriculture, crop health, drought analysis, irrigation scheduling, and environmental monitoring. It is affected by rainfall, temperature, evaporation, and the previous state of the soil. Because of this, forecasting future soil moisture can help both practical decision-making and scientific understanding. For medium-horizon forecasting such as 48 hours ahead, it is important to capture not only current conditions but also useful information from the past.

Many existing forecasting studies compare different models such as XGBoost, LSTM, GRU, Transformer, Random Forest, or hybrid systems. That work is useful, but it often puts more attention on model competition than on historical feature design. In many cases, lag features are chosen in a fixed way and then the models are compared without carefully checking which temporal history scales are actually helping. This creates a limitation, because good performance may depend not only on model choice, but also on how the past is represented.

In this project, the research direction is different. Instead of trying to build a new model architecture, the work studies whether a structured temporal feature design can improve 48-hour soil moisture forecasting. The methodology uses a rebuilt ISMN forecasting pipeline, a multi-scale lag framework, and an ablation study with XGBoost as the anchor model. Supporting benchmarks are then used to understand how this temporal design behaves across different model families.

### 3.1 Problem Statement

The main problem in this project is that soil moisture forecasting models are often compared without clearly analyzing which parts of past history are actually useful for prediction. For a 48-hour forecasting task, it is not enough to say that "more history" is better. We need to know whether short-term, medium-term, and weekly memory contribute differently, and whether combining them in a structured way improves prediction.

### 3.2 Research Gap

The gap addressed in this study is that many forecasting works focus mainly on model selection, while giving less attention to the design of temporal context itself. In soil moisture forecasting, it is reasonable to expect that different time scales matter in different ways. Very recent history may capture immediate persistence, medium-range history may reflect slower changes, and weekly memory may capture delayed hydrological behavior. However, these temporal roles are not always isolated and tested clearly.

### 3.3 Objectives and Contributions

The objectives of this project are:

1. Build a clean and methodologically safe hourly forecasting dataset from ISMN station data.
2. Design a multi-scale temporal feature framework for 48-hour forecasting.
3. Test the contribution of different lag groups through ablation.
4. Compare several benchmark model families as supporting evidence.
5. Check whether stacking gives additional value beyond strong single models.

The main contribution of the project is a structured and defensible lag-design study showing that multi-scale temporal context improves prediction for a strong tree-based model, even though the improvement is moderate.

## 4. Literature Review (Related Work)

This section is intentionally left as a placeholder for now.

What still needs to be added here:

- discussion of existing soil moisture forecasting studies
- short critical analysis of previous machine learning and deep learning work
- identification of limitations in earlier work
- positioning of this study compared with those works
- a short and concise comparison table

### Figure/Table Placeholder

- `[PLACEHOLDER: Literature review comparison table]`

## 5. Methodology (Proposed Approach)

### 5.1 Overall Framework

The proposed approach is centered on multi-scale temporal feature design for 48-hour soil moisture forecasting. The main idea is to prepare a clean hourly dataset, build structured lag and rolling features, and then test whether these temporal groups improve forecasting performance. XGBoost is used as the main ablation anchor because it is strong, stable, and easy to interpret when explicit lag features are used.

The full workflow of the study can be summarized as follows:

1. Read and clean raw ISMN station data.
2. Expand each station to a true hourly grid.
3. Create future targets for forecasting horizons.
4. Build current, calendar, lag, and rolling features.
5. Train and evaluate XGBoost for ablation.
6. Run benchmark models for supporting comparison.
7. Analyze results using descriptive and common-subset benchmark tables.

### 5.2 Data Pipeline Repair

One important issue found during the project was that direct row shifting is unsafe if the station time series contains gaps. In that case, "48 rows later" may not mean "48 hours later". This can produce wrong target definitions and misleading lag features.

To fix this, each station was expanded onto a true hourly timeline before forecasting targets and lag features were created. This means every hourly timestamp between the station start and end time exists explicitly in the data. Missing hours remain visible as gaps, and lag features such as `24h`, `48h`, and `168h` now represent real clock-hour differences.

### 5.3 Multi-Scale Temporal Feature Design

The main proposed feature framework uses several groups of predictors:

- current-state variables
- calendar and seasonal signals
- short-term lag features
- medium-term lag features
- weekly lag features
- rolling summary features

The lag design is organized into meaningful temporal groups:

- short-term context: `1, 3, 6, 12, 24` hours
- medium-term context: `48, 72` hours
- weekly context: `168` hours

This structure is more meaningful than simply adding many lag columns. The aim is to test whether different historical scales contribute differently to forecasting quality.

### 5.4 Ablation Design

The current saved XGBoost ablation artifact set includes these main feature groups:

- `baseline_limited`
- `short_lags`
- `short_plus_medium`
- `full_multiscale`

The main comparison is between the simpler baseline and the richer multi-scale setup. The project work log also contains exploratory notes from additional refined groups, but the report should treat those as supportive interpretation rather than primary saved artifacts.

### 5.5 Benchmark Models

To support the main novelty study, the project also evaluates:

- XGBoost
- CatBoost
- LightGBM
- KNN
- Prophet
- ARIMA
- LSTM
- GRU
- stacking

These models are used to provide context, not to replace the main lag-ablation contribution.

### 5.6 Mathematical Formulation

At a simple level, the forecasting task can be written as:

`y(t + h) = f(X_t)`

where:

- `y(t + h)` is the future soil moisture value at forecasting horizon `h`
- `X_t` is the feature vector available at time `t`
- `f(.)` is the model being used

In this study, the main horizon is:

- `h = 48 hours`

The feature vector includes:

- current environmental variables
- calendar variables
- lagged variables from multiple time scales
- rolling statistics

## 6. Experimental Setup / Data Description

### 6.1 Dataset Used

This project uses ISMN station data already available in the workspace. The current station scope includes:

- Gevenich
- Merzenhausen
- Selhausen

The main target variable is:

- `sm_0.05m`

Although datasets were also prepared for `24h` and `72h`, the main analysis in this study is based on:

- `ismn_forecasting_48h_ready.csv`

### 6.2 Preprocessing Steps

The main preprocessing steps were:

1. Read raw `.stm` station files.
2. Keep only observations with quality flag `G`.
3. Aggregate duplicate observations using the median.
4. Merge station data into a common hourly dataset.
5. Expand each station onto a true hourly grid.
6. Create forecasting targets for `24h`, `48h`, and `72h`.
7. Create current, calendar, lag, and rolling features.
8. Prepare model-ready datasets after filtering incomplete rows where required.

### 6.3 Tools and Environment

The implementation was carried out in Python. The main tools and libraries used in this project include:

- Python
- pandas
- NumPy
- XGBoost
- CatBoost
- LightGBM
- TensorFlow / Keras
- scikit-learn

The work was developed and tested in the active local project environment.

### 6.4 Parameter Settings

The exact parameter settings varied across models, but the main study used:

- XGBoost as the main ablation anchor
- repaired LSTM with contiguous hourly sequences
- repaired GRU with the same sequence pipeline
- repeated-seed evaluation for the major benchmark models

For the sequence models, the repaired main benchmark setup used:

- lookback: `168`
- forecasting horizon: `48h`
- repeated seeds: `42, 52, 62`

### Figure/Table Placeholders

- `[PLACEHOLDER: Dataset summary table]`
- `[PLACEHOLDER: Preprocessing workflow figure]`
- `[PLACEHOLDER: Feature-group summary table]`

## 7. Results and Discussion

### 7.1 Performance Metrics

The main metrics used in this project are:

- RMSE
- MAE
- R2

RMSE is especially important in this report because it is the main metric used to compare forecasting accuracy across the saved experiments.

### 7.2 Descriptive Benchmark Results

Because all models do not share the same effective test coverage, the project keeps a descriptive benchmark table for broad comparison. This table is useful for context, but it is not a strict apples-to-apples ranking.

Important descriptive benchmark results include:

- XGBoost `full_multiscale`: `RMSE 0.028472`
- CatBoost `baseline_limited__confirm`: `RMSE 0.028641`
- LightGBM `full_multiscale__full`: `RMSE 0.029128`
- KNN `baseline_limited__light`: `RMSE 0.036659`
- LSTM `lstm__repaired40`: `RMSE 0.026863`
- GRU `gru__repaired40`: `RMSE 0.027710`
- Prophet `prophet__safe`: `RMSE 0.068825`
- ARIMA `arima__safe180`: `RMSE 0.270940`

These numbers show that strong tree models and repaired sequence models clearly perform better than the classical baselines in this study.

### 7.3 Common-Subset Benchmark Results

To make comparison cleaner, a strict common-subset benchmark table was also exported. This table compares selected strong models only on the same exact test keys. It currently includes:

- XGBoost
- CatBoost
- LightGBM
- repaired LSTM
- repaired GRU
- one selected stacking run

On this strict common subset, the main results are:

1. XGBoost `full_multiscale`
   - `RMSE 0.021586`
   - `R2 0.928231`
2. Stacking `XGBoost + LSTM` (linear)
   - `RMSE 0.021619`
   - `R2 0.928010`
3. CatBoost `baseline_limited`
   - `RMSE 0.021935`
   - `R2 0.925889`
4. LightGBM `full_multiscale`
   - `RMSE 0.023002`
5. LSTM `repaired40`
   - `RMSE 0.026594`
6. GRU `repaired40`
   - `RMSE 0.027526`

This gives the cleanest final benchmark conclusion:

- XGBoost is the best final predictor on the strict common subset
- CatBoost is the next strongest tree benchmark
- LSTM is the strongest repaired sequence benchmark
- GRU is also strong, but weaker than LSTM

### 7.4 Interpretation of Results

The results suggest several important points.

First, the project's main story is not that one brand-new model beat everything else. The stronger story is that temporal feature design matters. XGBoost performed best when the historical context was organized in a more structured multi-scale way.

Second, the strong repaired LSTM and GRU results show that sequence models can be competitive when the sequence pipeline is methodologically correct. The earlier weak LSTM result was not proof that recurrent models are bad; it mainly reflected a pipeline problem.

Third, stacking was explored, but it did not become the best final system. The selected `XGBoost + LSTM` common-subset stack came very close to XGBoost, but it still remained slightly worse on the strict aligned comparison.

### 7.5 Insights, Strengths, and Limitations

The main strengths of this work are:

- a rebuilt and safer forecasting pipeline
- a clear temporal feature-design focus
- a clean XGBoost ablation study
- multiple supporting benchmark families

The main limitations are:

- the lag-design gain is moderate, not dramatic
- not all model families have the same test coverage
- the refined XGBoost interpretation is partly supported by log evidence rather than only saved benchmark artifacts
- `24h` and `72h` datasets were prepared, but not developed into full final reporting

### Figure/Table Placeholders

- `[PLACEHOLDER: Descriptive benchmark table]`
- `[PLACEHOLDER: Common-subset benchmark table]`
- `[PLACEHOLDER: Benchmark comparison graph at 300 DPI]`
- `[PLACEHOLDER: Model-family comparison bar chart at 300 DPI]`

## 8. Ablation Studies

### 8.1 Purpose of the Ablation Study

The ablation study was the most important part of the project. Its goal was to answer this question:

> Which temporal feature design works best for 48-hour soil moisture forecasting?

This is why XGBoost was used as the ablation anchor. It is strong, stable, and easy to interpret with explicit lag features.

### 8.2 Saved XGBoost Ablation Results

The saved XGBoost ablation artifact set includes these main groups:

- `baseline_limited`
- `short_lags`
- `short_plus_medium`
- `full_multiscale`

The best saved result was:

- `full_multiscale`: `RMSE 0.029122`

The main baseline was:

- `baseline_limited`: `RMSE 0.029588`

This means:

- absolute RMSE gain: about `0.000466`
- relative RMSE gain: about `1.57%`

This gain is not very large, but it is still meaningful because it is consistent and supports the idea that better temporal structure helps prediction.

### 8.3 What the Ablation Suggests

The saved ablation results show that the richer multi-scale setup performs better than the simpler baseline. In addition, the project work log records exploratory refined observations suggesting that:

- using no lag history is clearly worse
- weekly memory appears more useful than medium-only lag expansion
- rolling summaries help when combined with lag structure

Because those refined observations are preserved mainly in the work log, the report should present them carefully as supportive interpretation rather than the main saved result table.

### 8.4 CatBoost Confirmation Study

CatBoost was also tested as a confirmation model to see whether the same lag-design effect appears in another strong tree-based learner.

The result was useful:

- CatBoost performed best with `baseline_limited`
- `full_multiscale` did not beat it

This means the lag-design effect is:

- real
- useful
- but model-dependent

That is an important finding because it shows the effect is strongest and clearest in XGBoost, not universal across every tree model.

### Figure/Table Placeholders

- `[PLACEHOLDER: XGBoost ablation results table]`
- `[PLACEHOLDER: XGBoost ablation bar chart at 300 DPI]`
- `[PLACEHOLDER: CatBoost confirmation table]`

## 9. Conclusion and Future Work

### 9.1 Summary of Findings

This project developed a 48-hour soil moisture forecasting study centered on multi-scale temporal feature design. The data pipeline was rebuilt carefully, especially to fix the time-continuity issue in lag and target generation. A structured lag framework was then tested using XGBoost as the main ablation anchor.

The main finding is that the `full_multiscale` lag design performed better than the simpler `baseline_limited` setup. The improvement was modest, but it was still consistent and meaningful. Supporting benchmarks showed that CatBoost is also a strong tree model, while repaired LSTM and GRU models are competitive when the sequence pipeline is handled properly. Stacking was explored, but it did not beat the best aligned XGBoost model on the selected strict common-subset comparison.

### 9.2 Key Contributions

The main contributions of this project are:

1. A cleaner and safer hourly forecasting pipeline for ISMN station data.
2. A structured multi-scale temporal feature framework for 48-hour forecasting.
3. An ablation-based analysis showing that richer temporal context improves XGBoost performance over a simpler lag design.
4. Supporting benchmark evidence showing that the lag-design effect is model-dependent and that repaired sequence models can be strong.

### 9.3 Final Remarks

The final direction of this thesis is not stacking-first and not architecture-first. The project became a forecasting study about understanding which temporal history scales matter for medium-horizon soil moisture prediction. That makes the work more focused and more defensible.

### 9.4 Future Work

Possible future extensions include:

- fully extending the same study to `24h` and `72h`
- testing more stations and wider data coverage
- performing robustness analysis under different seasonal or wet/dry conditions
- adding uncertainty estimation
- using a stricter common-subset benchmark protocol from the beginning for all models
- revisiting ensemble learning only if a clear gain appears

## 10. References

This section is intentionally left as a placeholder for now.

What still needs to be added here:

- citation style selection
- full reference list
- minimum recent citations as required by the report format

### Placeholder

- `[PLACEHOLDER: Add formatted references here]`
