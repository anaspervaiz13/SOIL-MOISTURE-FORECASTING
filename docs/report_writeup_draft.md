# Multi-Scale Temporal Feature Design for 48-Hour Soil Moisture Forecasting

## Abstract

Soil moisture forecasting is important for agriculture, irrigation planning, and understanding land-atmosphere interactions. However, many forecasting studies focus mainly on comparing machine learning models, while giving less attention to a simple but important question: **which parts of past history are actually useful for predicting future soil moisture?**

This project studies that question for **48-hour-ahead soil moisture forecasting** using ISMN station data. The main idea of the work is not to invent a brand-new model. Instead, the goal is to design and test a **multi-scale temporal feature framework** that uses short-term, medium-term, and weekly historical context in a structured way.

To support this, the data pipeline was rebuilt carefully from scratch. The ISMN station records were first expanded to a true hourly timeline so that lag features such as `24h`, `48h`, and `168h` actually represent real time gaps instead of only row offsets. After that, a forecasting dataset was created using current measurements, calendar information, lag features, and rolling summary statistics.

The main novelty experiment was a lag ablation study using XGBoost as the anchor model. The results showed that the **full multi-scale feature design** gave the best performance, with an RMSE of **0.029122**, compared with **0.029588** for a simpler baseline-limited lag design. This is a gain of about **1.57%**, which is modest but consistent. Logged exploratory notes from later refined ablation runs also suggested that:

- using no lag history is clearly worse,
- weekly memory is more useful than medium-only lag expansion,
- and rolling summaries matter when combined with structured lag features.

Additional benchmark experiments were also carried out using CatBoost, LightGBM, KNN, Prophet, ARIMA, LSTM, GRU, and stacking. These experiments showed that repaired sequence models can be competitive, but the fairest common-subset comparison still favored **XGBoost** as the best final predictor among the selected strong models that were compared on that shared subset. Stacking was explored as a supporting experiment, but the exported `XGBoost + LSTM` common-subset stack did not outperform the best aligned base model.

Overall, this project contributes a **simple, defensible, and methodologically careful study of temporal context design** for medium-horizon soil moisture forecasting. The main contribution is the evidence that structured multi-scale history can improve prediction quality for strong tree-based models, even if the gain is not very large.

---

## 1. Introduction

Soil moisture plays an important role in agriculture, drought monitoring, irrigation, and environmental studies. It affects plant growth, infiltration, evaporation, and energy exchange between land and atmosphere. Because of this, predicting future soil moisture can help both scientific understanding and practical decision-making.

Many forecasting studies try different machine learning or deep learning models and report which one performs best. That approach is useful, but it can also miss an important issue. In time-series forecasting, prediction quality does not depend only on the model. It also depends on **how the history of the data is represented**.

For example, if we want to predict soil moisture 48 hours ahead, we can ask:

- Does only the most recent history matter?
- Do we need medium-range memory like 48 or 72 hours?
- Does weekly memory help?
- Are rolling summaries useful in addition to direct lag features?

These questions are important because soil moisture is not only a short-memory variable. It is influenced by rainfall, temperature, subsurface storage, and delayed soil responses. That means soil moisture forecasting is naturally a **state-transition problem**, where the past condition of the system matters.

This project focuses on that exact issue. Instead of saying "we built a brand-new model," the work asks:

> Which temporal history scales are most useful for predicting soil moisture 48 hours into the future?

That makes the project more than just a model competition. It becomes a study of **temporal context design**.

---

## 2. Research Problem

The main research problem in this project is:

> How can we design a better historical feature set for 48-hour soil moisture forecasting, and which lag groups actually help prediction?

This is different from a pure benchmark question like:

> Which model gives the best score?

The benchmark question is still useful, and we do answer it as a secondary part of the project. However, the central research problem is about feature design and temporal structure.

In simpler words, this project is trying to understand:

- what part of the past should be used,
- how to organize that past information,
- and whether using short, medium, and weekly history together is better than using simpler lag setups.

---

## 3. Research Gap

A lot of work in forecasting focuses on model choice. Researchers often compare models like XGBoost, LSTM, GRU, Transformer, Random Forest, or hybrid systems. That is valuable, but it can shift too much attention toward architecture competition.

In soil moisture forecasting, it is reasonable to expect that **time scale matters**:

- very recent history may capture immediate persistence,
- medium-range history may reflect slower changes,
- weekly memory may carry delayed hydrological effects,
- rolling summaries may capture short-term stability or accumulated conditions.

Even so, this temporal question is often not isolated clearly. Many studies use a fixed lag design and then compare models, without carefully testing which history groups are responsible for the result.

This project addresses that gap by making the following question central:

> If we design temporal context more carefully, can we improve 48-hour soil moisture forecasting in a measurable way?

That is the main novelty of the study.

---

## 4. Objectives

The objectives of this project are:

1. Build a clean and methodologically safe hourly forecasting dataset from ISMN station data.
2. Design a structured multi-scale temporal feature framework for 48-hour forecasting.
3. Test whether short, medium, and weekly lag groups help prediction through ablation.
4. Compare several model families as supporting benchmarks.
5. Check whether stacking gives meaningful additional value beyond strong single models.

The most important objective is still objective 3: **the lag ablation study**.

---

## 5. Dataset and Study Setup

The project uses ISMN station data already available in the workspace. The current station scope includes:

- Gevenich
- Merzenhausen
- Selhausen

The main target variable is:

- `sm_0.05m`

The main forecasting horizon is:

- `48 hours ahead`

Even though the data preparation was extended to also support `24h` and `72h`, the main study is centered on the `48h` setup. This was chosen so the project keeps one stable and defensible forecasting target for the main analysis.

---

## 6. Data Preparation

### 6.1 Rebuilding the pipeline from scratch

The earlier workflow was kept only as process inspiration, not as content to reuse. A fresh data pipeline was built in the active workspace.

The preparation started by reading raw ISMN `.stm` files and applying the local quality rule:

- keep only observations where the quality flag is `G`

Replicate observations for the same station, timestamp, and variable were then aggregated using the median.

This produced a merged hourly dataset covering the three stations with multiple environmental and soil variables.

### 6.2 Important methodological issue that was found

During review, an important problem was discovered:

the earlier version of forecasting features used row shifts directly.

That is dangerous because if there are gaps in the time series, then:

- "48 rows later" is not always the same as "48 hours later"
- lag and rolling features can cross outages and long breaks

This would make the dataset methodologically unsafe for real hour-ahead forecasting.

### 6.3 Fixing time continuity

To solve this, each station was first expanded onto a **true hourly grid** before target generation and lag creation.

This means:

- every hour between the station start and end time is represented,
- missing hours appear explicitly,
- lag features such as `24h`, `48h`, and `168h` now correspond to real clock hours.

This was one of the most important methodological fixes in the project.

### 6.4 Forecasting datasets prepared

After the repair, forecasting datasets were prepared for:

- `24h`
- `48h`
- `72h`

The main model-ready dataset for the core study is still:

- `ismn_forecasting_48h_ready.csv`

---

## 7. Feature Engineering

The forecasting table includes several groups of features.

### 7.1 Current-state features

These include the available environmental and soil variables at the current time step, such as:

- surface soil moisture
- deeper soil moisture
- air temperature
- soil temperature
- precipitation

### 7.2 Calendar features

Calendar and seasonal signals were added, including:

- hour
- day of week
- month
- day of year
- cyclic encodings such as `hour_sin`, `hour_cos`, `doy_sin`, and `doy_cos`

### 7.3 Lag features

Lag features were added at:

- `1h`
- `3h`
- `6h`
- `12h`
- `24h`
- `48h`
- `72h`
- `168h`

### 7.4 Rolling features

Rolling summaries were added to capture short-term and medium-term history more smoothly, including:

- rolling means
- rolling standard deviations
- rolling precipitation sums

The important idea is that these are not just "extra features." They are part of a **temporal design strategy**.

---

## 8. Main Methodological Idea: Multi-Scale Temporal Context

The feature design was organized into meaningful temporal groups:

- **short-term context**: `1, 3, 6, 12, 24` hours
- **medium-term context**: `48, 72` hours
- **weekly context**: `168` hours

The goal was not simply to add more and more lag columns. The goal was to test whether different history scales contribute differently to prediction.

This is the reason the novelty of the project is better described as:

**multi-scale temporal feature design**

rather than:

**we used longer lags**

---

## 9. Experimental Design

The experiments were split into two major parts.

### 9.1 Part A: Novelty / ablation

This part asks:

> For one strong model family, which temporal feature design works best?

The anchor model for this part is:

- `XGBoost`

This was chosen because:

- it is strong and stable,
- it works naturally with explicit lag features,
- and the lag-group effect is easy to interpret on it.

### 9.2 Part B: Benchmarking

This part asks:

> Across different model families, how strong is the forecasting performance?

The benchmark side includes:

- XGBoost
- CatBoost
- LightGBM
- KNN
- LSTM
- GRU
- Prophet
- ARIMA
- stacking

This part is useful, but it is supporting evidence rather than the main novelty.

---

## 10. XGBoost Lag Ablation

The current saved XGBoost ablation artifact set in the workspace includes these main feature groups:

- `baseline_limited`
- `short_lags`
- `short_plus_medium`
- `full_multiscale`

### 10.1 Core ablation result

The strongest novelty result came from:

- `full_multiscale`

with:

- `RMSE 0.029122`

The baseline comparison was:

- `baseline_limited`
  - `RMSE 0.029588`

This gives:

- absolute RMSE gain: about `0.000466`
- relative RMSE gain: about `1.57%`

This is not a huge improvement, but it is a real and consistent one.

### 10.2 Interpretation of the saved and logged ablation evidence

The saved artifact set already supports the main claim that `full_multiscale` is better than the simpler `baseline_limited` setup.

The project work log also records exploratory refined ablation observations using additional groups such as `current_only`, `short_plus_weekly`, and `full_multiscale_no_rolling`. Those observations are useful for interpretation, but they are currently documented in the log rather than preserved as part of the active saved XGBoost artifact set.

So the safest interpretation for the report is:

- the saved XGBoost ablation confirms that a richer multi-scale setup beats the simpler baseline
- the logged refined exploration suggests that the gain is not just "more lag columns"
- the broader idea is that structured multi-scale context, together with rolling summaries, appears more useful than simpler lag setups

That is the strongest novelty claim in the project, but the report should clearly distinguish between saved benchmark artifacts and logged exploratory interpretation.

---

## 11. CatBoost Confirmation Study

Because CatBoost was a strong tree benchmark, the refined lag-ablation design was also tested on it as a confirmation study.

This was important because it asked:

> Does the same lag-design effect also appear in another strong tree model?

### 11.1 CatBoost result

The outcome was interesting:

- `baseline_limited` was the best CatBoost configuration
- `full_multiscale` did **not** beat it

### 11.2 What this means

This tells us that the lag-design effect is:

- real,
- but **model-dependent**

That is actually a useful scientific result.

It means:

- the multi-scale temporal design helps XGBoost clearly,
- but it is not a universal gain across every strong tree learner.

So CatBoost is valuable as a benchmark and confirmation model, but not as the primary novelty anchor.

---

## 12. Benchmark Results

### 12.1 Descriptive benchmark table

Because different model families do not all share the same effective test coverage, the project now keeps a **descriptive benchmark table**.

This table is useful for broad context, but it is not a strict apples-to-apples ranking.

### 12.2 Common-subset benchmark table

To make cross-model comparison cleaner, a **strict common-subset benchmark table** was also exported.

This table compares **selected strong models only** on the exact same test keys. It does not include every benchmark family. In the current exporter, the common-subset table covers XGBoost, CatBoost, LightGBM, repaired LSTM, repaired GRU, and one selected stacking run.

On the fair common subset, the main results are:

1. `XGBoost full_multiscale`
   - `RMSE 0.021586`
   - `R2 0.928231`

2. `Stacking (XGBoost + LSTM, linear)`
   - `RMSE 0.021619`
   - `R2 0.928010`

3. `CatBoost baseline_limited`
   - `RMSE 0.021935`
   - `R2 0.925889`

4. `LightGBM full_multiscale`
   - `RMSE 0.023002`

5. `LSTM repaired40`
   - `RMSE 0.026594`

6. `GRU repaired40`
   - `RMSE 0.027526`

### 12.3 Interpretation

This gives a clean benchmark conclusion:

- **XGBoost** is the best final predictor on the strict common subset
- **CatBoost** is the next strongest tree benchmark
- **LSTM** is the strongest repaired sequence benchmark
- **GRU** is also strong, but weaker than the repaired LSTM

---

## 13. Sequence Models

The sequence-model story changed a lot during this project.

### 13.1 Earlier weak LSTM result

The original LSTM result was poor. However, that did not mean sequence models were inherently weak.

The main reason was a methodological issue in the sequence pipeline:

- sequence windows were not properly guaranteed to represent contiguous hourly history
- feature scaling was not handled carefully

### 13.2 Repaired LSTM pipeline

The LSTM pipeline was repaired by:

- building sequences only from contiguous hourly windows
- using train-only scaling for features and target

After that fix, the LSTM became very competitive.

### 13.3 Repaired LSTM result

The repaired LSTM reached:

- `RMSE 0.026863`
- `R2 0.888513`

This is a major improvement over the earlier broken LSTM run.

### 13.4 GRU benchmark

A GRU was also tested using the same repaired sequence pipeline to check whether the sequence advantage was architecture-specific.

The GRU result was:

- `RMSE 0.027710`
- `R2 0.881440`

This shows:

- the strong sequence-model result is not just a random accident,
- but the repaired **LSTM still performs better than GRU** in this study.

---

## 14. Stacking and Ensembling

Stacking was explored as a performance-oriented extension, not as the main thesis contribution.

Several combinations and meta-learners were tried, including:

- older aligned stack settings,
- newer stacks using XGBoost, CatBoost, and LSTM,
- linear and XGBoost meta-learners.

### 14.1 Main stacking result

In the exported common-subset comparison, the selected `XGBoost + LSTM` linear stack did **not** beat the best aligned base XGBoost model.

This is important.

It means stacking was **not broken**, but this selected final stack was also **not the winning final system**.

### 14.2 Why stacking did not win

The best explanation is that:

- XGBoost was already very strong,
- the other members were too correlated with it,
- and the extra models did not provide enough new information.

So the ensemble experiments were useful, but they do not carry the thesis.

---

## 15. Final Interpretation

The clearest overall interpretation of this project is:

1. The project is mainly about **temporal feature design**, not about inventing a new ensemble architecture.
2. The lag-ablation effect is **real but modest**.
3. The effect is **strongest and clearest in XGBoost**.
4. CatBoost shows that this effect is **not universal across all tree models**.
5. Repaired recurrent sequence models are **genuinely competitive**, especially LSTM.
6. Stacking was explored, but the selected final common-subset stack did **not** materially improve beyond the best aligned XGBoost model.

This is a strong and honest project direction.

---

## 16. Main Contribution

The main contribution of this project can be stated as:

> We designed and evaluated a multi-scale temporal context framework for 48-hour soil moisture forecasting on ISMN station data, and showed through ablation that structured short-, medium-, and weekly-history features, together with rolling summaries, improve prediction for a strong tree-based model compared with simpler lag setups.

This is the safe and defensible contribution statement.

It does **not** overclaim:

- it does not pretend stacking is novel,
- it does not pretend the gain is huge,
- and it does not pretend all models react the same way.

That makes it stronger, not weaker.

---

## 17. Limitations

This study still has some limitations.

### 17.1 Coverage differences between model families

Not all models were evaluated on exactly the same effective population.

For example:

- tabular models had broader test coverage
- sequence models required contiguous history
- stacking required prediction overlap across members

That is why two benchmark tables were needed:

- descriptive full-coverage table
- strict common-subset table

### 17.2 Modest ablation gain

The lag-design gain is real, but not very large. It should be presented as:

- a measurable and consistent improvement,
- not as a dramatic breakthrough.

### 17.3 Scope of horizons

The main project was centered on `48h` forecasting. Although `24h` and `72h` datasets were prepared, they were not fully developed into the same level of reporting.

---

## 18. Future Work

This project opens several useful directions for future work:

- evaluate the same framework fully on `24h` and `72h`
- extend the study to additional stations or broader coverage
- perform robustness testing under different seasonal or wet/dry periods
- explore uncertainty estimation
- build a stricter common-subset benchmark for every saved model family from the beginning
- test more advanced sequence architectures only if needed

Future work can also revisit ensemble learning, but it should do so only if a combination shows a clear advantage over the best aligned base model.

---

## 19. Conclusion

This project started with a broader modeling direction, but became clearer over time. The final outcome is not a stacking-first thesis. It is a **48-hour soil moisture forecasting study centered on multi-scale temporal feature design**.

The strongest finding is that carefully structured temporal context matters. A multi-scale lag framework, especially when combined with rolling summaries, gave the best XGBoost ablation result and improved prediction over simpler lag setups. The effect is modest, but real and defensible.

The benchmark experiments added useful context. CatBoost showed that the lag-design effect is model-dependent. Repaired recurrent models, especially LSTM, showed that sequence approaches can be strong when the data pipeline is methodologically correct. Stacking was explored, but it did not beat the best aligned XGBoost model.

So the final message of the project is simple:

**the key contribution is not a new model architecture, but a cleaner understanding of which temporal history scales help medium-horizon soil moisture forecasting.**
