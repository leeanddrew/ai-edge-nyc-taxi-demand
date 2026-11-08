# SmartFleet — Project 1: Business Problem, Benchmarks and Model Proposal

## 1\. Business Problem

Hourly yellow-taxi pickups vary across Manhattan and NYC airports. Forecasting demand 24 hours in advance could help participating taxi operators and drivers decide where to position available vehicles, potentially reducing idle capacity and missed ride opportunities.

**Business question:** Can more accurate zone-level, day-ahead pickup forecasts support better taxi-positioning decisions?

**Data and target:** 1,504,741 zone-hour observations (January 29, 2023–August 31, 2026), covering 47 Manhattan zones plus JFK and LaGuardia airports. The target, `pickups`, is the number of *completed* pickups per zone per hour, not total passenger demand.

## 2\. Benchmark Methods

|**Benchmark**|**How it predicts pickups 24 hours ahead**|**Purpose**|
|-|-|-|
|Previous-day naive|Use pickups in the same zone and hour 24 hours earlier (`lag\_h`).|Tests whether yesterday's demand is enough.|
|Previous-week naive|Use pickups in the same zone and hour 168 hours earlier (`lag\_168h`).|Captures weekday patterns.|
|Four-week historical average|Average the same zone, weekday and hour across the preceding four weeks (`same\_hour\_4wk`).|Provides a stable seasonal reference.|
|Single regression tree|Fit one shallow decision tree using lagged pickups, zone, hour, weekday and calendar indicators.|Tests whether simple, interpretable nonlinear rules improve forecasts.|

The first three are rule-based forecasts; the single decision tree requires training. All forecasts must use information available at least 24 hours before the target hour.

## 3\. Proposed Predictive Models and SmartFleet Trade-offs

|Model|Application to SmartFleet|Strength|Trade-off|
|-|-|-|-|
|**Multiple linear regression**|Predict pickups using 24-hour/weekly lags, four-week average, hour, weekday, month and properly encoded taxi-zone categories.|Fast and interpretable; shows whether combining demand history and calendar effects beats simple rules.|A fixed linear structure may miss rush-hour × zone interactions and demand spikes.|
|**Ridge regression**|Use the same predictors as linear regression, while shrinking coefficients for correlated demand lags such as `lag\_h`, `lag\_168h` and `same\_hour\_4wk`.|Can improve stability when historical demand features overlap.|Still relies on largely linear relationships; regularization strength requires time-ordered tuning.|
|**Random forest regression**|Learn different demand patterns across airport versus Manhattan zones, weekdays versus weekends, and high- versus low-volume hours.|Captures nonlinear effects and feature interactions without specifying them in advance.|Harder to interpret and more computationally demanding; may smooth rare demand spikes.|
|**Gradient boosting regression**|Fit sequential trees to correct remaining errors, potentially improving forecasts during rush hours, holidays and unusual demand periods.|Strong candidate for structured, nonlinear demand data.|Needs careful time-ordered tuning and can overfit unusual historical events.|

These are **candidate models**, not completed model results. A single regression tree is retained as a simple trained benchmark rather than counted as one of the four proposed models.

## 4\. Evaluation Methodology and Preliminary Results

**Validation design:** Use time-ordered validation within the development period (January 29, 2023–August 31, 2025). Preserve the final test year (September 1, 2025–August 31, 2026) for evaluation after model selection. All models must be compared on the same forecast rows.

**Metrics:**

* **MAE (pickups per zone-hour)** = sum of `|actual − predicted|` divided by number of observations. Lower means fewer pickups missed on average.
* **WAPE (%)** = sum of `|actual − predicted|` divided by total actual pickups, multiplied by 100. Lower means less total error relative to demand.

**Preliminary validation check:** The three rule-based forecasts were compared with actual `pickups` for **216,384 zone-hour records from March 1–August 31, 2025** in `model\_table.csv.gz`.

|Benchmark|MAE|WAPE|
|-|-:|-:|
|Previous-day naive|28.40|28.17%|
|Previous-week naive|20.45|20.28%|
|**Four-week historical average**|**17.41**|**17.27%**|

For example, a row with **19 actual pickups** and a `same\_hour\_4wk` forecast of **16.75** has an absolute error of **2.25 pickups**. Summing absolute errors over all 216,384 rows and dividing by that row count produces **MAE = 17.41**. Dividing that same sum by total actual pickups produces **WAPE = 17.27%**.

The four-week average is the strongest of these three preliminary historical benchmarks. Decision-tree and proposed-model results are **not yet available**. Results should be reproduced under the final shared time-ordered validation setup before reporting a final model ranking.

**Forecast realism:** Historical weather in the prepared table reflects conditions at the target hour rather than the weather forecast available 24 hours earlier. Weather-based experiments should be identified as optimistic unless replaced with authentic day-ahead forecast inputs.

## 5\. Business Implications

* **Overforecasting** can encourage excess vehicle positioning, increasing idle time or unnecessary repositioning.
* **Underforecasting** can leave busy zones short of available taxis, potentially reducing completed rides and contribution.

MAE and WAPE measure prediction accuracy, **not dollar savings**. Operational value requires a separate, consistent vehicle-allocation simulation with explicit assumptions for taxi capacity, excess capacity costs, and potentially unserved rides. Because the dataset records completed pickups rather than all attempted hails, simulated benefits must not be presented as observed revenue gains.

