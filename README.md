# Machine Failure Prediction

Machine learning project for predicting machine failures from operational
sensor data.

The goal of this project is to build a binary classification model capable
of identifying machines that are at risk of failure based on measurements
such as temperature, rotational speed, torque, tool wear, and machine type.

A major challenge in this problem is the strong class imbalance: machine
failures represent only a small fraction of the available observations.
Therefore, model performance is evaluated using metrics such as precision,
recall, F1, F2, ROC-AUC, and PR-AUC rather than relying on accuracy alone.

The project covers the complete machine-learning workflow, including:

- Exploratory data analysis
- Data preprocessing and feature preparation
- Handling class imbalance
- Logistic Regression, Random Forest, and XGBoost
- Hyperparameter tuning
- Decision-threshold optimization
- Model comparison
- Feature importance and permutation importance
- SHAP-based model explainability
- Train / validation / test separation
- Final evaluation on a reserved test set

## Problem Statement

Unexpected machine failures can result in production downtime, maintenance
costs, and operational disruption.

The objective of this project is to predict whether a machine is likely to
fail based on its current operating conditions.

This is formulated as a binary classification problem:

- `0` — No machine failure
- `1` — Machine failure

Because failures are rare, the objective is not simply to maximize accuracy.
In particular, missing a real failure may be more costly than generating an
additional maintenance alert.

For this reason, recall and F2-score play an important role in model and
decision-threshold selection.

## Dataset

The dataset contains 136,429 machine-operation observations.

Each observation contains measurements describing the operating state of a
machine:

- Air temperature [K]
- Process temperature [K]
- Rotational speed [rpm]
- Torque [Nm]
- Tool wear [min]
- Machine type

The target variable is:

- Machine failure

where `0` represents normal operation and `1` represents a machine failure.

### Class Distribution

The dataset is highly imbalanced:

- No failure: approximately 98.4%
- Machine failure: approximately 1.6%

This imbalance is an important characteristic of the problem. A classifier
that predicts "No Failure" for almost every observation could achieve very
high accuracy while failing to identify actual machine failures.

Therefore, accuracy alone is not an appropriate metric for evaluating the
models in this project.

## Exploratory Data Analysis

Exploratory data analysis was performed before model development to better
understand the dataset, feature distributions, class imbalance, and the
relationship between operating conditions and machine failures.

The analysis showed that machine failures are rare and that several numeric
operating variables contain useful predictive information.

Later model-explainability analysis supported the importance of variables
such as:

- Torque
- Rotational speed
- Air temperature
- Process temperature
- Tool wear

Machine type contributed less predictive information than the main numeric
operating measurements.

The EDA and subsequent explainability analysis suggested that machine failure
is not determined by a single variable. Instead, failure risk depends on
combinations of operating conditions.

## Data Preprocessing

The raw features contain both numerical and categorical variables, requiring
different preprocessing strategies.

### Numerical Features

The numerical features used by the models are:

- Air temperature [K]
- Process temperature [K]
- Rotational speed [rpm]
- Torque [Nm]
- Tool wear [min]

Missing numerical values are handled using median imputation, followed by
standardization using `StandardScaler`.

### Categorical Features

Machine type is treated as a categorical feature.

Missing categorical values are handled using the most frequent category,
and the feature is transformed using one-hot encoding.

### Preprocessing Pipeline

The preprocessing steps are combined with the classifier in a Scikit-learn
`Pipeline`.

This ensures that the same preprocessing operations are applied consistently
during both model training and prediction, while also reducing the risk of
data leakage.

## Modeling Approach

Three classification algorithms were investigated during model development:

### Logistic Regression

Logistic Regression was used as a simple and interpretable baseline model.

Because machine failures are highly underrepresented in the dataset,
class weighting was used to give greater importance to the minority
failure class.

The model provided a useful baseline but showed substantially weaker
discrimination performance than the tree-based models.

### Random Forest

Random Forest was evaluated as a nonlinear ensemble model capable of
capturing more complex relationships between operating conditions and
machine failures.

Both a baseline Random Forest and a hyperparameter-tuned version were
evaluated.

The model achieved substantially better ROC-AUC and PR-AUC than Logistic
Regression and also provided feature-importance information that was useful
for understanding the predictions.

### XGBoost

XGBoost was evaluated as a gradient-boosted tree model.

Both a baseline configuration and a hyperparameter-tuned configuration were
tested.

XGBoost achieved the strongest ranking performance among the evaluated
models, including the highest ROC-AUC and PR-AUC during model comparison.

For this reason, XGBoost was selected for the final validation-based model
selection workflow.

## Model Evaluation

Because the dataset is highly imbalanced, model performance was evaluated
using several complementary metrics rather than accuracy alone.

The main evaluation metrics were:

- **Precision** — the proportion of predicted failures that were actual failures.
- **Recall** — the proportion of actual failures successfully detected.
- **F1-score** — the harmonic mean of precision and recall.
- **F2-score** — similar to F1, but gives greater importance to recall.
- **ROC-AUC** — measures the model's ability to rank positive examples above
  negative examples across classification thresholds.
- **PR-AUC** — summarizes the precision-recall trade-off and is particularly
  informative for imbalanced classification problems.

Recall is especially important in this project because a false negative
represents a machine failure that the model failed to detect.

For this reason, F2-score was also used during threshold selection because
it gives more weight to recall than precision.

### Model Comparison

The evaluated models showed a clear improvement from the linear baseline
to the tree-based ensemble models.

| Model | ROC-AUC | PR-AUC |
|---|---:|---:|
| Logistic Regression | 0.8175 | 0.1535 |
| Random Forest | 0.9153 | 0.3882 |
| Tuned Random Forest | 0.9234 | 0.3991 |
| XGBoost | **0.9261** | **0.4374** |
| Tuned XGBoost | 0.9250 | 0.4369 |

XGBoost achieved the strongest overall ranking performance.

An important observation was that hyperparameter tuning did not
automatically improve performance. The tuned XGBoost configuration performed
similarly to, but slightly below, the baseline XGBoost model on ROC-AUC and
PR-AUC.

## Decision Threshold Optimization

Binary classifiers typically use a probability threshold of `0.50` to
convert predicted probabilities into class predictions.

However, `0.50` is not necessarily the optimal threshold for an imbalanced
machine-failure problem.

At the default threshold, XGBoost produced relatively high precision but
missed many actual failures. Lowering the threshold increased the model's
sensitivity to the failure class.

Thresholds were therefore evaluated using precision, recall, F1, and F2.

The final threshold was selected using the validation set only.

For XGBoost, the best validation threshold according to F2-score was:

| Metric | Validation Result |
|---|---:|
| Threshold | **0.09** |
| Precision | 0.3425 |
| Recall | **0.6512** |
| F1 | 0.4489 |
| F2 | **0.5517** |

The threshold of `0.09` was selected because the project prioritizes
detecting machine failures while still maintaining a meaningful level of
precision.

## Validation Strategy

To avoid using the test set for model or threshold selection, the dataset
was divided into three subsets:

- **Training set:** approximately 64%
- **Validation set:** approximately 16%
- **Test set:** approximately 20%

The training set was used to fit the model.

The validation set was used for model-selection decisions and decision
threshold optimization.

The test set remained reserved during this process and was not used to
select the final threshold.

After selecting XGBoost and locking the decision threshold at `0.09`, the
model was refitted using the combined training and validation data,
representing approximately 80% of the dataset.

The reserved 20% test set was then used once for final model evaluation.

## Model Explainability

In addition to predictive performance, model explainability was analyzed to
understand which operating conditions contribute most strongly to machine
failure predictions.

Several complementary explainability techniques were used:

- Random Forest feature importance
- Permutation importance
- SHAP values
- SHAP dependence analysis
- Interaction analysis between important operating variables

### Feature Importance

The tuned Random Forest identified the following features as the most
important:

| Feature | Importance |
|---|---:|
| Rotational speed | 0.3297 |
| Torque | 0.2842 |
| Tool wear | 0.1745 |
| Air temperature | 0.1223 |
| Process temperature | 0.0783 |

Machine type contributed substantially less importance than the main
continuous operating measurements.

Feature importance indicates how strongly the Random Forest relied on each
feature, but it does not directly explain whether high or low feature values
increase the predicted failure risk.

### Permutation Importance

Permutation importance was also calculated on held-out data using average
precision as the scoring metric.

The strongest features were:

| Feature | Mean Importance |
|---|---:|
| Air temperature | 0.2181 |
| Torque | 0.2110 |
| Rotational speed | 0.2062 |
| Process temperature | 0.0981 |
| Tool wear | 0.0881 |
| Machine type | 0.0133 |

Permutation importance measures how much model performance decreases when
the information contained in a feature is disrupted.

Although the exact ranking differs from the Random Forest's built-in feature
importance, both analyses indicate that the numerical operating conditions
contain substantially more predictive information than machine type.

### SHAP Analysis

SHAP was used to obtain a more detailed view of how individual feature
values influence the model's predictions.

The SHAP analysis showed that the effect of important features is nonlinear.
A feature cannot necessarily be described as simply "higher means more
failure risk" or "lower means less failure risk."

Torque and rotational speed showed particularly important patterns in the
model's predictions.

SHAP dependence plots were therefore used to examine how different values
of these variables affect the predicted failure risk.

### Feature Interactions

The relationship between torque and rotational speed was also investigated
using a SHAP interaction-style visualization.

The analysis suggested that the effect of torque on predicted failure risk
depends partly on the rotational-speed operating regime.

This supports an important conclusion from the project: machine failure risk
is associated with combinations of operating conditions rather than with a
single isolated measurement.

## Final Model

The final selected model is XGBoost with a decision threshold of 0.09.

The threshold was selected using the validation set with an emphasis on
F2-score, giving greater importance to recall.

After model and threshold selection, the model was refitted using the
combined training and validation data. The reserved test set was then used
once for final evaluation.

### Final Test Results

| Metric | Score |
|---|---:|
| Accuracy | 0.9727 |
| Precision | 0.3167 |
| Recall | 0.6349 |
| F1 | 0.4226 |
| F2 | 0.5287 |
| ROC-AUC | 0.9261 |
| PR-AUC | 0.4374 |


## Project Structure

The project is organized into separate directories for data, trained models,
generated reports, source code, notebooks, and automated tests.

machine-failure-prediction/
│
├── data/
│   ├── raw/
│   │   └── train.csv
│   ├── interim/
│   └── processed/
│
├── models/
│   ├── logistic_regression_production.joblib
│   ├── random_forest_production.joblib
│   ├── random_forest_tuned_production.joblib
│   ├── xgboost_production.joblib
│   ├── xgboost_tuned_production.joblib
│   ├── xgboost_validation.joblib
│   └── xgboost_final.joblib
│
├── notebooks/
│   └── archive/
│       └── original_notebook.ipynb
│
├── reports/
│   ├── figures/
│   ├── metrics/
│   └── tables/
│       └── model_comparison.csv
│
├── src/
│   └── machine_failure/
│       ├── data.py
│       ├── eda.py
│       ├── features.py
│       ├── preprocessing.py
│       ├── model.py
│       ├── train.py
│       ├── evaluate.py
│       ├── threshold.py
│       ├── tuning.py
│       ├── comparison.py
│       ├── explainability.py
│       ├── shap_analysis.py
│       ├── reporting.py
│       └── __init__.py
│
└── tests/
    ├── test_data.py
    ├── test_features.py
    ├── test_model.py
    └── test_preprocessing.py

## Installation and Usage

This project uses [uv](https://docs.astral.sh/uv/) for Python dependency and
environment management.

### 1. Clone the Repository

```bash
git clone <repository-url>
cd machine-failure-prediction
```

### 2. Install Dependencies

Install the project dependencies from the lock file:

```bash
uv sync
```

This creates the virtual environment and installs the exact dependency
versions required by the project.

### 3. Dataset

Place the training dataset at:

```text
data/raw/train.csv
```

### 4. Run Exploratory Data Analysis

```bash
uv run python -m machine_failure.eda
```

### 5. Train a Model

Model training is implemented in:

```text
src/machine_failure/train.py
```

For example:

```bash
uv run python -m machine_failure.train
```

### 6. Evaluate Models

```bash
uv run python -m machine_failure.evaluate
```

Evaluation generates classification metrics and figures including:

- Confusion matrices
- ROC curves
- Precision-recall curves

Generated results are stored under:

```text
reports/
├── figures/
├── metrics/
└── tables/
```

### 7. Threshold Analysis

Decision-threshold optimization can be performed using:

```bash
uv run python -m machine_failure.threshold
```

Threshold selection is performed on the validation set rather than the
reserved test set.

### 8. Model Explainability

Feature and permutation importance:

```bash
uv run python -m machine_failure.explainability
```

SHAP analysis:

```bash
uv run python -m machine_failure.shap_analysis
```

### 9. Run Tests

Run the automated test suite with:

```bash
uv run pytest
```

## Requirements

The project was developed and tested using Python 3.12.

Main dependencies include:

- pandas
- NumPy
- Scikit-learn
- XGBoost
- SHAP
- Matplotlib
- Seaborn
- Joblib

Development dependencies include:

- pytest
- pytest-cov
- Ruff
- JupyterLab

Dependencies and reproducible versions are managed using `uv` through
`pyproject.toml` and `uv.lock`.


## Key Takeaways

This project demonstrates an end-to-end machine-learning workflow for a
highly imbalanced predictive-maintenance problem.

The main conclusions are:

- Accuracy alone is misleading when machine failures represent only a small
  fraction of the dataset.
- Tree-based ensemble models substantially outperformed the Logistic
  Regression baseline.
- XGBoost achieved the strongest overall ROC-AUC and PR-AUC performance.
- Hyperparameter tuning did not automatically improve model performance,
  demonstrating the importance of empirical model comparison.
- Decision-threshold optimization significantly improved failure detection
  compared with the default threshold of 0.50.
- A threshold of 0.09 was selected using validation data with an emphasis on
  F2-score and recall.
- The final XGBoost model detected approximately 63.5% of machine failures
  on the previously untouched test set.
- Explainability analysis showed that operating variables such as torque,
  rotational speed, and temperature contain important predictive information.
- Feature effects are nonlinear and machine-failure predictions depend on
  combinations of operating conditions rather than a single measurement.
- Keeping training, validation, and test data separate prevents the final
  test set from influencing model-selection decisions.
  