# Hospital Readmission Risk Analysis for Diabetic Patients

## Overview

This project analyzes hospital readmission patterns for diabetic patients using a comprehensive dataset from 130 US hospitals spanning 1999-2008. The goal is to predict the likelihood of early readmission (within 30 days of discharge) to help hospitals improve diabetes management and reduce preventable readmissions.

## Dataset

The analysis uses the **UCI Diabetes 130-US Hospitals Dataset** (1999-2008), a multivariate clinical dataset containing:

- **101,766** patient encounter records
- **47** features including demographic, clinical, and medication information
- **Target Variable**: Readmission status within 30 days of discharge
  - `NO`: Patient was not readmitted
  - `<30`: Patient was readmitted within 30 days (positive class)
  - `>30`: Patient was readmitted after 30 days (not within target window)

**Data Source**: [UCI Machine Learning Repository - Diabetes 130-US Hospitals Dataset](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)

### Clinical Context

Despite evidence-based interventions improving outcomes for diabetic patients, many do not receive proper diabetes care in hospital settings. This leads to:
- Increased hospital readmissions and associated costs
- Higher patient morbidity and mortality
- Complications due to poor glycemic control

This project aims to identify patients at high risk of early readmission to enable targeted interventions.

## Project Goal

**Primary Objective**: Build a predictive classification model to identify diabetic patients at risk of readmission within 30 days of discharge.

**Secondary Objectives**:
- Understand which clinical and demographic factors most influence readmission risk
- Compare the performance of various machine learning algorithms
- Develop a preprocessing pipeline that handles missing data and heterogeneous feature types appropriately

## Dependencies

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
scipy>=1.7.0
xgboost>=1.5.0
lightgbm>=3.3.0
jupyter>=1.0.0
```

### Installation

```bash
pip install -r requirements.txt
```

## Preprocessing & Feature Engineering

### 1. Data Cleaning
- **Deduplication**: Retained one record per patient, selecting the encounter with the longest hospital stay
- **Missing Value Analysis**: Identified features with >95% missingness
  - Dropped: `weight` column (>95% missing)
  - Dropped: `medical_specialty` and `payer_code` (low association with target)
  - Rows with missing diagnostic codes dropped (race, diag_1, diag_2, diag_3)

### 2. Feature Selection
Used statistical association measures to identify predictive features:
- **Numerical Features** (9 features): Analyzed using **Eta-squared** to measure variance explained relative to readmission status
- **Categorical Features** (22 features): Evaluated using **Cramér's V** for association strength with the target variable

**Final Feature Set**:
- 9 numerical features (e.g., time_in_hospital, num_lab_procedures, num_medications)
- 22 categorical features (admission type, discharge disposition, medications, diagnoses)
- 1 ordinal feature (age group)

### 3. Feature Engineering
- **ICD-9 Code Mapping**: Consolidated three diagnostic code columns (diag_1, diag_2, diag_3) into 9 clinically meaningful categories:
  - Circulatory, Respiratory, Digestive, Diabetes, Injury, Musculoskeletal, Genitourinary, Neoplasms, Other, External causes

### 4. Preprocessing Pipeline
Implemented a `ColumnTransformer` with specialized handling for different feature types:

| Feature Type | Encoder | Preprocessing |
|---|---|---|
| Low-cardinality categorical (≤10 unique) | OneHotEncoder | Most frequent imputation → Encoding |
| High-cardinality categorical (>10 unique) | TargetEncoder | Most frequent imputation → Target encoding (reduces dimensionality) |
| Ordinal (age) | OrdinalEncoder | Most frequent imputation → Ordinal encoding |
| Numerical | StandardScaler | Mean imputation → Standardization |

**Rationale**: TargetEncoder for high-cardinality features prevents the curse of dimensionality while preserving target information.

## Model Building & Evaluation

### Methodology
- **Train-Test Split**: 70-30 stratified split (preserves class distribution)
- **Cross-Validation**: 5-fold Stratified K-Fold to assess generalization
- **Class Balancing**: Applied `class_weight='balanced'` to handle imbalanced classes
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-score, Confusion Matrix

### Models Evaluated

Six machine learning algorithms were compared:

1. **Decision Tree Classifier** (max_depth=10)
2. **Logistic Regression** (L2 regularization, max_iter=1000)
3. **Random Forest** (n_estimators=100)
4. **Gradient Boosting Classifier**
5. **XGBoost** (multi-class classification, objective='multi:softmax')
6. **LightGBM** (with class balancing)

### Key Findings

**Cross-Validated Performance Summary**:
- **Best Performers**: Ensemble methods (Random Forest, Gradient Boosting, XGBoost, LightGBM) showed superior performance compared to linear and tree-based models
- **Class Imbalance Challenge**: The dataset exhibits significant class imbalance (<30 readmission rate ~11%), requiring careful handling via stratified splitting and class weights
- **Feature Importance**: Medication usage, admission type, and diagnostic codes emerged as strong predictors
- **Trade-offs**: Models generally achieved higher accuracy on the majority class (NO readmission), suggesting potential for threshold tuning to improve minority class recall

## Conclusions

### Insights
1. **Readmission is Predictable**: Multiple algorithms achieved reasonable cross-validated performance, indicating readmission risk has identifiable patterns in clinical data
2. **Medication Patterns Matter**: Whether a patient is on specific diabetes medications (metformin, insulin, etc.) and medication changes significantly influence readmission risk
3. **Clinical Factors**: Admission source, discharge disposition, and primary diagnosis categories are important predictive signals
4. **Ensemble Methods Outperform**: Tree-based ensemble methods (Random Forest, XGBoost, LightGBM) consistently outperformed simpler models

### Practical Applications
- **Risk Stratification**: The models enable identification of high-risk patients at discharge for targeted intervention
- **Resource Allocation**: Hospitals can prioritize follow-up care and monitoring for patients flagged as high-risk
- **Intervention Design**: Focus areas include medication management, discharge planning, and post-discharge follow-up

### Next Steps for Improvement
1. Address data leakage and ensure consistent train-test splits across all model comparisons
2. Perform hyperparameter optimization (GridSearchCV) for top-performing models
3. Evaluate on held-out test set to assess real-world generalization
4. Consider probability calibration for risk scoring
5. Implement SHAP or feature importance analysis for model interpretability
6. Explore class rebalancing techniques (SMOTE, class weights) to improve minority class prediction
7. Validate findings on external datasets from different hospital systems

## Usage

Navigate to the notebooks directory and open the Jupyter notebook:

```bash
jupyter notebook notebooks/01-data_exploration.ipynb
```

The notebook contains:
- Detailed exploratory data analysis (EDA)
- Missing value pattern analysis
- Feature importance calculations
- Complete preprocessing pipeline implementation
- Model training and cross-validation results

## Project Structure

```
.
├── README.md
├── notebooks/
│   └── 01-data_exploration.ipynb    # Main analysis notebook
├── data/
│   └── raw/
│       ├── diabetic_data.csv        # UCI dataset
│       └── diabetic_data_variables.csv  # Data dictionary
└── requirements.txt
```

## References

Strack, B., DeShazo, J. P., Gennings, C., Olmo, J. L., Ventura, S., Cios, K. J., & Clore, J. N. (2014). Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records. *PLoS ONE*, 9(1), e86635.

Dataset: https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008

## License

This project uses the UCI Diabetes 130-US Hospitals Dataset, which is publicly available for educational and research purposes.
