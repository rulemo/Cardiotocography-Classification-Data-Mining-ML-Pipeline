# Cardiotocography Classification – Data Mining & ML Pipeline

A complete data mining and classification pipeline applied to the **Cardiotocography** dataset (fetal monitoring signals), built in Python with `scikit-learn`.

## Project Overview

### 1. Exploratory Data Analysis (EDA)
- Descriptive statistics, class distribution, and missing value detection
- Visualizations: class distribution, missing values per feature, and correlation matrix

### 2. Data Preparation
- Missing value imputation (mean strategy)
- Comparison of 3 preprocessing techniques: standardization (Z-score), Min-Max normalization, and dimensionality reduction with **PCA** (retaining 95% of variance)

### 3. Classification Experiments
- Cross-evaluation of **Naive Bayes**, **Decision Tree**, and variants with different hyperparameters (var_smoothing, max_depth, min_samples_split)
- Each classifier tested on every version of the dataset (original, standardized, normalized, PCA) → 24 combinations in total

### 4. Results Evaluation
- Metrics: Accuracy, Precision, Recall, F1-score (weighted)
- Full comparative table and ranking of the top 10 models by accuracy

## Tech Stack
`Python` · `pandas` · `NumPy` · `scikit-learn` · `matplotlib` · `seaborn`
