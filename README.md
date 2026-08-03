<h1 align="center">📈 Logistic Regression — Real-World Classification</h1>

<p align="center"><b>Logistic-regression projects solving practical classification problems, end to end.</b></p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white" alt="scikit-learn"/>
  <img src="https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white" alt="pandas"/>
  <img src="https://img.shields.io/badge/Task-Classification-blue?style=flat-square" alt="Classification"/>
</p>

---

## 🔍 Projects

| Notebook | Problem |
|---|---|
| [`loan status prediction .ipynb`](loan%20status%20prediction%20.ipynb) | Predict whether a loan application is approved from applicant and financial attributes |
| [`LogReg (1).ipynb`](LogReg%20(1).ipynb) | Core logistic-regression walkthrough: fitting, predicting, and reading a decision boundary on a single-feature dataset |

---

## 🏦 Loan approval prediction

614 applications, 12 attributes: income, co-applicant income, loan amount and term, credit history, plus demographics.

### Results

Held-out test set (20%, stratified), features standardised inside a pipeline.

| Metric | Value |
|---|---|
| Accuracy | **0.837** |
| ROC-AUC | 0.845 |
| 5-fold CV accuracy | 0.808 (± 0.029) |

**Read that accuracy against the baseline.** 68.7% of applications in this dataset are approved, so a model that blindly approves everything scores 0.687. The real gain over doing nothing is about 15 percentage points, not 84.

### The number that actually matters

| Class | Precision | Recall |
|---|---|---|
| Approved | 0.81 | **1.00** |
| Rejected | **1.00** | **0.47** |

The model approves every application it should approve, and catches **fewer than half of the rejections** (18 of 38). When it does predict a rejection it is never wrong, but it misses 20 bad applications entirely.

For a lender that is the expensive direction to be weak in. A false rejection costs a customer; a false approval costs a loan. This model is tuned, by accident rather than design, to be generous. Fixing that is a threshold decision, not a modelling one.

### Strongest predictor

`Credit_History` dominates, with a standardised coefficient of 1.24 against 0.33 for the next feature. Applicants with a credit history are far more likely to be approved, and nothing else comes close.

### ⚠️ Three corrections to the original analysis

**1. The model never converged.** Fitting raised `ConvergenceWarning: lbfgs failed to converge after 100 iteration(s)`, and the coefficients interpreted afterwards came from a model that had not finished training. The cause was scale: `ApplicantIncome` runs to five figures while the dummies are 0 or 1. Standardising fixed it. The good news is that the original finding, that credit history dominates, survives the correction.

**2. A dtype conversion truncated the data.**

```python
dummies = pd.get_dummies(final, drop_first=True).astype('int64', 'floatb4')
```

The second argument to `astype` is `order`, not a dtype, and `'floatb4'` is not a valid value for anything. It was ignored, everything became `int64`, and `LoanAmount` and `CoapplicantIncome` were rounded down.

**3. Missing categorical values were never filled.** `fillna(loan.mean(numeric_only=True))` only touches numeric columns, so `Gender` (13 missing), `Married` (3), `Dependents` (15) and `Self_Employed` (32) stayed empty. The notebook's own `isna().sum()` showed this immediately afterwards. `get_dummies` then encoded those rows as all-zero across every category, silently inventing a category that does not exist. Now filled with the mode.

Also changed: `test_size` was 0.1, leaving 62 test rows where one misclassification moves accuracy by 1.6 points. Now 0.2, with 5-fold cross-validation so the noise in any single split is visible.

---

## 📉 Core logistic regression walkthrough

`LogReg (1).ipynb` fits a logistic regression on a single feature, age, predicting whether someone buys insurance. 58 records.

It is deliberately small. The point is the shape of the decision boundary and what a probability threshold means, not predictive performance. Treated as a teaching notebook rather than a result, and listed here as such.

## 🛠️ Stack

`Python` · `pandas` · `numpy` · `scikit-learn` · `matplotlib / seaborn`

## 📁 Data

`train_u6lujuX_CVtuZ9i.csv` and `age_insurance_dataset.csv` are committed, so both notebooks reproduce.

---
<p align="center"><i>Part of my <a href="https://github.com/buseko-Actuary">applied ML & data science portfolio</a> · Buseko · Insight Analytics</i></p>
