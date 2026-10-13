# Homework 3 - Answers and Reasoning

Dataset: `course_lead_scoring_2026.csv` (5000 rows, 9 columns).
Target: `converted`.

**Preparation:** categorical missing values → `'NA'`, numerical missing values → `0.0`.

---

## Q1 - Mode of `industry`

`technology` is the most frequent value (1173 out of 5000 rows), ahead of
`retail` (925) and `healthcare` (840).

**Answer: `technology`**

---

## Q2 - Biggest correlation

Absolute Pearson correlations between the numerical features:

| Feature pair | Correlation |
|---|---|
| `interaction_count` & `lead_score` | **0.916** |
| `number_of_courses_viewed` & `lead_score` | 0.757 |
| `number_of_courses_viewed` & `interaction_count` | 0.722 |
| `annual_income` & `interaction_count` | 0.123 |

**Answer: `interaction_count` and `lead_score`**

---

## Q3 - Mutual information (training set, rounded to 2 decimals)

| Feature | MI |
|---|---|
| `lead_source` | **0.03** |
| `employment_status` | 0.02 |
| `industry` | 0.00 |
| `location` | 0.00 |

**Answer: `lead_source`**

---

## Q4 - Logistic regression accuracy

One-hot encoding (`DictVectorizer`) + `LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)`,
fit on train, evaluated on validation.

Validation accuracy = 0.645 → **`0.65`**

---

## Q5 - Feature elimination (least useful feature)

Difference between the baseline accuracy and the accuracy without each feature:

| Removed feature | \|Δaccuracy\| |
|---|---|
| `number_of_courses_viewed` | **0.002** |
| `lead_source` | 0.003 |
| `interaction_count` | 0.044 |

**Answer: `number_of_courses_viewed`** (removing it barely changes the result, so it is the least useful).

---

## Q6 - Regularized logistic regression

Validation accuracy for each `C` (rounded to 3 decimals):

| C | Accuracy |
|---|---|
| 0.000001 | 0.598 |
| 0.00001 | 0.598 |
| 0.0001 | 0.613 |
| 0.001 | **0.645** |

**Answer: `0.001`**
