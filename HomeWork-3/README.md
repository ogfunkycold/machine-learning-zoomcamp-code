# Machine Learning Zoomcamp 2026 - Homework 3

**Machine Learning for Classification** - lead-scoring dataset.

| | |
|---|---|
| **Course** | [Machine Learning Zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp) (DataTalksClub, 2026 cohort) |
| **Module** | 03 - Machine Learning for Classification |
| **Homework** | [HW3 - course lead scoring](https://courses.datatalks.club/ml-zoomcamp-2026/homework/hw03) |
| **Dataset** | [`course_lead_scoring_2026.csv`](data/course_lead_scoring_2026.csv) (5000 rows) |
| **Target** | `converted` - did the lead sign up for the platform? |

## Answers

| # | Question | Answer | Value |
|---|----------|--------|-------|
| 1 | Mode of `industry` | `technology` | 1173 leads |
| 2 | Biggest correlation | `interaction_count` & `lead_score` | 0.916 |
| 3 | Biggest mutual information | `lead_source` | 0.03 |
| 4 | Logistic regression accuracy | `0.65` | 0.645 |
| 5 | Least useful feature (elimination) | `number_of_courses_viewed` | Δacc 0.002 |
| 6 | Best regularization `C` | `0.001` | acc 0.645 |

Full write-up: [`ANSWERS.md`](ANSWERS.md).

## Repository layout

```
HomeWork-3/
├── README.md               # this file
├── ANSWERS.md              # question-by-question reasoning
├── SUBMISSION.md           # the text needed for the homework form
├── learning-in-public.md   # blog-style public progress write-up
├── solution.ipynb          # notebook with executed outputs
├── solution.py             # same solution as a script
├── requirements.txt
├── homework.md             # original assignment statement
└── data/
    └── course_lead_scoring_2026.csv
```

## Reproduce

```bash
pip install -r requirements.txt

# Option A - script
python solution.py

# Option B - notebook
jupyter notebook solution.ipynb
```

## Method summary

1. **Data preparation** - categorical missings → `'NA'`, numerical missings → `0.0`.
2. **Split** - `train_test_split(test_size=0.2, random_state=42)` then `0.25` on the remainder (60/20/20).
3. **Correlation** - absolute Pearson correlations over the numerical features.
4. **Mutual information** - `mutual_info_score` on the training split, rounded to 2 decimals.
5. **Model** - one-hot encoding via `DictVectorizer` + `LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)`.
6. **Feature elimination** - retrain without each candidate feature and compare validation accuracy.
7. **Regularization** - sweep `C ∈ {1e-6, 1e-5, 1e-4, 1e-3}`, pick the best (smallest on ties).

## Learning in public

Up to 7 public progress links for the submission form are collected in
[`learning-in-public.md`](learning-in-public.md).
