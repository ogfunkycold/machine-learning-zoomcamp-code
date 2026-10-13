# Learning in Public - ML Zoomcamp 2026, Homework 3

This file doubles as **the write-up** (ready to publish) and the **source of the
up to 7 public progress links** requested by the submission form.

GitHub account: [`ogfunkycold`](https://github.com/ogfunkycold)

---

## The 7 public progress links

Replace the placeholder repo name after pushing (see "Publish" below). Once the
repo is public and the blog post is live, the rows below are copy-paste ready.

| # | Type | Link |
|---|------|------|
| 1 | GitHub repo (code + executed notebook) | `https://github.com/ogfunkycold/ML-ZoomCamp/tree/main/HomeWork-3` |
| 2 | Notebook (`solution.ipynb`) with outputs | `https://github.com/ogfunkycold/ML-ZoomCamp/blob/main/HomeWork-3/solution.ipynb` |
| 3 | Solution script (`solution.py`) | `https://github.com/ogfunkycold/machine-learning-zoomcamp-code/blob/main/HomeWork-3/solution.py` |
| 4 | Blog / dev.to write-up (this document) |  |
| 5 | LinkedIn progress post |                                                              |
| 6 | X / Twitter thread |  |
| 7 | Kaggle / nbviewer render of the notebook |  |

> The homework form accepts **up to 7** links. If you publish fewer, just use the
> first 1–3 (repo + notebook + write-up).

### Publish

```bash
# from the folder that contains HomeWork-3
git init
git add .
git commit -m "ML Zoomcamp 2026 - Homework 3"
git branch -M main
git remote add origin https://github.com/ogfunkycold/machine-learning-zoomcamp-code.git
git push -u origin main
```

Then paste the published URLs into the table above and into the submission form.

---

## Blog write-up (ready to post)

### ML Zoomcamp 2026 - Homework 3: predicting which leads convert

**Module 3** of Machine Learning Zoomcamp moves from regression to
**classification**. The task: given a marketing dataset of 5000 leads, predict the
binary target `converted` - did the person sign up?

#### What I did

1. **Cleaning** - filled categorical missings with `'NA'` and numerical missings
   with `0.0`, then split 60/20/20 with a fixed seed so results reproduce.
2. **Exploration** - a correlation matrix showed `interaction_count` and
   `lead_score` are almost redundant (0.92). On the categorical side, mutual
   information singled out `lead_source` as the most informative feature.
3. **Modeling** - one-hot encoding with `DictVectorizer` and a `liblinear`
   logistic regression. This is the first classifier of the course and it already
   hits **0.65 accuracy** on validation.
4. **Feature elimination** - dropping `number_of_courses_viewed` changed accuracy
   by only 0.002, so it is the least useful of the candidates.
5. **Regularization** - sweeping the inverse regularization strength `C` down to
   `1e-3` showed that strong regularization hurts; `C = 0.001` was the best.

#### What I learned

- **Correlation between features ≠ usefulness.** `lead_score` correlates heavily
  with `interaction_count`, which inflates their apparent importance.
- **Mutual information is a cheap, model-free feature ranking** for categorical
  variables and matched intuition here.
- **Accuracy alone is a weak metric.** A classifier can look "65% right" while
  ignoring class balance; the next module (evaluation) addresses this with ROC AUC,
  precision, and recall.

#### Answers

| # | Answer |
|---|--------|
| 1 | `technology` |
| 2 | `interaction_count` & `lead_score` |
| 3 | `lead_source` |
| 4 | `0.65` |
| 5 | `number_of_courses_viewed` |
| 6 | `0.001` |

Repository and executed notebook: see the links above.

---

## LinkedIn post draft

> ✅ ML Zoomcamp 2026 - Homework 3 done.
>
> This week I moved from regression to **classification**: predicting whether a
> marketing lead converts. Highlights:
>
> • Cleaned + split the lead-scoring dataset (5000 rows, 60/20/20).
> • `interaction_count` and `lead_score` are ~92% correlated → nearly redundant.
> • Mutual information picked `lead_source` as the most informative categorical feature.
> • One-hot + logistic regression → **0.65 validation accuracy**.
> • Feature elimination flagged `number_of_courses_viewed` as the least useful feature.
> • Best regularization: `C = 0.001`.
>
> Code + executed notebook 👇
> #MLZoomcamp #MachineLearning #DataScience #Python #LearningInPublic

## X / Twitter thread draft

> 🧵 ML Zoomcamp HW3 - classification with lead scoring.
>
> 1/ Data: 5000 leads, target = `converted`. Cleaned missings (cats → NA, nums → 0) and split 60/20/20.
>
> 2/ Correlation: `interaction_count` ↔ `lead_score` = 0.92 → almost the same signal.
>
> 3/ Mutual info → `lead_source` wins (0.03).
>
> 4/ One-hot + liblinear logistic regression → 0.65 val accuracy.
>
> 5/ Eliminating `number_of_courses_viewed` barely changed accuracy (Δ = 0.002) → least useful.
>
> 6/ Best C = 0.001. Full notebook 👇 #MLZoomcamp
