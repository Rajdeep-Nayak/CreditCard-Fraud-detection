# What was wrong with the first version (and what changed)

The first version of this project was two notebooks. This page records the issues found in a review
of that code, so the reasons behind the rewrite are on the record.

| # | Issue in the legacy notebooks | Why it matters | Fix in `fraudguard` |
|---|---|---|---|
| 1 | "Precision" is computed as `cm[0][0] / (cm[1][0] + cm[0][0])`, i.e. **TN / (TN + FN)**, and "recall" as `cm[0][0] / (cm[0][1] + cm[0][0])`, i.e. **TN / (TN + FP)**. | The first is negative predictive value, the second is specificity. Both are near 1.0 on data that is 99.8 % legitimate, so the numbers look great and say nothing about fraud detection. | `metrics.py` reports fraud-class precision, recall, F1 and PR-AUC. A unit test pins the formulas. |
| 2 | Weighted-average F1 (~0.99) is quoted as the headline. | Dominated by the majority class. | PR-AUC and fraud-class metrics only. |
| 3 | The write-up says recall is prioritised "because false positives are worse". | Prioritising recall means *missed frauds* are worse. The reasoning and the metric contradict each other. | The threshold is chosen from explicit costs (`threshold.*` in the config). |
| 4 | Logistic regression is fitted on raw `Amount` (0 to ~25 000) and `Time`. | Unscaled features are the usual root cause of convergence warnings and then a larger `max_iter`. | Scaling lives inside the sklearn Pipeline (`features.py`). |
| 5 | Random, non-stratified splits; the undersampling notebook builds its balanced subsample from the full dataframe. | Fraud counts in the test set vary by chance, and a subsample built before the split can put test rows into training. Worth checking in the original. | Time-based split (train on the past, test on the future) or stratified, plus a separate validation slice for the threshold. |
| 6 | Exact duplicate rows are kept (the Kaggle file has about a thousand). | The same transaction can land in both train and test. | `data.drop_duplicates: true`. |
| 7 | Everything is notebook-only. | Not reproducible, not testable, not deployable. | Package, config file, CLI, tests, saved artifacts. |

The README and the notebook text also disagree on how many frauds were caught (11 missed vs. 91 of 101 caught).
Single numbers like that come from one random split and shouldn't be treated as precise.
