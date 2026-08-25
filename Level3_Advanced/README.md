# Level 3 – Advanced

## Task 1: Predictive Modeling (Classification)
**Dataset:** Customer Churn (3,333 customers, pre-split 80/20 train/test)

Preprocessed categorical fields, trained and compared Logistic Regression and Random Forest classifiers, then tuned the Random Forest with GridSearchCV.

**Files:** level3_task1_simple.py

**Result:**

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.853 | 0.459 | 0.179 | 0.258 |
| Random Forest | 0.957 | 0.946 | 0.737 | 0.828 |
| Random Forest (tuned) | 0.954 | 0.921 | 0.737 | 0.819 |

Random Forest substantially outperformed Logistic Regression, especially on recall — critical for catching customers likely to churn. Grid search confirmed the default settings were already near-optimal.

---

## Task 2: Building Dashboards with Power BI
**Dataset:** Customer Churn (combined, 3,333 rows)

Built an interactive dashboard with KPI cards (total customers, churn rate, average service calls), a churn split chart, and a churn-by-plan-type chart, with slicers for International plan and Voice mail plan.

**Files:** Churn_Dashboard.xlsx (data + working formulas/charts used to build the dashboard), churn_for_powerbi.csv (source data for Power BI import)

**Result:** Churn rate is 14.5% overall; customers with an international plan churn at a disproportionately higher rate relative to group size.

## How to Run
### Task 1
pip install pandas scikit-learn
python3 level3_task1_simple.py

### Task 2
Open Churn_Dashboard.xlsx to view working charts/KPIs directly, or import churn_for_powerbi.csv into Power BI Desktop to rebuild the interactive dashboard.
