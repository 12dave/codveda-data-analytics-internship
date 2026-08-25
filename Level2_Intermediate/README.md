# Level 2 – Intermediate

## Task 1: Regression Analysis
**Dataset:** House Prediction Prices (506 houses, 13 features)

Built a simple linear regression model predicting house price (MEDV) from average number of rooms (RM).

**Files:** level2_task1_simple.py, regression_rooms_vs_price.png

**Result:** R-squared = 0.371, MSE = 46.14. Room count alone explains about 37% of price variation.

---

## Task 3: Clustering Analysis (K-Means)
**Dataset:** Iris (unsupervised, species label dropped)

Standardized features, used the elbow method to choose k=3, applied K-Means, and compared results against real species labels.

**Files:** level2_task3_simple.py, elbow_plot.png, kmeans_clusters.png

**Result:** K-Means perfectly separated all 50 setosa flowers with zero error. Versicolor and virginica showed natural overlap.

## How to Run
pip install pandas matplotlib scikit-learn
python3 level2_task1_simple.py
python3 level2_task3_simple.py
