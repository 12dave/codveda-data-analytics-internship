# Level 1 – Basic

## Task 1: Data Cleaning and Preprocessing
**Dataset:** Sentiment Dataset (732 social media posts)

Cleaned a raw, messy dataset by:
- Removing 2 junk index columns
- Stripping inconsistent whitespace from text fields
- Standardizing text casing (e.g. "positive" -> "Positive")
- Removing 22 duplicate rows that were hidden by whitespace differences

**Files:** task1_data_cleaning/level1_task1_simple.py, task1_data_cleaning/sentiment_cleaned.csv

**Result:** 732 rows -> 710 clean, deduplicated rows.

---

## Task 2: Exploratory Data Analysis (EDA)
**Dataset:** Iris (150 flowers, 4 measurements + species)

Calculated summary statistics and visualized feature distributions and relationships across the three Iris species.

**Files:** task2_eda/level1_task2_simple.py, task2_eda/histogram_petal_length.png, task2_eda/boxplot_petal_length.png, task2_eda/scatter_sepal_vs_petal.png

**Result:** Found a strong correlation (0.87) between sepal length and petal length; species cluster clearly by petal size.
