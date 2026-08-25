# Codveda Technology – Data Analytics Internship

This repository contains my completed tasks for the Codveda Technology Data
Analytics internship, covering data cleaning, exploratory analysis,
regression, clustering, classification, NLP sentiment analysis, and
dashboarding across three progressive difficulty levels.

## Project Structure



Each folder has its own README explaining the task, approach, and results,
along with the Python script, dataset, and output charts/images.

## Tools & Libraries Used
- **Python**: pandas, matplotlib, seaborn, scikit-learn, nltk, textblob, wordcloud
- **Power BI**: interactive dashboard

## Summary of Key Results

# Level 1 – Basic

## Task 1: Data Cleaning and Preprocessing
**Dataset:** Sentiment Dataset (732 social media posts)

Cleaned a raw, messy dataset by:
- Removing 2 junk index columns
- Stripping inconsistent whitespace from text fields
- Standardizing text casing (e.g. "positive" -> "Positive")
- Removing 22 duplicate rows that were hidden by whitespace differences

**Files:** level1_task1_cleeaning.py, sentiment_cleaned.csv

**Result:** 732 rows -> 710 clean, deduplicated rows.

---

## Task 2: Exploratory Data Analysis (EDA)
**Dataset:** Iris (150 flowers, 4 measurements + species)

Calculated summary statistics and visualized feature distributions and relationships across the three Iris species.

**Files:** level1_task2_simple.py, histogram_petal_length.png, boxplot_petal_length.png, scatter_sepal_vs_petal.png

**Result:** Found a strong correlation (0.87) between sepal length and petal length; species cluster clearly by petal size.

## How to Run
pip install pandas matplotlib seaborn
python3 level1_task1_cleeaning.py
python3 level1_task2_simple.py
```

## Author
David Chinaza Nnadozie| Codveda Data Analytics Intern
