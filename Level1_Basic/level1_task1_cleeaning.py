import pandas as pd

df = pd.read_csv("3__Sentiment_dataset.csv")

df = df.drop(columns=["Unnamed: 0", "Unnamed: 0.1"])

text_columns = df.select_dtypes(include="object").columns
for col in text_columns:
    df[col] = df[col].str.strip()

df["Sentiment"] = df["Sentiment"].str.title()
df["Platform"] = df["Platform"].str.title()

df["Timestamp"] = pd.to_datetime(df["Timestamp"])

missing_before = df.isnull().sum().sum()
num_cols = df.select_dtypes(include="number").columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())
df[text_columns] = df[text_columns].fillna("Unknown")
missing_after = df.isnull().sum().sum()

dupes_before = df.duplicated().sum()
df = df.drop_duplicates()

df.to_csv("sentiment_cleaned.csv", index=False)