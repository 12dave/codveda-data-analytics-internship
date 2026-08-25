import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("1__iris.csv")
print("Loaded", len(df), "rows")

print("\n--- Summary Statistics ---")
print(df.describe())

print("\n--- Most common species (mode) ---")
print(df["species"].mode()[0])

plt.figure()
df["petal_length"].hist(bins=20)
plt.title("Distribution of Petal Length")
plt.xlabel("Petal Length (cm)")
plt.ylabel("Count")
plt.savefig("histogram_petal_length.png")
plt.close()
print("\nSaved histogram_petal_length.png")

plt.figure()
sns.boxplot(x="species", y="petal_length", data=df)
plt.title("Petal Length by Species")
plt.savefig("boxplot_petal_length.png")
plt.close()
print("Saved boxplot_petal_length.png")

plt.figure()
sns.scatterplot(x="sepal_length", y="petal_length", hue="species", data=df)
plt.title("Sepal Length vs Petal Length")
plt.savefig("scatter_sepal_vs_petal.png")
plt.close()
print("Saved scatter_sepal_vs_petal.png")

print("\n--- Correlation Matrix ---")
print(df.corr(numeric_only=True))