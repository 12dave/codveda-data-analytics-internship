import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

df = pd.read_csv("1__iris.csv")
print("Loaded", len(df), "rows")

X = df.drop(columns=["species"])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inertia_values = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia_values.append(kmeans.inertia_)

plt.figure()
plt.plot(list(k_range), inertia_values, marker="o")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia (lower = tighter clusters)")
plt.title("Elbow Method - Choosing k")
plt.savefig("elbow_plot.png")
plt.close()
print("Saved elbow_plot.png - look for the 'bend' in the line")

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X_scaled)

print("\nHow many flowers landed in each cluster:")
print(df["cluster"].value_counts())

print("\nHow clusters compare to real species:")
print(pd.crosstab(df["cluster"], df["species"]))

plt.figure()
for cluster_id in df["cluster"].unique():
    subset = df[df["cluster"] == cluster_id]
    plt.scatter(subset["petal_length"], subset["petal_width"], label=f"Cluster {cluster_id}")
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.title("K-Means Clusters (k=3)")
plt.legend()
plt.savefig("kmeans_clusters.png")
plt.close()
print("\nSaved kmeans_clusters.png")