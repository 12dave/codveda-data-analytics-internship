import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

train_df = pd.read_csv("churn-bigml-80.csv")
test_df = pd.read_csv("churn-bigml-20.csv")
print("Training rows:", len(train_df), "| Testing rows:", len(test_df))

for df in [train_df, test_df]:
    df["International plan"] = df["International plan"].map({"Yes": 1, "No": 0})
    df["Voice mail plan"] = df["Voice mail plan"].map({"Yes": 1, "No": 0})
    df["Churn"] = df["Churn"].astype(int)

train_df = train_df.drop(columns=["State"])
test_df = test_df.drop(columns=["State"])

X_train = train_df.drop(columns=["Churn"])
y_train = train_df["Churn"]
X_test = test_df.drop(columns=["Churn"])
y_test = test_df["Churn"]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_scaled, y_train)
log_predictions = log_model.predict(X_test_scaled)

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_scaled, y_train)
rf_predictions = rf_model.predict(X_test_scaled)

def print_scores(name, y_true, y_pred):
    print(f"\n--- {name} ---")
    print("Accuracy: ", round(accuracy_score(y_true, y_pred), 3))
    print("Precision:", round(precision_score(y_true, y_pred), 3))
    print("Recall:   ", round(recall_score(y_true, y_pred), 3))
    print("F1-score: ", round(f1_score(y_true, y_pred), 3))

print_scores("Logistic Regression", y_test, log_predictions)
print_scores("Random Forest", y_test, rf_predictions)

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    scoring="f1",
    cv=3,
)
grid_search.fit(X_train_scaled, y_train)

print("\n--- Grid Search Results ---")
print("Best settings found:", grid_search.best_params_)

best_model = grid_search.best_estimator_
best_predictions = best_model.predict(X_test_scaled)
print_scores("Tuned Random Forest", y_test, best_predictions)