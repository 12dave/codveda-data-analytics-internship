import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

column_names = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS",
                 "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]
df = pd.read_csv("4__house_Prediction_Data_Set.csv", sep=r"\s+", names=column_names)
print("Loaded", len(df), "houses")

X = df[["RM"]]
y = df["MEDV"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training rows:", len(X_train), "| Testing rows:", len(X_test))

model = LinearRegression()
model.fit(X_train, y_train)

print("\nModel learned this formula:")
print(f"Price = {model.coef_[0]:.2f} * RM + {model.intercept_:.2f}")

predictions = model.predict(X_test)
r2 = r2_score(y_test, predictions)
mse = mean_squared_error(y_test, predictions)

print("\n--- Model Performance ---")
print("R-squared:", round(r2, 3), "(closer to 1 = better)")
print("Mean Squared Error:", round(mse, 2), "(lower = better)")

plt.figure()
plt.scatter(X_test, y_test, label="Actual prices")
plt.plot(X_test, predictions, color="red", label="Predicted line")
plt.xlabel("Average Number of Rooms (RM)")
plt.ylabel("House Price (MEDV, $1000s)")
plt.title("House Price vs Number of Rooms")
plt.legend()
plt.savefig("regression_rooms_vs_price.png")
plt.close()
print("\nSaved regression_rooms_vs_price.png")