import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error


df = pd.read_csv('statboticsData.csv')

X = df[["rp_1_epa", "rp_2_epa", "rp_3_epa"]].values
y = df["winrate"].values

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Cross validation setup
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# --------- PART 1: Evaluate a chosen K ---------

k_value = 104
knn = KNeighborsRegressor(n_neighbors=k_value, metric="minkowski", p=2)

y_true_all = []
y_pred_all = []

for train_index, test_index in kf.split(X_scaled):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y[train_index], y[test_index]

    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    y_true_all.extend(y_test)
    y_pred_all.extend(y_pred)


y_true_all = np.array(y_true_all)
y_pred_all = np.array(y_pred_all)

# Metrics
mse = mean_squared_error(y_true_all, y_pred_all)
rmse = np.sqrt(mse)
r2 = r2_score(y_true_all, y_pred_all)

print("MSE:", mse)
print("RMSE:", rmse)
print("R^2:", r2)

# Predicted vs Actual plot
plt.figure()
plt.scatter(y_true_all, y_pred_all)
plt.xlabel("Actual Winrate")
plt.ylabel("Predicted Winrate")
plt.title("KNN Regression: Predicted vs Actual Winrate")

min_val = min(y_true_all.min(), y_pred_all.min())
max_val = max(y_true_all.max(), y_pred_all.max())
plt.plot([min_val, max_val], [min_val, max_val])

plt.show()


# --------- PART 2: Cross-validated MSE vs K ---------

k_values = range(1, 201)
mse_values = []

for k in k_values:

    knn = KNeighborsRegressor(n_neighbors=k, metric="minkowski", p=2)
    fold_mse = []

    for train_index, test_index in kf.split(X_scaled):

        X_train, X_test = X_scaled[train_index], X_scaled[test_index]
        y_train, y_test = y[train_index], y[test_index]

        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)

        fold_mse.append(mean_squared_error(y_test, y_pred))

    mse_values.append(np.mean(fold_mse))


# Find best k
best_index = np.argmin(mse_values)
best_k = list(k_values)[best_index]
best_mse = mse_values[best_index]

print("Best k:", best_k)
print("Best cross-validated MSE:", best_mse)

# Plot MSE vs K
plt.figure()
plt.plot(k_values, mse_values)
plt.scatter(best_k, best_mse)
plt.axvline(best_k, linestyle='--')

plt.xlabel("K (Number of Neighbors)")
plt.ylabel("Cross-Validated MSE")
plt.title("KNN: MSE vs K")

plt.show()

autoRP = 1.0
coralRP = 0.23
bargeRP = -0.02

test_point = [[autoRP, coralRP, bargeRP]]
predicted_winrate = knn.predict(test_point)

print(f"AutoRP: {autoRP}")
print(f"CoralRP: {coralRP}")
print(f"BargeRP: {bargeRP}")
print(f"Predicted Winrate: {predicted_winrate[0]:.3f}")