
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Set a seed for reproducibility
np.random.seed(42)

# Data generation
X = 6 * np.random.rand(100, 1) - 3
y = 0.5 * X**2 + 1.5 * X + 2 + np.random.randn(100, 1)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Polynomial features
poly = PolynomialFeatures(degree=2, include_bias=True)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Model training
model = LinearRegression()
model.fit(X_train_poly, y_train)

# Create new data for plotting the curve
X_new = np.linspace(-3, 3, 200).reshape(200, 1)
X_new_poly = poly.transform(X_new)
y_new = model.predict(X_new_poly)

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(X_new, y_new, "r-", linewidth=2, label="Polynomial Regression")
plt.plot(X_train, y_train, "b.", label="Training points")
plt.plot(X_test, y_test, "g.", label="Testing points")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Polynomial Regression")
plt.legend()

# Save the plot
plt.savefig("polynomial_regression.png")

# Close the plot to free up memory
plt.close()

print("Plot saved as polynomial_regression.png")
