import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample data (House size vs Price)
data = {
    'Size': [500, 800, 1000, 1200, 1500, 1800, 2000],
    'Price': [50, 80, 100, 120, 150, 180, 200]
}

df = pd.DataFrame(data)

# Input and output
X = df[['Size']]
y = df['Price']

# Model
model = LinearRegression()
model.fit(X, y)

# Predict new house prices
new_sizes = [[2200], [2500]]
predictions = model.predict(new_sizes)

print("Predicted Prices:", predictions)

# Graph
plt.scatter(df['Size'], df['Price'])
plt.plot(df['Size'], model.predict(X))
plt.xlabel("Size")
plt.ylabel("Price")
plt.title("House Price Prediction")
plt.show()