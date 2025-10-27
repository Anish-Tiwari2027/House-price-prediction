# house_price_model.py

import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
import joblib  # used to save & load models

# 🏠 Step 1: Create some training data
data = {
    'Area': [1000, 1500, 1800, 2400, 3000, 3500],
    'Bedrooms': [2, 3, 4, 3, 4, 5],
    'Age': [10, 5, 5, 2, 1, 4],
    'Price': [30, 50, 70, 85, 110, 150]  # price in lakhs
}
df = pd.DataFrame(data)

# 🧩 Step 2: Split into features (X) and target (y)
X = df[['Area', 'Bedrooms', 'Age']]
y = df['Price']

# ⚙️ Step 3: Create and train KNN model
model = KNeighborsRegressor(n_neighbors=3)
model.fit(X, y)

# 💾 Step 4: Save the trained model to a file
joblib.dump(model, "house_price_model.pkl")

print("✅ Model trained and saved successfully!")
