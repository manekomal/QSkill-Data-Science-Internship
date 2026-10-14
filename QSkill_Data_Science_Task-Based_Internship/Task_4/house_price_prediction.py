# 1. Import Pandas
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error, 
    root_mean_squared_error, 
    r2_score
)

# 2. Load Dataset
df = pd.read_csv("Task_4/Housing.csv")

# 3. Check Dataset

print(df.head())
print(df.info())

# 4. Handle Missing Vales
missing_values = df.isnull().sum()

print("Missing Values:")
print(missing_values)

# 5. Handle Outliers
numeric_features = [
    "area", "bedrooms", "bathrooms", "stories", "parking"
]

for column in numeric_features:

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    df[column] = df[column].clip(
        lower_limit,
        upper_limit
    )

print("Outliers handled successfully.")

# 6. Exploaratory Data Analysis & Visualization

print("\nStatistical Summary:")
print(df.describe())

plt.figure(figsize=(8, 5))
sns.histplot(df["price"], bins=30, kde=True)
plt.xlabel("House Price")
plt.ylabel("Number od Houses")
plt.title("House Price Distribution")
plt.savefig("Task_4/house_price_distribution.png", dpi=300, bbox_inches="tight")
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="area", y="price")
plt.xlabel("Area")
plt.ylabel("House Price")
plt.title("Area vs House Price")
plt.savefig("Task_4/area_vs_house_price.png", dpi=300, bbox_inches="tight")
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="bedrooms", y="price")
plt.xlabel("Number of Bedrooms")
plt.ylabel("House Price")
plt.title("Bedrooms vs House Price")
plt.savefig("Task_4/bedrooms_vs_house_price.png", dpi=300, bbox_inches="tight")
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="furnishingstatus", y="price")
plt.xlabel("Furnishing Status")
plt.ylabel("House Price")
plt.title("Furnising Status vs Hose Price")
plt.savefig("Task_4/furnishing_status_vs_house_price.png", dpi=300, bbox_inches="tight")
plt.show()

# 7. Encode Categorical Variables

df = pd.get_dummies(
    df, columns=[
        "mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea", "furnishingstatus"
    ],
    drop_first=True,
    dtype=int
)

print("\nEncoded Date:")
print(df.head())

# 8. Select Meaningful Features
x = df.drop("price", axis=1)
y = df["price"]

print("\nFeatures:")
print(x.columns)

print("\nTarget:")
print(y.name)

# 9. Prepare Data for Machine Learning
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2,
    random_state=42
)

print("\nTraining Data Shape:")
print(x_train.shape)

print("\nTesting Data Shape:")
print(x_test.shape)

# 10. Build Regression Model
linear_model = LinearRegression()

random_forest_model = RandomForestRegressor(
    n_estimators=100, random_state=42
)

linear_model.fit(x_train, y_train)

random_forest_model.fit(x_train, y_train)

print("Regression model trained successfully.")

# 11. Predict HOuse Prices
linear_predictions = linear_model.predict(x_test)

random_forest_predictions = random_forest_model.predict(x_test)

print("House price predictions generated successfully.")


# 12. Evaluate Regression Models
# Linear Regression Evaluation
linear_mae = mean_absolute_error(
    y_test, linear_predictions
)

linear_rmse = root_mean_squared_error(
    y_test, linear_predictions
)

linear_r2 = r2_score(
    y_test, linear_predictions
)

# Random Forest Regression Evaluation
random_forest_mae = mean_absolute_error(
    y_test, random_forest_predictions
)

random_forest_rmse = root_mean_squared_error(
    y_test, random_forest_predictions
)

random_forest_r2 = r2_score(
    y_test, random_forest_predictions
)

# Compare Models
model_comparison = pd.DataFrame({
    "Model": [
        "Linear Regression", "Random Forest Regression"
    ],
    "MAE": [
        linear_mae, random_forest_mae
    ],
    "RMSE" : [
        linear_rmse, random_forest_rmse
    ],
    "R2": [
        linear_r2, random_forest_r2
    ]
})

print("\nModel Comparison:")
print(model_comparison.round(2).to_string(index=False))

# 13. Analyze Important Features
feature_importance = pd.DataFrame({
    "Feature": x.columns,
    "Importance": random_forest_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by ="Importance", ascending=False
)

print("\nImportant Features:")
print(feature_importance.round(4).to_string(index=False))

plt.figure(figsize=(10, 6))

sns.barplot(
    data=feature_importance,
    x="Importance",
    y="Feature"
)

plt.title("Improtant Features for House Price Prediction")
plt.xlabel("Feature Importance")
plt.ylabel("Property Features")
plt.tight_layout()
plt.savefig("Task_4/feature_importance.png", dpi=300, bbox_inches="tight")
plt.show()

# 14. Final Observations
print("\nFinal Observations:")

if (
    linear_mae < random_forest_mae
    and linear_rmse < random_forest_rmse
    and linear_r2 > random_forest_r2
):
    print("Linear Regression performed better than Random Forest Regression.")
else:
    print("Model performance differs across evaluation metrics.")

print(
    f"Linear Regression: MAE = {linear_mae:.2f}, "
    f"RMSE = {linear_rmse:.2f}, R2 = {linear_r2:.2f}"
)

print(
    f"Random Forest Regression: MAE = {random_forest_mae:.2f}, "
    f"RMSE = {random_forest_rmse:.2f}, R2 = {random_forest_r2:.2f}"
)

top_feature = feature_importance.iloc[0]

print(
    f"Most important feature: {top_feature['Feature']}"
)

print(
    "Area has the highest feature importance in the Random Forest model."
)

print(
    "The results are based on the current training and testing split."
)
    