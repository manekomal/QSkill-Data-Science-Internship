# 1. Import Libraries

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 2. Load Dataset

df = pd.read_csv("Task_2/StudentPerformanceFactors.csv")

print("First 5 rows:")
print(df.head())


# 3. Check Missing Values and Duplicate Records

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())


# 4. Handle Missing Values

df["Teacher_Quality"] = df["Teacher_Quality"].fillna(
    df["Teacher_Quality"].mode()[0]
)

df["Parental_Education_Level"] = df["Parental_Education_Level"].fillna(
    df["Parental_Education_Level"].mode()[0]
)

df["Distance_from_Home"] = df["Distance_from_Home"].fillna(
    df["Distance_from_Home"].mode()[0]
)

print("\nMissing values after cleaning:")
print(df.isnull().sum())


# 5. Study Hours and Exam Score Analysis

print("\nAverage exam score by study hours:")
print(df.groupby("Hours_Studied")["Exam_Score"].mean())


# 6. Attendance and Exam Score Analysis

print("\nAverage exam score by attendance:")
print(df.groupby("Attendance")["Exam_Score"].mean())


# 7. Previous Scores and Exam Score Analysis

print("\nAverage exam score by previous scores:")
print(df.groupby("Previous_Scores")["Exam_Score"].mean())


# 8. Parental Education and Exam Score Analysis

print("\nAverage exam score by parental education level:")
print(df.groupby("Parental_Education_Level")["Exam_Score"].mean())


# 9. Internet Access and Exam Score Analysis

print("\nAverage exam score by internet access:")
print(df.groupby("Internet_Access")["Exam_Score"].mean())


# 10. Extracurricular Activities and Exam Score Analysis

print("\nAverage exam score by extracurricular activities:")
print(df.groupby("Extracurricular_Activities")["Exam_Score"].mean())


# 11. Study Hours vs Exam Score Visualization

plt.figure(figsize=(8, 5))

plt.scatter(
    df["Hours_Studied"],
    df["Exam_Score"]
)

plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.title("Study Hours vs Exam Score")
plt.savefig("Task_2/study_hours.png", dpi=300, bbox_inches="tight")
plt.show()


# 12. Attendance vs Exam Score Visualization

plt.figure(figsize=(8, 5))

plt.scatter(
    df["Attendance"],
    df["Exam_Score"]
)

plt.xlabel("Attendance")
plt.ylabel("Exam Score")
plt.title("Attendance vs Exam Score")
plt.savefig("Task_2/attendance.png", dpi=300, bbox_inches="tight")
plt.show()


# 13. Previous Scores vs Exam Score Visualization

plt.figure(figsize=(8, 5))

plt.scatter(
    df["Previous_Scores"],
    df["Exam_Score"]
)

plt.xlabel("Previous Scores")
plt.ylabel("Exam Score")
plt.title("Previous Scores vs Exam Score")
plt.savefig("Task_2/previous_scores.png", dpi=300, bbox_inches="tight")
plt.show()


# 14. Parental Education vs Exam Score

plt.figure(figsize=(8, 5))

sns.boxplot(
    x="Parental_Education_Level",
    y="Exam_Score",
    data=df
)

plt.xlabel("Parental Education Level")
plt.ylabel("Exam Score")
plt.title("Parental Education Level vs Exam Score")
plt.savefig("Task_2/parental_education.png", dpi=300, bbox_inches="tight")
plt.show()


# 15. Internet Access vs Exam Score

plt.figure(figsize=(6, 5))

sns.boxplot(
    x="Internet_Access",
    y="Exam_Score",
    data=df
)

plt.xlabel("Internet Access")
plt.ylabel("Exam Score")
plt.title("Internet Access vs Exam Score")
plt.savefig("Task_2/internet_access.png", dpi=300, bbox_inches="tight")
plt.show()


# 16. Extracurricular Activities vs Exam Score

plt.figure(figsize=(6, 5))

sns.boxplot(
    x="Extracurricular_Activities",
    y="Exam_Score",
    data=df
)

plt.xlabel("Extracurricular Activities")
plt.ylabel("Exam Score")
plt.title("Extracurricular Activities vs Exam Score")
plt.savefig("Task_2/extracurricular_activities.png", dpi=300, bbox_inches="tight")
plt.show()


# 17. Correlation Heatmap

numeric_columns = df.select_dtypes(include="number")

plt.figure(figsize=(10, 7))

sns.heatmap(
    numeric_columns.corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")
plt.savefig("Task_2/correlation_heatmap.png", dpi=300, bbox_inches="tight")
plt.show()


# 18. Prepare Data for Machine Learning

X = df.drop("Exam_Score", axis=1)
y = df["Exam_Score"]


# 19. Separate Categorical and Numerical Columns

categorical_columns = X.select_dtypes(
    include="object"
).columns

numeric_columns = X.select_dtypes(
    exclude="object"
).columns


# 20. Preprocessing

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        ),
        (
            "numeric",
            "passthrough",
            numeric_columns
        )
    ]
)


# 21. Split Data into Training and Testing

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# 22. Linear Regression Model

linear_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", LinearRegression())
    ]
)

linear_model.fit(
    X_train,
    y_train
)

linear_predictions = linear_model.predict(
    X_test
)


# 23. Random Forest Regression Model

random_forest_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestRegressor(
                n_estimators=100,
                random_state=42
            )
        )
    ]
)

random_forest_model.fit(
    X_train,
    y_train
)

random_forest_predictions = random_forest_model.predict(
    X_test
)


# 24. Evaluate Linear Regression

linear_mae = mean_absolute_error(
    y_test,
    linear_predictions
)

linear_mse = mean_squared_error(
    y_test,
    linear_predictions
)

linear_r2 = r2_score(
    y_test,
    linear_predictions
)

print("\nLinear Regression Results:")
print("MAE:", linear_mae)
print("MSE:", linear_mse)
print("R2 Score:", linear_r2)


# 25. Evaluate Random Forest

rf_mae = mean_absolute_error(
    y_test,
    random_forest_predictions
)

rf_mse = mean_squared_error(
    y_test,
    random_forest_predictions
)

rf_r2 = r2_score(
    y_test,
    random_forest_predictions
)

print("\nRandom Forest Results:")
print("MAE:", rf_mae)
print("MSE:", rf_mse)
print("R2 Score:", rf_r2)


# 26. Compare Both Models

model_comparison = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Random Forest"
    ],
    "MAE": [
        linear_mae,
        rf_mae
    ],
    "MSE": [
        linear_mse,
        rf_mse
    ],
    "R2 Score": [
        linear_r2,
        rf_r2
    ]
})

print("\nModel Comparison:")
print(model_comparison)


# 27. Find Important Factors

feature_names = (
    random_forest_model
    .named_steps["preprocessor"]
    .get_feature_names_out()
)

feature_importance = (
    random_forest_model
    .named_steps["model"]
    .feature_importances_
)


# 28. Create Feature Importance Table

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": feature_importance
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop factors influencing student performance:")
print(importance_df.head(10))


# 29. Visualize Top Factors

top_features = importance_df.head(10)

plt.figure(figsize=(10, 6))

sns.barplot(
    x="Importance",
    y="Feature",
    data=top_features
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top Factors Influencing Student Performance")
plt.savefig("Task_2/top_factors.png", dpi=300, bbox_inches="tight")
plt.show()
