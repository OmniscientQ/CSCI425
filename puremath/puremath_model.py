import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

original_csv = "./resampled_validated_with_durations.csv"
summary_csv = "./vocal_tract_summary.csv"

df_meta = pd.read_csv(original_csv)
df_summary = pd.read_csv(summary_csv)

df = pd.merge(df_meta, df_summary, on="filename")

df = df[df["gender"] != "non-binary"]
df["gender"] = df["gender"].map({"male_masculine": 0, "female_feminine": 1})

X = df[["mean_L", "median_L", "min_L", "max_L", "std_L", "mean_f2_f1",
        "std_f2_f1", "mean_f1", "mean_f2", "std_f1", "std_f2"]]
y = df["gender"]

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_poly, y, test_size=0.2, stratify=y, random_state=42
)

scaler = RobustScaler()
# scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
feature_names = poly.get_feature_names_out(X.columns)

"""
# Logistic Regresssion attempt - 0.69 accuracy
clf = LogisticRegression()
clf.fit(X_train_scaled, y_train)

coefficients = clf.coef_[0]

y_pred = clf.predict(X_test_scaled)


# Random Forest attempt - 0.71 accuracy
clf = RandomForestClassifier(n_estimators=100, random_state=42)

clf.fit(X_train_scaled, y_train)

y_pred_rf = clf.predict(X_test_scaled)
"""

# XG Boost classifier - 0.71 accuracy
clf = xgb.XGBClassifier(
    n_estimators=100,   # Number of boosting rounds
    max_depth=3,        # Maximum depth of the trees
    learning_rate=0.1,  # Shrinkage step
    random_state=42,    # For reproducibility
    eval_metric="mlogloss"  # For multi-class classification (log-loss)
)

clf.fit(X_train_scaled, y_train)

y_pred = clf.predict(X_test_scaled)


# Output section
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

importances = clf.feature_importances_
importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values(by="importance", ascending=False)

print("\nTop 20 Most Important Features (XGBoost):")
print(importance_df.head(20).to_string(index=False))