import pandas as pd
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def prep_data():
    original_csv = "./resampled_validated_with_durations.csv"
    summary_csv = "./vocal_tract_summary.csv"

    df_meta = pd.read_csv(original_csv)
    df_summary = pd.read_csv(summary_csv)

    df = pd.merge(df_meta, df_summary, on="filename")

    df = df[df["gender"] != "non-binary"]
    df["gender"] = df["gender"].map({"male_masculine": 0, "female_feminine": 1})

    return df

# VIZUALIZATIONS

def runVis(df):
    """
    # Box Plot
    sns.boxplot(x="gender", y="mean_L", data=df)
    plt.title("Mean Vocal Tract Length by Gender")
    plt.show()
    """

    """
    # PCA
    features = ["mean_L", "median_L", "min_L", "max_L", "std_L", 
                "mean_f1", "std_f1", "mean_f2", "std_f2", 
                "mean_f2_f1", "std_f2_f1"]

    X = df[features]
    y = df["gender"]

    X_scaled = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame(data=components, columns=["PC1", "PC2"])
    pca_df["gender"] = y.values

    sns.scatterplot(x="PC1", y="PC2", hue="gender", data=pca_df)
    plt.title("PCA of Acoustic Features by Gender")
    plt.show()
    """

    top_features = ["mean_L", "mean_f2_f1", "mean_f2"]
    sns.pairplot(df[top_features + ["gender"]], hue="gender", diag_kind="kde")
    plt.suptitle("Top Feature Relationships by Gender", y=1.02)
    plt.show()



# MODEL TRAINING
def runModels(df):
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

    X_train_scaled = scaler.fit_transform(X_train)q
    X_test_scaled = scaler.transform(X_test)
    feature_names = poly.get_feature_names_out(X.columns)

    """    
    # Logistic Regresssion attempt - 0.69 accuracy
    clf = LogisticRegression()
    clf.fit(X_train_scaled, y_train)

    coefficients = clf.coef_[0]

    y_pred = clf.predict(X_test_scaled)
    """

    """
    # Random Forest attempt - 0.71 accuracy
    clf = RandomForestClassifier(n_estimators=100, random_state=42)

    clf.fit(X_train_scaled, y_train)

    y_pred = clf.predict(X_test_scaled)
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

    y_probs = clf.predict_proba


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

    clf.save_model("XGBoost.model")


if __name__ == "__main__":
    df = prep_data()
    # runVis(df)
    runModels(df)