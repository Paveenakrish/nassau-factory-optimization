import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor


FEATURE_COLUMNS = [
    "Product ID",
    "Division",
    "Current Factory",
    "Region",
    "Ship Mode",
    "Order Year",
    "Order Month",
    "Order Quarter",
    "Order DayOfWeek",
    "Units",
    "Sales",
    "Cost",
    "Gross Profit"
]

CATEGORICAL_FEATURES = [
    "Product ID",
    "Division",
    "Current Factory",
    "Region",
    "Ship Mode"
]

NUMERICAL_FEATURES = [
    "Order Year",
    "Order Month",
    "Order Quarter",
    "Order DayOfWeek",
    "Units",
    "Sales",
    "Cost",
    "Gross Profit"
]


def train_gradient_boosting(df_analysis):
    """
    Train the final Gradient Boosting model
    using the same configuration as the analysis notebook.
    """

    X = df_analysis[FEATURE_COLUMNS].copy()
    y = df_analysis["Shipping Lead Time"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                ),
                CATEGORICAL_FEATURES
            ),
            (
                "numerical",
                "passthrough",
                NUMERICAL_FEATURES
            )
        ]
    )

    X_train_encoded = (
        preprocessor.fit_transform(X_train)
    )

    X_test_encoded = (
        preprocessor.transform(X_test)
    )

    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )

    model.fit(
        X_train_encoded,
        y_train
    )

    return {
        "model": model,
        "preprocessor": preprocessor,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_columns": FEATURE_COLUMNS
    }


def predict_scenario(
    model,
    preprocessor,
    scenario_df,
    feature_columns
):
    """
    Predict average shipping lead time for a scenario.
    """

    X_scenario = scenario_df[
        feature_columns
    ].copy()

    X_encoded = (
        preprocessor.transform(X_scenario)
    )

    predictions = model.predict(X_encoded)

    return predictions.mean()