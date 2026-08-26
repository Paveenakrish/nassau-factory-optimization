# ================================================================
# STEP 10.21 — STREAMLIT DASHBOARD
# NASSAU FACTORY OPTIMIZATION
# ================================================================

import os
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor


# ================================================================
# PAGE CONFIGURATION
# ================================================================

st.set_page_config(
    page_title="Nassau Factory Optimization",
    page_icon="🍫",
    layout="wide"
)


# ================================================================
# TITLE
# ================================================================

st.title("🍫 Nassau Factory Optimization")
st.subheader("ML-Based Shipping Lead-Time & Factory Reallocation Dashboard")

st.caption(
    "Decision-support system using Gradient Boosting and "
    "factory what-if scenario simulation."
)


# ================================================================
# DATA LOADING
# ================================================================

DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    "data",
    "dataset.csv"
)

if not os.path.exists(DATA_PATH):
    st.error(
        f"Dataset not found:\n{DATA_PATH}"
    )
    st.stop()

df = pd.read_csv(DATA_PATH)


# ================================================================
# DATA PREPARATION
# ================================================================

df_analysis = df.copy()

df_analysis["Order Date"] = pd.to_datetime(
    df_analysis["Order Date"],
    dayfirst=True,
    errors="coerce"
)

df_analysis["Ship Date"] = pd.to_datetime(
    df_analysis["Ship Date"],
    dayfirst=True,
    errors="coerce"
)

df_analysis["Shipping Lead Time"] = (
    df_analysis["Ship Date"]
    - df_analysis["Order Date"]
).dt.days

df_analysis["Order Year"] = (
    df_analysis["Order Date"].dt.year
)

df_analysis["Order Month"] = (
    df_analysis["Order Date"].dt.month
)

df_analysis["Order Quarter"] = (
    df_analysis["Order Date"].dt.quarter
)

df_analysis["Order DayOfWeek"] = (
    df_analysis["Order Date"].dt.dayofweek
)


# ================================================================
# FACTORY MAPPING
# ================================================================

factory_mapping = {
    "Wonka Bar - Nutty Crunch Surprise": "Lot's O' Nuts",
    "Wonka Bar - Fudge Mallows": "Lot's O' Nuts",
    "Wonka Bar -Scrumdiddlyumptious": "Lot's O' Nuts",

    "Wonka Bar - Milk Chocolate": "Wicked Choccy's",
    "Wonka Bar - Triple Dazzle Caramel": "Wicked Choccy's",

    "Laffy Taffy": "Sugar Shack",
    "SweeTARTS": "Sugar Shack",
    "Nerds": "Sugar Shack",
    "Fun Dip": "Sugar Shack",
    "Fizzy Lifting Drinks": "Sugar Shack",

    "Everlasting Gobstopper": "Secret Factory",

    "Hair Toffee": "The Other Factory",

    "Lickable Wallpaper": "Secret Factory",
    "Wonka Gum": "Secret Factory",

    "Kazookles": "The Other Factory"
}

df_analysis["Current Factory"] = (
    df_analysis["Product Name"]
    .map(factory_mapping)
)

if df_analysis["Current Factory"].isna().any():
    st.error("Some products could not be mapped to a factory.")
    st.stop()


# ================================================================
# ML FEATURE SET
# ================================================================

feature_columns = [
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

categorical_features = [
    "Product ID",
    "Division",
    "Current Factory",
    "Region",
    "Ship Mode"
]

numerical_features = [
    "Order Year",
    "Order Month",
    "Order Quarter",
    "Order DayOfWeek",
    "Units",
    "Sales",
    "Cost",
    "Gross Profit"
]

X = df_analysis[feature_columns].copy()
y = df_analysis["Shipping Lead Time"].copy()


# ================================================================
# TRAIN MODEL
# ================================================================

@st.cache_resource
def train_model(data):

    X = data[feature_columns].copy()
    y = data["Shipping Lead Time"].copy()

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
                categorical_features
            ),
            (
                "numerical",
                "passthrough",
                numerical_features
            )
        ]
    )

    X_train_encoded = preprocessor.fit_transform(X_train)

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

    return model, preprocessor


gb_model, preprocessor = train_model(df_analysis)


# ================================================================
# FACTORY LIST
# ================================================================

factories = sorted(
    df_analysis["Current Factory"]
    .unique()
    .tolist()
)


# ================================================================
# SIDEBAR
# ================================================================

st.sidebar.header("Dashboard Filters")

selected_region = st.sidebar.selectbox(
    "Region",
    ["All"] + sorted(
        df_analysis["Region"].unique().tolist()
    )
)

selected_ship_mode = st.sidebar.selectbox(
    "Ship Mode",
    ["All"] + sorted(
        df_analysis["Ship Mode"].unique().tolist()
    )
)

selected_product = st.sidebar.selectbox(
    "Product",
    sorted(
        df_analysis["Product Name"]
        .unique()
        .tolist()
    )
)


# ================================================================
# FILTERED DATA
# ================================================================

filtered_df = df_analysis.copy()

if selected_region != "All":
    filtered_df = filtered_df[
        filtered_df["Region"] == selected_region
    ]

if selected_ship_mode != "All":
    filtered_df = filtered_df[
        filtered_df["Ship Mode"] == selected_ship_mode
    ]


# ================================================================
# KPI SECTION
# ================================================================

st.markdown("## 📊 Key Performance Indicators")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

kpi1.metric(
    "Total Orders",
    f"{len(filtered_df):,}"
)

kpi2.metric(
    "Avg Lead Time",
    f"{filtered_df['Shipping Lead Time'].mean():,.2f} days"
)

kpi3.metric(
    "Products",
    f"{filtered_df['Product Name'].nunique()}"
)

kpi4.metric(
    "Factories",
    f"{filtered_df['Current Factory'].nunique()}"
)


# ================================================================
# FACTORY PERFORMANCE
# ================================================================

st.markdown("## 🏭 Factory Performance")

factory_performance = (
    filtered_df
    .groupby("Current Factory")
    .agg(
        Orders=("Order ID", "count"),
        Avg_Lead_Time=("Shipping Lead Time", "mean"),
        Median_Lead_Time=("Shipping Lead Time", "median"),
        Avg_Sales=("Sales", "mean"),
        Avg_Gross_Profit=("Gross Profit", "mean")
    )
    .round(2)
    .reset_index()
    .sort_values("Avg_Lead_Time")
)

st.dataframe(
    factory_performance,
    use_container_width=True,
    hide_index=True
)


# ================================================================
# PRODUCT SHIPPING PERFORMANCE
# ================================================================

st.markdown("## 📦 Product Shipping Performance")

product_performance = (
    filtered_df
    .groupby("Product Name")
    .agg(
        Orders=("Order ID", "count"),
        Avg_Lead_Time=("Shipping Lead Time", "mean"),
        Avg_Sales=("Sales", "mean"),
        Avg_Gross_Profit=("Gross Profit", "mean"),
        Total_Units=("Units", "sum")
    )
    .round(2)
    .reset_index()
    .sort_values("Avg_Lead_Time", ascending=False)
)

st.dataframe(
    product_performance,
    use_container_width=True,
    hide_index=True
)


# ================================================================
# WHAT-IF FACTORY SIMULATOR
# ================================================================

st.markdown("## 🔬 Factory What-If Simulator")

selected_data = df_analysis[
    df_analysis["Product Name"] == selected_product
].copy()

if selected_region != "All":
    selected_data = selected_data[
        selected_data["Region"] == selected_region
    ]

if selected_ship_mode != "All":
    selected_data = selected_data[
        selected_data["Ship Mode"] == selected_ship_mode
    ]

if selected_data.empty:
    st.warning(
        "No observations available for the selected "
        "product/filter combination."
    )
else:

    current_factory = (
        selected_data["Current Factory"]
        .mode()
        .iloc[0]
    )

    actual_current_lead_time = (
        selected_data["Shipping Lead Time"].mean()
    )

    scenario_results = []

    for factory in factories:

        scenario = selected_data.copy()

        scenario["Current Factory"] = factory

        X_scenario = scenario[
            feature_columns
        ].copy()

        X_scenario_encoded = (
            preprocessor.transform(X_scenario)
        )

        predicted_lead_time = (
            gb_model
            .predict(X_scenario_encoded)
            .mean()
        )

        predicted_days_saved = (
            actual_current_lead_time
            - predicted_lead_time
        )

        scenario_results.append({
            "Factory": factory,
            "Current Factory": (
                "Yes"
                if factory == current_factory
                else "No"
            ),
            "Observed Current Lead Time":
                actual_current_lead_time,
            "Predicted Lead Time":
                predicted_lead_time,
            "Predicted Days Saved":
                predicted_days_saved
        })

    scenario_df = pd.DataFrame(
        scenario_results
    )

    scenario_df[
        [
            "Observed Current Lead Time",
            "Predicted Lead Time",
            "Predicted Days Saved"
        ]
    ] = (
        scenario_df[
            [
                "Observed Current Lead Time",
                "Predicted Lead Time",
                "Predicted Days Saved"
            ]
        ].round(2)
    )

    scenario_df = scenario_df.sort_values(
        "Predicted Lead Time"
    ).reset_index(drop=True)

    st.write(
        f"**Selected product:** {selected_product}"
    )

    st.write(
        f"**Current factory:** {current_factory}"
    )

    st.write(
        f"**Observed current lead time:** "
        f"{actual_current_lead_time:.2f} days"
    )

    st.dataframe(
        scenario_df,
        use_container_width=True,
        hide_index=True
    )

    best_factory = (
        scenario_df.iloc[0]["Factory"]
    )

    best_prediction = (
        scenario_df.iloc[0]["Predicted Lead Time"]
    )

    best_saving = (
        scenario_df.iloc[0]["Predicted Days Saved"]
    )

    if best_factory == current_factory:

        st.info(
            "The current factory has the lowest predicted "
            "lead time for this scenario. No reallocation "
            "is indicated by the model."
        )

    else:

        st.success(
            f"Model scenario: consider **{best_factory}** "
            f"as the alternative factory. "
            f"Predicted lead time: "
            f"**{best_prediction:.2f} days**, "
            f"with approximately **{best_saving:.2f} "
            f"predicted days saved**."
        )


# ================================================================
# FINAL RECOMMENDATIONS
# ================================================================

st.markdown("## 🎯 Final ML Recommendations")

candidate_products = [
    "Wonka Bar - Triple Dazzle Caramel",
    "Wonka Bar - Nutty Crunch Surprise"
]

recommendation_rows = []

for product_name in candidate_products:

    product_data = df_analysis[
        df_analysis["Product Name"] == product_name
    ].copy()

    current_factory = (
        product_data["Current Factory"]
        .mode()
        .iloc[0]
    )

    actual_lead_time = (
        product_data["Shipping Lead Time"].mean()
    )

    predictions = []

    for factory in factories:

        scenario = product_data.copy()
        scenario["Current Factory"] = factory

        X_scenario = scenario[
            feature_columns
        ].copy()

        X_scenario_encoded = (
            preprocessor.transform(X_scenario)
        )

        predicted = (
            gb_model
            .predict(X_scenario_encoded)
            .mean()
        )

        predictions.append({
            "Factory": factory,
            "Predicted Lead Time": predicted
        })

    prediction_df = pd.DataFrame(
        predictions
    )

    prediction_df = prediction_df.sort_values(
        "Predicted Lead Time"
    ).reset_index(drop=True)

    best_factory = (
        prediction_df.iloc[0]["Factory"]
    )

    best_prediction = (
        prediction_df.iloc[0]["Predicted Lead Time"]
    )

    predicted_reduction = (
        actual_lead_time
        - best_prediction
    )

    recommendation_rows.append({
        "Product": product_name,
        "Current Factory": current_factory,
        "Recommended Factory": best_factory,
        "Current Lead Time": round(
            actual_lead_time,
            2
        ),
        "Predicted New Lead Time": round(
            best_prediction,
            2
        ),
        "Predicted Days Saved": round(
            predicted_reduction,
            2
        )
    })

recommendation_df = pd.DataFrame(
    recommendation_rows
)

st.dataframe(
    recommendation_df,
    use_container_width=True,
    hide_index=True
)


# ================================================================
# DATA QUALITY NOTICE
# ================================================================

st.markdown("## ⚠️ Data Quality Notice")

st.warning(
    "The supplied dataset contains unusually long shipping "
    "lead times. Order dates span 2024–2025 while ship dates "
    "span 2026–2030, producing lead times of approximately "
    "904–1,642 days. The ML recommendations should therefore "
    "be treated as scenario-based decision support and validated "
    "operationally before implementation."
)


# ================================================================
# FOOTER
# ================================================================

st.markdown("---")

st.caption(
    "Nassau Factory Optimization | "
    "Gradient Boosting Decision-Support System"
)