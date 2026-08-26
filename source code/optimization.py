import pandas as pd


def get_factory_comparison(
    df_analysis,
    model,
    preprocessor,
    feature_columns,
    product_name,
    factories,
    region=None,
    ship_mode=None
):
    """
    Compare predicted lead time for a product
    across all available factories.
    """

    filtered = df_analysis[
        df_analysis["Product Name"] == product_name
    ].copy()

    if region is not None and region != "All":
        filtered = filtered[
            filtered["Region"] == region
        ]

    if ship_mode is not None and ship_mode != "All":
        filtered = filtered[
            filtered["Ship Mode"] == ship_mode
        ]

    if filtered.empty:
        return pd.DataFrame()

    current_factory = (
        filtered["Current Factory"]
        .mode()
        .iloc[0]
    )

    actual_current_lead_time = (
        filtered["Shipping Lead Time"].mean()
    )

    results = []

    for factory in factories:

        scenario = filtered.copy()
        scenario["Current Factory"] = factory

        X_scenario = scenario[
            feature_columns
        ].copy()

        X_encoded = (
            preprocessor.transform(X_scenario)
        )

        predicted_lead_time = (
            model.predict(X_encoded).mean()
        )

        results.append({
            "Product Name": product_name,
            "Factory": factory,
            "Current Factory": (
                "Yes"
                if factory == current_factory
                else "No"
            ),
            "Actual Current Lead Time":
                actual_current_lead_time,
            "Predicted Lead Time":
                predicted_lead_time,
            "Predicted Days Saved":
                actual_current_lead_time
                - predicted_lead_time
        })

    result = pd.DataFrame(results)

    numeric_columns = [
        "Actual Current Lead Time",
        "Predicted Lead Time",
        "Predicted Days Saved"
    ]

    result[numeric_columns] = (
        result[numeric_columns].round(2)
    )

    return result.sort_values(
        "Predicted Lead Time"
    ).reset_index(drop=True)


def get_factory_performance(df_analysis):
    """
    Return observed factory-level performance.
    """

    return (
        df_analysis
        .groupby("Current Factory")
        .agg(
            Orders=("Order ID", "count"),
            Avg_Lead_Time=("Shipping Lead Time", "mean"),
            Median_Lead_Time=("Shipping Lead Time", "median"),
            Avg_Sales=("Sales", "mean"),
            Avg_Gross_Profit=("Gross Profit", "mean"),
            Total_Units=("Units", "sum")
        )
        .round(2)
        .reset_index()
        .sort_values("Avg_Lead_Time")
        .reset_index(drop=True)
    )


def get_regional_performance(
    df_analysis,
    region=None,
    ship_mode=None
):
    """
    Return regional shipping performance.
    """

    filtered = df_analysis.copy()

    if region is not None and region != "All":
        filtered = filtered[
            filtered["Region"] == region
        ]

    if ship_mode is not None and ship_mode != "All":
        filtered = filtered[
            filtered["Ship Mode"] == ship_mode
        ]

    return (
        filtered
        .groupby("Region")
        .agg(
            Orders=("Order ID", "count"),
            Avg_Lead_Time=("Shipping Lead Time", "mean"),
            Avg_Sales=("Sales", "mean"),
            Avg_Gross_Profit=("Gross Profit", "mean"),
            Total_Units=("Units", "sum")
        )
        .round(2)
        .reset_index()
    )