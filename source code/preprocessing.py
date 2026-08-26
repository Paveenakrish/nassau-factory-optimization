import pandas as pd


FACTORY_MAPPING = {
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


def prepare_analysis_dataframe(df):
    """
    Prepare the raw dataset for analysis and ML.
    """

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

    df_analysis["Current Factory"] = (
        df_analysis["Product Name"]
        .map(FACTORY_MAPPING)
    )

    if df_analysis["Current Factory"].isna().any():
        missing = (
            df_analysis.loc[
                df_analysis["Current Factory"].isna(),
                "Product Name"
            ]
            .unique()
            .tolist()
        )

        raise ValueError(
            f"Unmapped products found: {missing}"
        )

    return df_analysis