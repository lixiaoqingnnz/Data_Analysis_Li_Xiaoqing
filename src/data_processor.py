
import pandas as pd
import os
import logging

def sales_volume_value_year(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute annual total sales volume, total sales value, and year-over-year
    growth rates for both metrics.

    Args:
        df (pd.DataFrame):
            Cleaned DataFrame containing at least:
            ['year', 'price_usd', 'sales_volume'].

    Returns:
        pd.DataFrame:
            df_sales_volume_value_year containing:
                - total_sales_volume
                - total_sales_value
                - volume_growth (%)
                - value_growth (%)
            indexed by year.
    """

    df = df.copy()

    # ---- 1. Compute row-level revenue ----
    df["sales_value_usd"] = df["price_usd"] * df["sales_volume"]

    # ---- 2. Aggregate by year ----
    df_sales_volume_value_year = df.groupby("year").agg(
        total_sales_volume=("sales_volume", "sum"),
        total_sales_value=("sales_value_usd", "sum"),
    )

    # ---- 3. Compute YoY growth (%), fill first year as 0 ----
    df_sales_volume_value_year["volume_growth"] = (
        df_sales_volume_value_year["total_sales_volume"].pct_change() * 100
    )
    df_sales_volume_value_year["value_growth"] = (
        df_sales_volume_value_year["total_sales_value"].pct_change() * 100
    )

    df_sales_volume_value_year.fillna(
        {"volume_growth": 0, "value_growth": 0},
        inplace=True
    )

    return df_sales_volume_value_year



def sales_region_year(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate total sales volume by region and year, then pivot the table
    into a Region × Year matrix for radar-chart visualization.

    Args:
        df (pd.DataFrame):
            Input DataFrame containing:
            ['region', 'year', 'sales_volume'].

    Returns:
        pd.DataFrame:
            df_region_sales_by_year where each row represents a region
            and each column represents a year, with aggregated sales volume.
    """

    df = df.copy()

    # ---- 1. Aggregate sales volume by region and year ----
    df_grouped = (
        df.groupby(["region", "year"], observed=False)["sales_volume"]
          .sum()
          .reset_index()
    )

    # ---- 2. Pivot into Region × Year matrix ----
    df_region_sales_by_year = (
        df_grouped.pivot(index="region", columns="year", values="sales_volume")
                  .fillna(0)
    )

    return df_region_sales_by_year




def total_sales_volume_region(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates and aggregates the total sales volume 
    for each region across the years (e.g., 2020-2024).

    Args:
        df (pd.DataFrame): The input DataFrame containing sales data. 
                          It is expected to have 'region', 'sales_volume' columns.

    Returns:
        pd.DataFrame: A DataFrame indexed by Region, containing 'total_sales_volume' 
                      and 'total_revenue_usd' columns.
    """
    
    df_total_sales_region = df.groupby(['region'])['sales_volume'].sum().reset_index()
    df_total_sales_region = df_total_sales_region.rename(columns={"sales_volume": "sales"})
    
    return df_total_sales_region

def total_sales_value_region(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate total sales volume and total sales value for each region.

    Args:
        df (pd.DataFrame):
            Original dataset containing at least:
            ['region', 'price_usd', 'sales_volume'].

    Returns:
        pd.DataFrame:
            df_total_sales_region containing:
            ['region', 'total_sales', 'total_sales_value'].
    """

    # ---- 1. Compute per-row revenue (price × volume) ----
    df = df.copy()
    df["sales_value_usd"] = df["price_usd"] * df["sales_volume"]

    # ---- 2. Group by region and aggregate total volume + value ----
    df_total_sales_region = (
        df.groupby("region", observed=False)
          .agg(
              total_sales=("sales_volume", "sum"),
              total_sales_value=("sales_value_usd", "sum")
          )
          .reset_index()
    )

    return df_total_sales_region



def region_price(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare a region-level price dataset for downstream price distribution analysis.

    Args:
        df (pd.DataFrame): Original dataset containing at least
                           ['region', 'price_usd'] columns.

    Returns:
        pd.DataFrame: df_region_price including only rows where price_usd > 0,
                      with columns ['region', 'price_usd'].
    """

    # ---- 1. Filter valid price records (> 0) ----
    df_filtered = df[df["price_usd"] > 0]

    # ---- 2. Select columns for price-by-region analysis ----
    df_region_price = df_filtered.loc[:, ["region", "price_usd"]]

    return df_region_price



def model_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate total sales volume for each car model and rank them from
    highest to lowest.

    Args:
        df (pd.DataFrame): Input dataset containing 'model' and 'sales_volume'
                           columns.

    Returns:
        pd.DataFrame: df_model_sales containing:
                      'model' | 'total_sales_volume'
                      sorted in descending order of total_sales_volume.
    """

    # ---- 1. Aggregate total sales by model ----
    model_sales_series = (
        df.groupby("model")["sales_volume"]
          .sum()
          .sort_values(ascending=False)
    )

    # ---- 2. Convert Series to DataFrame ----
    df_model_sales = model_sales_series.reset_index(name="total_sales_volume")

    return df_model_sales





def mileage_sales(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process mileage information and assign each record to a mileage group
    (Low / Mid / High). This function only processes data and does not
    generate any summaries.

    Args:
        df (pd.DataFrame): Input dataset containing 'mileage_km' and
                           'sales_volume' columns.

    Returns:
        pd.DataFrame: df_mileage_sales containing:
                      'mileage_group' | 'total_sales' | 'avg_mileage'
    """

    # ---- 1. Define mileage bins and categorize each record ----
    bins = [0, 40000, 120000, df["mileage_km"].max()]
    labels = ["Low Mileage", "Mid Mileage", "High Mileage"]

    df_proc = df.copy()  # avoid modifying original
    df_proc["mileage_group"] = pd.cut(
        df_proc["mileage_km"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    # ---- 2. Aggregate sales and average mileage for each group ----
    df_mileage_sales = (
        df_proc.groupby("mileage_group", observed=False)
               .agg(
                   total_sales=("sales_volume", "sum"),
                   avg_mileage=("mileage_km", "mean"),
               )
               .reset_index()
    )

    # ---- 3. Enforce ordering of mileage groups ----
    df_mileage_sales["mileage_group"] = pd.Categorical(
        df_mileage_sales["mileage_group"],
        categories=labels,
        ordered=True
    )
    df_mileage_sales = df_mileage_sales.sort_values("mileage_group")

    return df_mileage_sales



def analyze_fuel_type_by_region(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate total sales of each fuel type within each region
    across all years (year is not differentiated).

    Args:
        df (pd.DataFrame): Input dataset containing at least
                           'region', 'fuel_type', and 'sales_volume' columns.

    Returns:
        pd.DataFrame: df_fuel_type_region containing:
                      'region' | 'fuel_type' | 'total_sales'
    """

    # ---- 1. Copy dataset to prevent modification of original ----
    df_proc = df.copy()

    # ---- 2. Aggregate total sales by region and fuel_type ----
    df_region_sales = (
        df_proc.groupby(["region", "fuel_type"], observed=False)["sales_volume"]
               .sum()
               .reset_index(name="total_sales")
    )

    # ---- 3. Sort for structured output ----
    df_fuel_type_region = df_region_sales.sort_values(["region", "fuel_type"])

    return df_fuel_type_region



def analyze_fuel_type_trend(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze annual sales and market share trends for each fuel type.

    Args:
        df (pd.DataFrame): Input dataset containing at least
                           'year', 'fuel_type', and 'sales_volume' columns.

    Returns:
        pd.DataFrame: df_fuel_trend containing:
                      'year' | 'fuel_type' | 'total_sales' | 'sales_share'
    """

    # ---- 1. Copy input and ensure year is numeric ----
    df_proc = df.copy()
    df_proc["year"] = pd.to_numeric(df_proc["year"], errors="coerce").astype("Int64")

    # ---- 2. Aggregate total sales by year and fuel_type ----
    df_trend = (
        df_proc.groupby(["year", "fuel_type"], observed=False)["sales_volume"]
               .sum()
               .reset_index(name="total_sales")
    )

    # ---- 3. Compute annual total sales for market share calculation ----
    total_sales_year = df_trend.groupby("year")["total_sales"].transform("sum")
    df_trend["sales_share"] = df_trend["total_sales"] / total_sales_year

    # ---- 4. Sort for chronological and categorical readability ----
    df_fuel_trend = df_trend.sort_values(["year", "fuel_type"])

    return df_fuel_trend



def analyze_transmission_trend(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute annual sales trends for each transmission type.

    Args:
        df (pd.DataFrame): Input dataset containing at least
                           'year', 'transmission', and 'sales_volume' columns.

    Returns:
        pd.DataFrame: df_transmission_trend containing:
                      'year' | 'transmission' | 'total_sales'
    """

    # ---- 1. Copy input to avoid modifying original ----
    df_proc = df.copy()

    # ---- 2. Aggregate sales by year and transmission ----
    df_trend = (
        df_proc.groupby(["year", "transmission"], observed=False)["sales_volume"]
               .sum()
               .reset_index(name="total_sales")
    )

    # ---- 3. Sort for chronological trend analysis ----
    df_transmission_trend = df_trend.sort_values(["year", "transmission"])

    return df_transmission_trend


def analyze_color_sales(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate total sales by vehicle color.

    Args:
        df (pd.DataFrame): Input dataset containing at least
                           'color' and 'sales_volume' columns.

    Returns:
        pd.DataFrame: df_color_sales containing:
                      'color' | 'total_sales'
    """

    # ---- 1. Copy to avoid modifying original data ----
    df_proc = df.copy()

    # ---- 2. Aggregate sales by color ----
    df_color_sales = (
        df_proc.groupby("color", observed=False)["sales_volume"]
               .sum()
               .reset_index(name="total_sales")
               .sort_values("total_sales", ascending=False)
    )

    return df_color_sales


def analyze_engine_size_sales(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate total sales by engine displacement ranges.
    The engine size (in liters) is bucketed into five predefined bins:
        <=1.5L, 1.6–2.0L, 2.1–3.0L, 3.1–4.0L, >4.0L

    Args:
        df (pd.DataFrame): Input dataset containing at least
                           'engine_size_l' and 'sales_volume' columns.

    Returns:
        pd.DataFrame: df_engine_size_sales with columns:
                      'engine_bin' | 'total_sales'
    """

    # ---- 1. Copy to avoid modifying original data ----
    df_proc = df.copy()

    # ---- 2. Define engine displacement bins ----
    bins = [0, 1.5, 2.0, 3.0, 4.0, float("inf")]
    labels = [
        "<=1.5L",
        "1.6–2.0L",
        "2.1–3.0L",
        "3.1–4.0L",
        ">4.0L"
    ]

    # ---- 3. Bucket engine sizes into discrete categories ----
    df_proc["engine_bin"] = pd.cut(
        df_proc["engine_size_l"],
        bins=bins,
        labels=labels,
        right=True
    )

    # ---- 4. Aggregate total sales per engine-size bin ----
    df_engine_size_sales = (
        df_proc.groupby("engine_bin", observed=False)["sales_volume"]
               .sum()
               .reset_index(name="total_sales")
               .sort_values("total_sales", ascending=False)
    )

    return df_engine_size_sales
