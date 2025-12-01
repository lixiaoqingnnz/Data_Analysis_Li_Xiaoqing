import warnings
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
    median_absolute_error,
)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def catboost_forecast_top3(
    df: pd.DataFrame,
) -> Tuple[Dict[str, float], float, pd.DataFrame, str, Dict]:
    """
    Train a CatBoost-based time-series style model for sales forecasting and
    generate a Top 3 segment summary comparing 2024 actuals vs. 2025 forecasts.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned row-level dataset containing at least the following columns:
        [
            'year', 'model', 'region', 'fuel_type', 'transmission', 'color',
            'engine_size_l', 'mileage_km', 'price_usd', 'sales_volume'
        ]

    Returns
    -------
    metrics_2024 : dict
        Evaluation metrics of CatBoost on the 2024 validation set
        (MAE, RMSE, MAPE, R2, and percentage-based metrics).

    total_2025 : float
        Forecasted 2025 total sales (sum across all combinations), using
        a CatBoost model retrained on all data from 2020–2024.

    combined_top3_df : pd.DataFrame
        Unified Top 3 segment summary table (display-ready, business naming),
        with the following columns:
            - Segment Category
            - Segment Name
            - 2024 Actual Sales
            - 2025 Forecast Sales
            - Growth Rate (%)
        The last row is the overall total market line, with total sales and
        overall growth rate.

    md_combined_top3 : str
        Markdown representation of combined_top3_df, suitable for direct
        insertion into a .md report and subsequent conversion to PDF
        by the main report-generation pipeline.

    analysis_summary : dict
        Structured numerical summary for LLM-based narrative generation.
        Contains raw (non-formatted) values for model performance, total
        market growth, and detailed Top 3 segment records.
    """
    # -----------------------------------------------------
    # Local feature configuration (only used within this function)
    # -----------------------------------------------------
    CAT_FEATURES = ["model", "region", "fuel_type", "transmission", "color"]
    BASE_NUM_FEATURES = ["year", "engine_size_l", "mileage_km", "price_usd"]

    # =====================================================
    # 2. Helper function definitions (internal use only)
    # =====================================================
    def load_and_aggregate_from_df(df_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate row-level transaction data to the combination level:

        Grouping keys:
            year + model + region + fuel_type + transmission + color

        Aggregations:
            - engine_size_l, mileage_km, price_usd: mean
            - sales_volume: sum

        This provides a compact time-series per segment for modeling.
        """
        group_cols = ["year", "model", "region", "fuel_type", "transmission", "color"]
        agg_df = (
            df_raw.groupby(group_cols, as_index=False)
            .agg(
                {
                    "engine_size_l": "mean",
                    "mileage_km": "mean",
                    "price_usd": "mean",
                    "sales_volume": "sum",
                }
            )
        )
        return agg_df

    def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Compute generic regression metrics and their percentage variants.

        Metrics:
            - MAE
            - RMSE
            - MedianAE
            - MAPE (percentage)
            - R2

        Percentage metrics relative to the mean of y_true:
            - MAE_pct
            - RMSE_pct
            - MedianAE_pct
        """
        mae = mean_absolute_error(y_true, y_pred)
        rmse = root_mean_squared_error(y_true, y_pred)
        med_ae = median_absolute_error(y_true, y_pred)

        mask = y_true != 0
        if mask.sum() > 0:
            mape = (np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])).mean() * 100
        else:
            mape = np.nan

        if len(y_true) > 1:
            r2 = r2_score(y_true, y_pred)
        else:
            # R2 is not meaningful with only a single observation
            r2 = np.nan

        mean_y = y_true.mean()
        mae_pct = mae / mean_y * 100 if mean_y != 0 else np.nan
        rmse_pct = rmse / mean_y * 100 if mean_y != 0 else np.nan
        med_ae_pct = med_ae / mean_y * 100 if mean_y != 0 else np.nan

        return {
            "MAE": mae,
            "RMSE": rmse,
            "MedianAE": med_ae,
            "MAPE": mape,
            "R2": r2,
            "MAE_pct": mae_pct,
            "RMSE_pct": rmse_pct,
            "MedianAE_pct": med_ae_pct,
        }

    def add_lag_features(agg_df: pd.DataFrame):
        """
        Construct lag-based features for each segment defined by:
            (model, region, fuel_type, transmission, color).

        Created features:
            - sales_last1, sales_last2
            - sales_avg3 (3-year rolling average)
            - sales_growth (growth from last2 to last1)
            - price_last1, price_trend
            - mileage_last1, mileage_trend

        Returns
        -------
        df_lagged : pd.DataFrame
            Aggregated data with lag features added.

        num_features : list of str
            List of numeric feature names including lag-based features.

        feature_cols : list of str
            Full feature list (categorical + numeric) for model training.
        """
        df_lagged = agg_df.sort_values(
            ["model", "region", "fuel_type", "transmission", "color", "year"]
        ).copy()

        group_cols = ["model", "region", "fuel_type", "transmission", "color"]

        # Sales lags (t-1, t-2)
        df_lagged["sales_last1"] = df_lagged.groupby(group_cols)["sales_volume"].shift(1)
        df_lagged["sales_last2"] = df_lagged.groupby(group_cols)["sales_volume"].shift(2)

        # Rolling 3-year average sales
        df_lagged["sales_avg3"] = df_lagged.groupby(group_cols)["sales_volume"].transform(
            lambda x: x.rolling(3).mean()
        )

        # Sales growth rate
        df_lagged["sales_growth"] = (
            df_lagged["sales_last1"] - df_lagged["sales_last2"]
        ) / df_lagged["sales_last2"]

        # Price and mileage lags and trends
        df_lagged["price_last1"] = df_lagged.groupby(group_cols)["price_usd"].shift(1)
        df_lagged["price_trend"] = (
            df_lagged["price_usd"] - df_lagged["price_last1"]
        ) / df_lagged["price_last1"]

        df_lagged["mileage_last1"] = df_lagged.groupby(group_cols)["mileage_km"].shift(1)
        df_lagged["mileage_trend"] = (
            df_lagged["mileage_km"] - df_lagged["mileage_last1"]
        ) / df_lagged["mileage_last1"]

        lag_num_features = [
            "sales_last1",
            "sales_last2",
            "sales_avg3",
            "sales_growth",
            "price_last1",
            "price_trend",
            "mileage_last1",
            "mileage_trend",
        ]

        num_features = BASE_NUM_FEATURES + lag_num_features
        feature_cols = CAT_FEATURES + num_features

        return df_lagged, num_features, feature_cols

    def build_catboost_model() -> CatBoostRegressor:
        """
        Create a CatBoostRegressor instance as the core forecasting model.

        The model is configured with RMSE as both loss function and
        evaluation metric, and uses a fixed random seed for reproducibility.
        """
        return CatBoostRegressor(
            depth=8,
            learning_rate=0.05,
            n_estimators=800,
            loss_function="RMSE",
            eval_metric="RMSE",
            random_seed=42,
            verbose=False,
            train_dir='tmp/my_catboost_logs',
        )

    def train_catboost_2020_2023_2024(df_lagged: pd.DataFrame, feature_cols):
        """
        Train CatBoost using 2020–2023 as the training set and validate on 2024.

        This function evaluates the model on the latest available year
        (2024) to measure predictive performance before forecasting 2025.

        Returns
        -------
        metrics_2024 : dict
            Evaluation metrics of CatBoost on 2024.
        """
        df_clean = df_lagged.dropna(subset=feature_cols + ["sales_volume"]).copy()

        train_df = df_clean[df_clean["year"] < 2024]
        valid_df = df_clean[df_clean["year"] == 2024]

        X_train = train_df[feature_cols]
        y_train = train_df["sales_volume"].values

        X_valid = valid_df[feature_cols]
        y_valid = valid_df["sales_volume"].values

        cat_model = build_catboost_model()
        cat_feature_indices = [list(feature_cols).index(col) for col in CAT_FEATURES]

        cat_model.fit(X_train, y_train, cat_features=cat_feature_indices)
        y_pred_valid = cat_model.predict(X_valid)

        metrics_2024 = compute_metrics(y_valid, y_pred_valid)
        return metrics_2024

    def build_2025_features_from_lagged(df_lagged: pd.DataFrame, feature_cols):
        """
        Construct 2025 feature rows by extrapolating from the last available
        record (year 2024) for each segment.

        Logic:
            - Use the most recent row per (model, region, fuel_type, transmission, color)
            - Require complete lag information to ensure valid extrapolation
            - Shift sales-related lag features forward by one year
            - Reuse price and mileage trend information

        Returns
        -------
        feat_2025 : pd.DataFrame
            Feature set for 2025, aligned with feature_cols and including
            an empty sales_volume column (target unknown).
        """
        group_cols = ["model", "region", "fuel_type", "transmission", "color"]

        df_sorted = df_lagged.sort_values(group_cols + ["year"])
        last_rows = df_sorted.groupby(group_cols, as_index=False).tail(1)

        # Require complete lag information for safe extrapolation
        last_rows = last_rows.dropna(
            subset=["sales_last1", "sales_last2", "sales_avg3", "price_last1", "mileage_last1"]
        )

        rows_2025 = []
        for _, row in last_rows.iterrows():
            new_row = row.copy()
            new_row["year"] = 2025

            sales_last1_2025 = row["sales_volume"]
            sales_last2_2025 = row["sales_last1"]
            sales_avg3_2025 = (
                row["sales_volume"] + row["sales_last1"] + row["sales_last2"]
            ) / 3

            new_row["sales_last1"] = sales_last1_2025
            new_row["sales_last2"] = sales_last2_2025
            new_row["sales_avg3"] = sales_avg3_2025

            if not pd.isna(sales_last2_2025) and sales_last2_2025 != 0:
                new_row["sales_growth"] = (
                    sales_last1_2025 - sales_last2_2025
                ) / sales_last2_2025
            else:
                new_row["sales_growth"] = 0.0

            new_row["price_last1"] = row["price_usd"]
            new_row["mileage_last1"] = row["mileage_km"]
            new_row["price_trend"] = row.get("price_trend", 0.0)
            new_row["mileage_trend"] = row.get("mileage_trend", 0.0)

            # Actual 2025 sales are unknown
            new_row["sales_volume"] = np.nan

            rows_2025.append(new_row)

        feat_2025 = pd.DataFrame(rows_2025)
        feat_2025 = feat_2025[
            [c for c in feat_2025.columns if c not in ["sales_volume"]] + ["sales_volume"]
        ]
        feat_2025 = feat_2025[[c for c in feature_cols] + ["sales_volume"]]

        return feat_2025

    def retrain_catboost_full_and_predict_2025(df_lagged: pd.DataFrame, feature_cols):
        """
        Retrain CatBoost on the full 2020–2024 dataset and forecast 2025.

        Steps:
            - Clean rows with valid features and target (sales_volume)
            - Retrain CatBoost on all years 2020–2024
            - Construct 2025 feature set from lagged data
            - Predict 2025 row-level sales
            - Aggregate predictions by:
                model / region / fuel_type / transmission / color

        Returns
        -------
        total_2025 : float
            Sum of predicted 2025 sales across all combinations.

        by_model_2025, by_region_2025, by_fuel_2025, by_trans_2025, by_color_2025 :
            Aggregated prediction tables by each dimension.
        """
        df_clean = df_lagged.dropna(subset=feature_cols + ["sales_volume"]).copy()

        X_full = df_clean[feature_cols]
        y_full = df_clean["sales_volume"].values

        feat_2025 = build_2025_features_from_lagged(df_lagged, feature_cols)
        X_2025 = feat_2025[feature_cols]

        cat_model = build_catboost_model()
        cat_feature_indices = [list(feature_cols).index(col) for col in CAT_FEATURES]
        cat_model.fit(X_full, y_full, cat_features=cat_feature_indices)

        preds_2025 = cat_model.predict(X_2025)
        total_2025 = preds_2025.sum()

        feat_2025_out = feat_2025.copy()
        feat_2025_out["pred_sales_volume_2025_catboost"] = preds_2025

        col_pred = "pred_sales_volume_2025_catboost"

        by_model_2025 = (
            feat_2025_out.groupby("model")[col_pred]
            .sum()
            .reset_index()
        )
        by_region_2025 = (
            feat_2025_out.groupby("region")[col_pred]
            .sum()
            .reset_index()
        )
        by_fuel_2025 = (
            feat_2025_out.groupby("fuel_type")[col_pred]
            .sum()
            .reset_index()
        )
        by_trans_2025 = (
            feat_2025_out.groupby("transmission")[col_pred]
            .sum()
            .reset_index()
        )
        by_color_2025 = (
            feat_2025_out.groupby("color")[col_pred]
            .sum()
            .reset_index()
        )

        return (
            total_2025,
            by_model_2025,
            by_region_2025,
            by_fuel_2025,
            by_trans_2025,
            by_color_2025,
        )

    def make_comparison_table(
        df_2024: pd.DataFrame,
        df_2025: pd.DataFrame,
        key_col: str,
    ) -> pd.DataFrame:
        """
        Build a comparison table between 2024 actual and 2025 forecast.

        Inputs are 2024 and 2025 aggregated tables on the same dimension,
        e.g. region, fuel_type, transmission, or color.

        The output includes:
            - 2024 actual sales
            - 2025 forecast sales
            - Absolute growth
            - Growth rate (%)
        """
        df_2024 = df_2024.rename(columns={"sales_volume": "sales_2024"})
        df_2025 = df_2025.rename(columns={"pred_sales_volume_2025_catboost": "sales_2025"})

        merged = pd.merge(df_2024, df_2025, on=key_col, how="outer").fillna(0.0)

        merged["growth_abs"] = merged["sales_2025"] - merged["sales_2024"]
        merged["growth_pct"] = np.where(
            merged["sales_2024"] != 0,
            merged["growth_abs"] / merged["sales_2024"] * 100,
            np.nan,
        )

        # Sort by 2025 forecasted sales in descending order
        merged = merged.sort_values("sales_2025", ascending=False)

        return merged[[key_col, "sales_2024", "sales_2025", "growth_abs", "growth_pct"]]

    def select_top3_segment_tables(
        tables: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Select Top 3 segments per dimension and merge into a unified table.

        Dimensions:
            - model
            - region
            - fuel_type
            - transmission
            - color

        The resulting table has one row per top segment instance, with
        standardized fields:
            segment_type, segment_name,
            sales_2024, sales_2025, growth_abs, growth_pct
        """
        tbl_model = tables["model"].copy()
        tbl_region = tables["region"].copy()
        tbl_fuel = tables["fuel_type"].copy()
        tbl_trans = tables["transmission"].copy()
        tbl_color = tables["color"].copy()

        # For each dimension, sort by 2025 forecast and keep Top 3
        tbl_model_top3 = tbl_model.sort_values("sales_2025", ascending=False).head(3)
        tbl_region_top3 = tbl_region.sort_values("sales_2025", ascending=False).head(3)
        tbl_fuel_top3 = tbl_fuel.sort_values("sales_2025", ascending=False).head(3)
        tbl_trans_top3 = tbl_trans.sort_values("sales_2025", ascending=False).head(3)
        tbl_color_top3 = tbl_color.sort_values("sales_2025", ascending=False).head(3)

        # Attach segment_type / segment_name fields and remove original key columns
        df_model = tbl_model_top3.assign(
            segment_type="Model",
            segment_name=tbl_model_top3["model"],
        ).drop(columns=["model"])

        df_region = tbl_region_top3.assign(
            segment_type="Region",
            segment_name=tbl_region_top3["region"],
        ).drop(columns=["region"])

        df_fuel = tbl_fuel_top3.assign(
            segment_type="Fuel Type",
            segment_name=tbl_fuel_top3["fuel_type"],
        ).drop(columns=["fuel_type"])

        df_trans = tbl_trans_top3.assign(
            segment_type="Transmission",
            segment_name=tbl_trans_top3["transmission"],
        ).drop(columns=["transmission"])

        df_color = tbl_color_top3.assign(
            segment_type="Color",
            segment_name=tbl_color_top3["color"],
        ).drop(columns=["color"])

        combined_top3_df = pd.concat(
            [df_model, df_region, df_fuel, df_trans, df_color],
            axis=0,
            ignore_index=True,
        )

        combined_top3_df = combined_top3_df[
            ["segment_type", "segment_name", "sales_2024", "sales_2025", "growth_abs", "growth_pct"]
        ]

        return combined_top3_df

    # =====================================================
    # 3. Main flow: from raw df to Top 3 tables and Markdown
    # =====================================================

    # 3.1 Aggregate to segment level and construct lag features
    agg_df = load_and_aggregate_from_df(df)
    df_lagged, _num_features, feature_cols = add_lag_features(agg_df)

    # 3.2 Train on 2020–2023 and validate on 2024 to obtain metrics_2024
    metrics_2024 = train_catboost_2020_2023_2024(df_lagged, feature_cols)

    # 3.3 Retrain CatBoost on full 2020–2024 data and forecast 2025 by segment
    (
        total_2025,
        by_model_2025,
        by_region_2025,
        by_fuel_2025,
        by_trans_2025,
        by_color_2025,
    ) = retrain_catboost_full_and_predict_2025(df_lagged, feature_cols)

    # 3.4 Compute 2024 actual sales by dimension
    df_2024 = agg_df[agg_df["year"] == 2024]

    by_model_2024 = df_2024.groupby("model")["sales_volume"].sum().reset_index()
    by_region_2024 = df_2024.groupby("region")["sales_volume"].sum().reset_index()
    by_fuel_2024 = df_2024.groupby("fuel_type")["sales_volume"].sum().reset_index()
    by_trans_2024 = df_2024.groupby("transmission")["sales_volume"].sum().reset_index()
    by_color_2024 = df_2024.groupby("color")["sales_volume"].sum().reset_index()

    # 3.4.x Total market sales and overall growth rate (for summary and final row)
    total_sales_2024 = df_2024["sales_volume"].sum()
    overall_growth_abs = total_2025 - total_sales_2024
    overall_growth_pct = (
        overall_growth_abs / total_sales_2024 * 100 if total_sales_2024 > 0 else 0
    )

    # 3.5 Build 2024 vs. 2025 comparison tables for each dimension
    table_model = make_comparison_table(by_model_2024, by_model_2025, "model")
    table_region = make_comparison_table(by_region_2024, by_region_2025, "region")
    table_fuel = make_comparison_table(by_fuel_2024, by_fuel_2025, "fuel_type")
    table_trans = make_comparison_table(by_trans_2024, by_trans_2025, "transmission")
    table_color = make_comparison_table(by_color_2024, by_color_2025, "color")

    tables = {
        "model": table_model,
        "region": table_region,
        "fuel_type": table_fuel,
        "transmission": table_trans,
        "color": table_color,
    }

    # 3.6 Select the Top 3 segments per dimension and combine into a single raw table
    combined_top3_df_raw = select_top3_segment_tables(tables)

    # ======= Display version: rename columns and format values for reporting =======
    combined_top3_df_display = combined_top3_df_raw.rename(columns={
        "segment_type": "Segment Category",
        "segment_name": "Segment Name",
        "sales_2024": "2024 Actual Sales",
        "sales_2025": "2025 Forecast Sales",
        "growth_abs": "Absolute Growth (Units)",
        "growth_pct": "Growth Rate (%)",
    })

    # Drop Absolute Growth (Units); keep only percentage growth for display
    combined_top3_df_display = combined_top3_df_display[
        ["Segment Category", "Segment Name", "2024 Actual Sales", "2025 Forecast Sales", "Growth Rate (%)"]
    ]

    # Format numbers with thousand separators and percentage with two decimals
    combined_top3_df_display["2024 Actual Sales"] = combined_top3_df_display[
        "2024 Actual Sales"
    ].apply(lambda x: f"{x:,.0f}")
    combined_top3_df_display["2025 Forecast Sales"] = combined_top3_df_display[
        "2025 Forecast Sales"
    ].apply(lambda x: f"{x:,.0f}")
    combined_top3_df_display["Growth Rate (%)"] = combined_top3_df_display[
        "Growth Rate (%)"
    ].apply(lambda x: f"{x:.2f}%")

    # Append an overall total market row at the bottom
    total_row = {
        "Segment Category": "Overall",
        "Segment Name": "Total Market",
        "2024 Actual Sales": f"{total_sales_2024:,.0f}",
        "2025 Forecast Sales": f"{total_2025:,.0f}",
        "Growth Rate (%)": f"{overall_growth_pct:.2f}%",
    }

    combined_top3_df_display = pd.concat(
        [combined_top3_df_display, pd.DataFrame([total_row])],
        ignore_index=True,
    )

    # 3.7 Convert the display table to Markdown for direct insertion into a report
    md_combined_top3 = combined_top3_df_display.to_markdown(index=False)

    # -----------------------------------------------------
    # 4. Analysis summary (for LLM consumption) — uses raw numeric data
    # -----------------------------------------------------

    # Top segment by predicted 2025 sales
    top_2025_segment = combined_top3_df_raw.sort_values("sales_2025", ascending=False).iloc[0]

    # Top segment by growth rate (excluding NaN growth_pct)
    df_growth = combined_top3_df_raw.dropna(subset=["growth_pct"])
    top_growth_segment = (
        df_growth.sort_values("growth_pct", ascending=False).iloc[0]
        if not df_growth.empty
        else None
    )

    # Full Top 3 records (raw numeric version)
    top3_records = combined_top3_df_raw.to_dict("records")

    analysis_summary = {
        "insight_key": "Sales_Forecast_2025_Top3_Segments",
        "description": (
            "CatBoost-generated time-series predictions for 2025 sales volume. "
            "The analysis highlights the forecasted market outlook, evaluates model "
            "segments across key dimensions. The accompanying tables present the forecasted "
            "Top 3 categories within each dimension (Model, Region, Fuel Type, Transmission, "
            "and Color), providing a clear view of which sub-segments are expected to lead "
            "sales growth in 2025."
        ),
        "findings": {
            # 1. Model reliability and overall trend
            "model_validation_metrics_2024": metrics_2024,
            "forecast_period": "2025 vs 2024 Actual",
            "total_sales_2024_actual": float(total_sales_2024),
            "total_sales_2025_predicted": float(total_2025),
            "overall_growth_pct": f"{overall_growth_pct:.2f}%",

            # 2. Key segment focus (highest predicted 2025 sales)
            "segment_with_highest_2025_sales": {
                "type": str(top_2025_segment["segment_type"]),
                "name": str(top_2025_segment["segment_name"]),
                "predicted_sales": float(top_2025_segment["sales_2025"]),
            },

            # 3. Key segment focus (highest growth potential)
            "segment_with_highest_growth_pct": (
                {
                    "type": str(top_growth_segment["segment_type"]),
                    "name": str(top_growth_segment["segment_name"]),
                    "growth_pct": f"{float(top_growth_segment['growth_pct']):.2f}%",
                }
                if top_growth_segment is not None
                else {"status": "N/A", "reason": "No segments showed significant positive growth"}
            ),

            # 4. Full Top 3 comparison records (for detailed LLM commentary)
            "top3_segment_comparison_records": top3_records,
            "sales_unit": "units",
        },
    }

    # 5. Return all outputs
    #    DataFrame returned is the display version for direct reporting use.
    return (
        metrics_2024,
        float(total_2025),
        combined_top3_df_display,
        md_combined_top3,
        analysis_summary,
    )
