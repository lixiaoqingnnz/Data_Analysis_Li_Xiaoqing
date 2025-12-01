"""
analysis_module.py

封装所有与 CatBoost 销量预测相关的数据处理逻辑：
- 原始明细数据聚合
- 构造 lag 特征
- 使用 CatBoost 训练 & 验证（2020-2023 训练，2024 验证）
- 使用 2020-2024 全部数据重训模型，预测 2025
- 生成 2024 实际 vs 2025 预测 的对比表
- 从各个维度（Region / Fuel Type / Transmission / Color）中选取 Top3
- 生成统一的 Top3 汇总表（DataFrame + Markdown）
"""

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
    使用 CatBoost 对销量进行时间序列式建模，并输出 2024 实际 vs 2025 预测的
    Top3 细分市场汇总表。

    输入:
        df: 清洗后的原始“行级”数据 DataFrame，至少包含以下列：
            ['year','model','region','fuel_type','transmission','color',
             'engine_size_l','mileage_km','price_usd','sales_volume']

    返回:
        metrics_2024: dict
            CatBoost 在 2024 年验证集上的指标（MAE, RMSE, MAPE, R2 等）。

        total_2025: float
            使用 2020-2024 所有数据训练的 CatBoost 模型，
            对 2025 年总销量的预测（所有组合求和）。

        combined_top3_df: pd.DataFrame
            统一的 Top3 汇总表（已商业化改名），字段为：
                - Segment Category
                - Segment Name
                - 2024 Actual Sales
                - 2025 Forecast Sales
                - Growth Rate (%)
              最后一行为 Overall / Total Market，总销量与整体增长率。

        md_combined_top3: str
            combined_top3_df 的 Markdown 表格形式，适合直接写入 .md
            然后由 main.py 的 Markdown 生成脚本转换成 PDF。

        analysis_summary: dict
            提供给 LLM 生成自然语言分析用的结构化摘要（使用数值 raw 版）。
    """
    # -----------------------------------------------------
    # 全局特征配置（仅在此函数及其内部子函数中使用）
    # -----------------------------------------------------
    CAT_FEATURES = ["model", "region", "fuel_type", "transmission", "color"]
    BASE_NUM_FEATURES = ["year", "engine_size_l", "mileage_km", "price_usd"]

    # =====================================================
    # 2. 工具函数定义（仅在本函数内部使用）
    # =====================================================
    def load_and_aggregate_from_df(df_raw: pd.DataFrame) -> pd.DataFrame:
        """
        将“行级明细数据”聚合到
        year + model + region + fuel_type + transmission + color
        这一层级上，便于做时间序列 + 类别组合建模。
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
        计算通用回归指标（含百分率版本）。
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
        针对每个 (model, region, fuel_type, transmission, color) 组合，
        构造销量和价格/里程的 lag 特征。

        返回:
            df_lagged   : 含 lag 特征的数据
            num_features: 数值特征列表
            feature_cols: 全部特征列表（类别 + 数值）
        """
        df_lagged = agg_df.sort_values(
            ["model", "region", "fuel_type", "transmission", "color", "year"]
        ).copy()

        group_cols = ["model", "region", "fuel_type", "transmission", "color"]

        # 前 1 年和前 2 年销量
        df_lagged["sales_last1"] = df_lagged.groupby(group_cols)["sales_volume"].shift(1)
        df_lagged["sales_last2"] = df_lagged.groupby(group_cols)["sales_volume"].shift(2)

        # 滚动 3 年平均销量
        df_lagged["sales_avg3"] = df_lagged.groupby(group_cols)["sales_volume"].transform(
            lambda x: x.rolling(3).mean()
        )

        # 销量增长率
        df_lagged["sales_growth"] = (
            df_lagged["sales_last1"] - df_lagged["sales_last2"]
        ) / df_lagged["sales_last2"]

        # 价格 & 里程滞后与趋势
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
        仅使用一个 CatBoostRegressor 作为核心模型。
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
        使用 2020-2023 作为训练集，2024 作为验证集，
        用于评估 CatBoost 在最新一年的预测表现。
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
        使用每个组合在 2024 年的最后一条记录为基础，
        构造 2025 年的特征行（进行 1 步外推）。
        """
        group_cols = ["model", "region", "fuel_type", "transmission", "color"]

        df_sorted = df_lagged.sort_values(group_cols + ["year"])
        last_rows = df_sorted.groupby(group_cols, as_index=False).tail(1)

        # 要求有完整的 lag 信息，否则无法安全外推
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

            # 2025 年真实销量未知
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
        使用 2020-2024 全部数据训练 CatBoost，并对 2025 所有组合进行预测；
        然后按以下维度聚合：
            - model
            - region
            - fuel_type
            - transmission
            - color
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
        输入：2024 实际 + 2025 预测（按同一维度聚合后的表），
        输出：含绝对增长和增长率的对比表。
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

        # 默认按 2025 预测销量降序排序
        merged = merged.sort_values("sales_2025", ascending=False)

        return merged[[key_col, "sales_2024", "sales_2025", "growth_abs", "growth_pct"]]

    def select_top3_segment_tables(
        tables: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        从每个维度（model / region / fuel_type / transmission / color）的对比表中，
        按 2025 预测销量降序选取 Top3，并合并为统一的一张表。
        """
        tbl_model = tables["model"].copy()
        tbl_region = tables["region"].copy()
        tbl_fuel = tables["fuel_type"].copy()
        tbl_trans = tables["transmission"].copy()
        tbl_color = tables["color"].copy()

        # 各维度按 sales_2025 排序后取前 3
        tbl_model_top3 = tbl_model.sort_values("sales_2025", ascending=False).head(3)
        tbl_region_top3 = tbl_region.sort_values("sales_2025", ascending=False).head(3)
        tbl_fuel_top3 = tbl_fuel.sort_values("sales_2025", ascending=False).head(3)
        tbl_trans_top3 = tbl_trans.sort_values("sales_2025", ascending=False).head(3)
        tbl_color_top3 = tbl_color.sort_values("sales_2025", ascending=False).head(3)

        # 加上 segment_type / segment_name
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
    # 3. 主流程：从原始 df 到 Top3 表格 + Markdown
    # =====================================================

    # 3.1 聚合到组合层并构造 lag 特征
    agg_df = load_and_aggregate_from_df(df)
    df_lagged, _num_features, feature_cols = add_lag_features(agg_df)

    # 3.2 使用 2020-2023 训练，2024 验证，得到 metrics_2024
    metrics_2024 = train_catboost_2020_2023_2024(df_lagged, feature_cols)

    # 3.3 使用 2020-2024 全部数据重训 CatBoost，预测 2025，并按维度聚合
    (
        total_2025,
        by_model_2025,
        by_region_2025,
        by_fuel_2025,
        by_trans_2025,
        by_color_2025,
    ) = retrain_catboost_full_and_predict_2025(df_lagged, feature_cols)

    # 3.4 计算 2024 实际销量（分维度）
    df_2024 = agg_df[agg_df["year"] == 2024]

    by_model_2024 = df_2024.groupby("model")["sales_volume"].sum().reset_index()
    by_region_2024 = df_2024.groupby("region")["sales_volume"].sum().reset_index()
    by_fuel_2024 = df_2024.groupby("fuel_type")["sales_volume"].sum().reset_index()
    by_trans_2024 = df_2024.groupby("transmission")["sales_volume"].sum().reset_index()
    by_color_2024 = df_2024.groupby("color")["sales_volume"].sum().reset_index()

    # 3.4.x 总体总销量与总体增长率（供“总计”行 & summary 使用）
    total_sales_2024 = df_2024["sales_volume"].sum()
    overall_growth_abs = total_2025 - total_sales_2024
    overall_growth_pct = (
        overall_growth_abs / total_sales_2024 * 100 if total_sales_2024 > 0 else 0
    )

    # 3.5 生成 2024 vs 2025 对比表（完整的）
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

    # 3.6 只选择各维度 Top3，并合并为一张统一的表（raw 数值版）
    combined_top3_df_raw = select_top3_segment_tables(tables)

    # ======= display 版：改列名 + 格式化，供 Markdown / 报告使用 =======
    combined_top3_df_display = combined_top3_df_raw.rename(columns={
        "segment_type": "Segment Category",
        "segment_name": "Segment Name",
        "sales_2024": "2024 Actual Sales",
        "sales_2025": "2025 Forecast Sales",
        "growth_abs": "Absolute Growth (Units)",
        "growth_pct": "Growth Rate (%)",
    })

    # 删除 Absolute Growth (Units) 列，只保留增长率
    combined_top3_df_display = combined_top3_df_display[
        ["Segment Category", "Segment Name", "2024 Actual Sales", "2025 Forecast Sales", "Growth Rate (%)"]
    ]

    # 数字格式化为千分位，增长率格式为 xx.xx%
    combined_top3_df_display["2024 Actual Sales"] = combined_top3_df_display[
        "2024 Actual Sales"
    ].apply(lambda x: f"{x:,.0f}")
    combined_top3_df_display["2025 Forecast Sales"] = combined_top3_df_display[
        "2025 Forecast Sales"
    ].apply(lambda x: f"{x:,.0f}")
    combined_top3_df_display["Growth Rate (%)"] = combined_top3_df_display[
        "Growth Rate (%)"
    ].apply(lambda x: f"{x:.2f}%")

    # 在表格底部追加总体总销量行
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

    # 3.7 生成 Markdown 表格文本，main.py 直接用即可写入 .md
    md_combined_top3 = combined_top3_df_display.to_markdown(index=False)

    # -----------------------------------------------------
    # 4. Analysis Summary (for LLM) —— 用 raw 数值版
    # -----------------------------------------------------

    # Top Predicted Segment (Max 2025 sales)
    top_2025_segment = combined_top3_df_raw.sort_values("sales_2025", ascending=False).iloc[0]

    # Top Growth Segment (Max growth_pct, 排除 NaN)
    df_growth = combined_top3_df_raw.dropna(subset=["growth_pct"])
    top_growth_segment = (
        df_growth.sort_values("growth_pct", ascending=False).iloc[0]
        if not df_growth.empty
        else None
    )

    # Top3 记录（raw 数值版）
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
            # 1. 模型可靠性与总体趋势
            "model_validation_metrics_2024": metrics_2024,
            "forecast_period": "2025 vs 2024 Actual",
            "total_sales_2024_actual": float(total_sales_2024),
            "total_sales_2025_predicted": float(total_2025),
            "overall_growth_pct": f"{overall_growth_pct:.2f}%",

            # 2. 关键细分市场焦点 (最高销量预测)
            "segment_with_highest_2025_sales": {
                "type": str(top_2025_segment["segment_type"]),
                "name": str(top_2025_segment["segment_name"]),
                "predicted_sales": float(top_2025_segment["sales_2025"]),
            },

            # 3. 关键细分市场焦点 (最高增长潜力)
            "segment_with_highest_growth_pct": (
                {
                    "type": str(top_growth_segment["segment_type"]),
                    "name": str(top_growth_segment["segment_name"]),
                    "growth_pct": f"{float(top_growth_segment['growth_pct']):.2f}%",
                }
                if top_growth_segment is not None
                else {"status": "N/A", "reason": "No segments showed significant positive growth"}
            ),

            # 4. Top 3 完整对比数据 (供 LLM 撰写详细分析)
            "top3_segment_comparison_records": top3_records,
            "sales_unit": "units",
        },
    }

    # 5. 返回所有结果（DataFrame 返回的是 display 版，更适合报告直接用）
    return (
        metrics_2024,
        float(total_2025),
        combined_top3_df_display,
        md_combined_top3,
        analysis_summary,
    )