import os
import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import geopandas as gpd
from mpl_toolkits.mplot3d import Axes3D, art3d
from math import pi

from .style import set_global_style, apply_common_format
set_global_style()


PLOTS_DIR = 'reports/report_figures'
os.makedirs(PLOTS_DIR, exist_ok=True)


SHP_MAP = "external_src/ne_110m_admin_0_countries_lakes/ne_110m_admin_0_countries_lakes.shp"

def plot_sales_volume_value_year(df_agg: pd.DataFrame) -> tuple[str, dict]:
    """
    Description:
        Annual sales volume and sales value visualization (bars + trend lines),
        including YOY growth annotations.

    Args:
        df_agg (pd.DataFrame): DataFrame indexed by year, containing:
            'total_sales_volume', 'total_sales_value',
            'volume_growth', 'value_growth'

    Returns:
        tuple[str, dict]:
            chart_path: saved chart path
            analysis_summary: structured insights for LLM analysis
    """

    # ---- 1. Data Preparation ----
    df = df_agg.copy()
    years = df.index

    # Fix accidental trailing space in column name
    if "value_growth " in df.columns:
        df = df.rename(columns={"value_growth ": "value_growth"})

    set2 = plt.get_cmap("Set2")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(
        "BMW Global Sales Performance 2020–2024",
        fontsize=20,
        fontweight="bold",
        y=0.97,
    )

    # ---- 2. Left Chart: Sales Volume (million units) ----
    volume_million = df["total_sales_volume"] / 1_000_000

    bars1 = ax1.bar(
        years,
        volume_million,
        color=set2(0),
        width=0.6,
        edgecolor="white",
        linewidth=1.5,
        zorder=3,
    )

    # Value labels (centered)
    for bar, val in zip(bars1, volume_million):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            height / 2,
            f"{val:.2f}",
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
        )

    # Trend line
    ax1.plot(
        years,
        volume_million,
        color=set2(1),
        linewidth=2.5,
        marker="o",
        markersize=8,
        markerfacecolor="white",
        markeredgewidth=2,
        markeredgecolor=set2(1),
        zorder=4,
    )

    # YOY annotations
    for i in range(1, len(df)):
        growth = df["volume_growth"].iloc[i]
        if growth != 0:
            ax1.annotate(
                f"+{growth:.1f}%",
                xy=(years[i], volume_million.iloc[i]),
                xytext=(6, 20),
                textcoords="offset points",
                fontsize=12,
                fontweight="bold",
                color="#d32f2f",
                arrowprops=dict(arrowstyle="->", color="#d32f2f", lw=1.8),
            )

    ax1.set_ylim(0, volume_million.max() * 1.18)
    apply_common_format(
        ax1,
        title="Sales Volume Changes",
        xlabel="Year",
        ylabel="Units Sold (million)",
    )

    # ---- 3. Right Chart: Sales Value (trillion USD) ----
    value_trillion = df["total_sales_value"] / 1e12

    bars2 = ax2.bar(
        years,
        value_trillion,
        color=set2(2),
        width=0.6,
        edgecolor="white",
        linewidth=1.5,
        zorder=3,
    )

    for bar, val in zip(bars2, value_trillion):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            height / 2,
            f"{val:.3f}",
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
        )

    # Trend line
    ax2.plot(
        years,
        value_trillion,
        color=set2(3),
        linewidth=2.5,
        marker="o",
        markersize=8,
        markerfacecolor="white",
        markeredgewidth=2,
        markeredgecolor=set2(3),
        zorder=4,
    )

    # YOY annotation
    for i in range(1, len(df)):
        growth = df["value_growth"].iloc[i]
        if growth != 0:
            ax2.annotate(
                f"+{growth:.1f}%",
                xy=(years[i], value_trillion.iloc[i]),
                xytext=(6, 22),
                textcoords="offset points",
                fontsize=12,
                fontweight="bold",
                color="#d32f2f",
                arrowprops=dict(arrowstyle="->", color="#d32f2f", lw=1.8),
            )

    ax2.set_ylim(0, value_trillion.max() * 1.18)
    apply_common_format(
        ax2,
        title="Sales Value Changes",
        xlabel="Year",
        ylabel="Sales Value (trillion USD)",
    )

    fig.tight_layout()
    fig.subplots_adjust(top=0.9, wspace=0.25)

    chart_path = os.path.join(PLOTS_DIR, "sales_performance_year.png")
    fig.savefig(chart_path, bbox_inches="tight")
    plt.close(fig)

    # ---- 4. Analysis Summary ----
    # Calculate ASP (Average Selling Price) metrics
    # Note: ASP is proxied by Value / Volume
    df['asp'] = df['total_sales_value'] / df['total_sales_volume']
    
    # Calculate ASP CAGR
    cagr_asp = ((df['asp'].iloc[-1] / df['asp'].iloc[0])**(1/(len(df)-1)) - 1)*100

    analysis_summary = {
        "insight_key": "Annual_Sales_Performance",
        "description": (
            "Annual comparison of BMW global sales volume and sales value, "
            "highlighting peak performance years, growth trends, and Average Selling Price (ASP) trajectory."
        ),
        "findings": {
            # Extremes
            "peak_volume_year": int(df['total_sales_volume'].idxmax()),
            "peak_value_year": int(df['total_sales_value'].idxmax()),
            "lowest_volume_year": int(df['total_sales_volume'].idxmin()),
            "lowest_value_year": int(df['total_sales_value'].idxmin()),
            
            # Growth Rates
            "highest_volume_growth_yoy": f"{df['volume_growth'].max():.1f}%",
            "highest_value_growth_yoy": f"{df['value_growth'].max():.1f}%",
            
            # Long-term Trend (CAGR)
            "cagr_volume": (
                f"{((df['total_sales_volume'].iloc[-1] / df['total_sales_volume'].iloc[0])**(1/(len(df)-1)) - 1)*100:.1f}%"
            ),
            "cagr_value": (
                f"{((df['total_sales_value'].iloc[-1] / df['total_sales_value'].iloc[0])**(1/(len(df)-1)) - 1)*100:.1f}%"
            ),
            
            # Key Business Metric (ASP Insight)
            "cagr_asp": f"{cagr_asp:.1f}%",
            "asp_trend_implication": "Positive Pricing/Premiumization" if cagr_asp > 0 else "Negative Pricing/Market Shift"
        },
    }

    return chart_path, analysis_summary



def plot_region_annual_sales_radar(df_radar: pd.DataFrame) -> tuple[str, dict]:
    """
    Description:
        Radar chart comparing annual sales trends across regions.

    Args:
        df_radar (pd.DataFrame): Rows are regions and columns are years,
                                 containing annual sales volumes.

    Returns:
        tuple[str, dict]:
            chart_path: saved chart path
            analysis_summary: structured insights for LLM analysis
    """

    # ---- 1. Chart Setup ----
    categories = df_radar.columns.tolist()
    N = len(categories)
    angles = np.array([n / float(N) * 2 * pi for n in range(N)])

    # ---- 2. Data Normalization ----
    data_min_raw = df_radar.values.min()
    data_max_raw = df_radar.values.max()

    if data_min_raw > 0:
        adjusted_min_for_plot = data_min_raw * 0.95
    else:
        adjusted_min_for_plot = data_min_raw - (data_max_raw - data_min_raw) * 0.1

    df_normalized = (df_radar - adjusted_min_for_plot) / (data_max_raw - adjusted_min_for_plot)
    df_normalized = df_normalized.clip(lower=0)

    # ---- 3. Build Radar Chart ----
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    set2 = plt.get_cmap("Set2")
    line_colors = list(set2.colors)

    ax.set_xticks(angles)
    ax.set_xticklabels(categories)

    # Y-axis tick labels (show original scale)
    num_yticks = 6
    original_ytick_values = np.linspace(adjusted_min_for_plot, data_max_raw, num_yticks)
    normalized_ytick_positions = (original_ytick_values - adjusted_min_for_plot) / (
        data_max_raw - adjusted_min_for_plot
    )

    ax.set_yticks(normalized_ytick_positions)
    ax.set_yticklabels(
        [f"{val:,.0f}" for val in original_ytick_values],
        color="grey",
        size=10,
    )
    ax.set_ylim(0, 1)

    regional_sales_summary = {}

    # ---- 4. Plot Each Region ----
    for i, region in enumerate(df_normalized.index):
        values = df_normalized.loc[region].values.flatten()

        angles_closed = np.append(angles, angles[0])
        values_closed = np.append(values, values[0])

        color_code = line_colors[i % len(line_colors)]

        ax.plot(
            angles_closed,
            values_closed,
            color=color_code,
            linewidth=1.5,
            linestyle="solid",
            label=region,
            zorder=2,
        )

        ax.fill(
            angles_closed,
            values_closed,
            color=color_code,
            alpha=0.2,
            zorder=1,
        )

        ax.plot(
            angles,
            values,
            color=color_code,
            marker="o",
            markersize=4,
            linestyle="None",
            zorder=3,
        )

        regional_sales_summary[region] = df_radar.loc[region].sum()

    ax.set_title("Regional Annual Sales Volume Trend", y=1.1)
    ax.legend(loc="lower left", bbox_to_anchor=(1.05, 0.05))

    chart_path = os.path.join(PLOTS_DIR, "regional_sales_radar_chart.png")
    plt.savefig(chart_path, bbox_inches="tight", dpi=300)
    plt.close()

    # ---- 5. Analysis Summary ----
    total_sales_df = pd.Series(regional_sales_summary).sort_values(ascending=False)

    regional_sales_summary = {}
    annual_volatility = {}

    for region in df_radar.index:
        series = df_radar.loc[region]
        
        # Calculate Coefficient of Variation (CV) for stability analysis
        cv = series.std() / series.mean()
        annual_volatility[region] = cv
        
        regional_sales_summary[region] = {
            "total_sales": series.sum(),
            "peak_year": series.idxmax(),
            "trough_year": series.idxmin(),
            "sales_data": series.to_dict() # Provide raw data for LLM to reason about shape
        }

    # Identify most/least volatile regions
    volatility_series = pd.Series(annual_volatility)
    most_stable_region = volatility_series.idxmin()
    most_volatile_region = volatility_series.idxmax()

    analysis_summary = {
        "insight_key": "Regional_Sales_Radar",
        "description": (
            "Radar chart comparing multi-year sales performance across regions. The chart is normalized "
            "to show the relative pattern and stability of sales contribution over time."
        ),
        "findings": {
            # 1. Total Ranking (For context)
            "top_regions_by_total_sales": total_sales_df.head(2).index.tolist(),
            
            # 2. Pattern/Stability Analysis (Crucial for Radar Chart)
            "most_stable_region_by_cv": most_stable_region,
            "most_volatile_region_by_cv": most_volatile_region,
            
            # 3. Time-based Extremes
            "leading_region_peak_year": {
                "region": df_radar.loc[total_sales_df.index[0]].idxmax(), 
                "year": total_sales_df.index[0] # Note: This line might need fixing depending on how you want to present. 
                                                # Corrected: Get peak year for the top region
            },
            
            # 4. Detailed data (To enable LLM to discuss annual changes)
            "regional_annual_sales_data": {
                region: regional_sales_summary[region]['sales_data'] for region in df_radar.index
            }
        },
    }

    return chart_path, analysis_summary




def plot_total_sales_region_3d(df_sales: pd.DataFrame) -> tuple[str, dict]:
    """
    Description:
        3D extruded world map showing total sales volume by region.

    Args:
        df_sales (pd.DataFrame): Aggregated dataframe with columns:
            'region' and 'sales' (total sales volume per region).

    Returns:
        tuple[str, dict]:
            chart_path: saved chart path
            analysis_summary: structured insights for LLM analysis
    """

    # ---- 1. Map Countries to Regions and Merge Sales ----
    world = gpd.read_file(SHP_MAP)

    middle_east_countries = [
        "Saudi Arabia", "United Arab Emirates", "Qatar", "Kuwait", "Bahrain",
        "Oman", "Yemen", "Iraq", "Syria", "Jordan", "Lebanon",
        "Israel", "Palestine", "Iran", "Turkey", "Egypt"
    ]

    def map_region(row):
        name = row.get("NAME", row.get("name", ""))
        cont = row["CONTINENT"]

        if name in middle_east_countries:
            return "Middle East"
        if cont == "North America":
            return "North America"
        elif cont == "South America":
            return "South America"
        elif cont == "Europe":
            return "Europe"
        elif cont == "Africa":
            return "Africa"
        elif cont in ["Asia"]:
            return "Asia"
        else:
            return None

    world["region"] = world.apply(map_region, axis=1)
    world = world.merge(df_sales, on="region", how="left")

    world["sales"] = world["sales"].fillna(0)
    world["sales_million"] = world["sales"] / 1e6  # for color scaling

    real_sales_m = world.loc[world["sales"] > 0, "sales_million"]
    if not real_sales_m.empty:
        vmin = real_sales_m.min()
        vmax = real_sales_m.max()
    else:
        vmin, vmax = 0, 1

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("Blues")

    # ---- 2. Draw 3D Extruded Polygons ----
    fig = plt.figure(figsize=(20, 12))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")

    max_sales = world["sales"].max() if world["sales"].max() > 0 else 1
    height_scale = 0.22 / max_sales

    for _, row in world.iterrows():
        if row["region"] is None or row["sales"] == 0:
            continue

        geom = row.geometry
        sales_raw = row["sales"]
        val_million = row["sales_million"]
        height = sales_raw * height_scale
        color = cmap(norm(val_million))

        if geom.geom_type == "Polygon":
            polys = [geom]
        elif geom.geom_type == "MultiPolygon":
            polys = geom.geoms
        else:
            continue

        for poly in polys:
            if poly.exterior is None:
                continue

            xs, ys = poly.exterior.coords.xy
            bottom = np.array([xs, ys, np.zeros(len(xs))]).T
            top = np.array([xs, ys, np.full(len(xs), height)]).T

            verts = [
                list(zip(xs, ys, np.zeros(len(xs)))),
                list(zip(xs, ys, np.full(len(xs), height)))
            ]

            for i in range(len(xs) - 1):
                verts.append([bottom[i], bottom[i + 1], top[i + 1], top[i]])
            verts.append([bottom[-1], bottom[0], top[0], top[-1]])

            poly3d = Poly3DCollection(
                verts,
                facecolors=color,
                linewidths=0.1,
                edgecolor="grey",
                alpha=1.0,
            )
            ax.add_collection3d(poly3d)

    # ---- 3. View Settings ----
    ax.view_init(elev=78, azim=-90)
    ax.set_box_aspect([20, 12, 8])
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 85)
    ax.set_zlim(0, 0.5)
    ax.set_axis_off()
    ax.grid(False)

    # ---- 4. Region Labels (Name + Sales) ----
    region_centroids = {}
    for region in df_sales["region"]:
        subset = world[world["region"] == region]
        if subset.empty or subset["sales"].iloc[0] == 0:
            continue

        united_geom = subset.geometry.union_all()
        if united_geom is None:
            continue

        centroid = united_geom.centroid
        sales_raw = subset["sales"].iloc[0]
        height = sales_raw * height_scale

        region_centroids[region] = {
            "x": centroid.x,
            "y": centroid.y,
            "z": height + 0.008,
            "sales": sales_raw,
        }

    for region, pos in region_centroids.items():
        label_z = pos["z"] + 0.025
        MAX_LABEL_HEIGHT = 0.48

        ax.text(
            pos["x"],
            pos["y"],
            MAX_LABEL_HEIGHT,
            f"{region}\n{pos['sales']:,}",
            fontsize=10.5,
            fontweight="bold",
            color="white",
            ha="center",
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.32",
                facecolor="#1e3a5e",
                edgecolor="white",
                linewidth=1.1,
                alpha=0.92,
            ),
            zorder=99999,
        )

        ax.plot(
            [pos["x"], pos["x"]],
            [pos["y"], pos["y"]],
            [pos["z"], label_z - 0.008],
            color="white",
            linewidth=1.2,
            alpha=0.7,
            zorder=999,
        )

    # ---- 5. Title and Colorbar ----
    plt.title(
        "Sales by Region",
        fontsize=20,
        fontweight="bold",
        pad=30,
        loc="center",
    )
    plt.subplots_adjust(top=0.80)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.7, aspect=20, pad=0.02)
    cbar.set_label("Sales Volume (million)", fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    # ---- 6. Save Figure ----
    chart_path = os.path.join(PLOTS_DIR, "regional_sales_3d_map.png")
    plt.savefig(chart_path, dpi=300, bbox_inches="tight")
    plt.close()

    # ---- 7. Analysis Summary (Optimized) ----
    df_sorted = df_sales.sort_values("sales", ascending=False)
    total_sales = df_sorted["sales"].sum()
    
    # 1. Calculate the share of the top N regions (e.g., Top 3)
    top_n = 3
    top_regions = df_sorted.head(top_n)
    top_regions_sales = top_regions["sales"].sum()
    concentration_ratio = (top_regions_sales / total_sales) * 100 if total_sales > 0 else 0
    
    # 2. Calculate the disparity ratio
    highest_sales = df_sorted.iloc[0]["sales"]
    lowest_sales = df_sorted.iloc[-1]["sales"]
    disparity_ratio = (highest_sales / lowest_sales) if lowest_sales > 0 else float('inf')

    analysis_summary = {
        "insight_key": "Regional_3D_Sales_Map",
        "description": (
            "3D extruded world map showing total sales volume by region. This visualization is used to assess "
            "market concentration and regional dominance over the entire analysis period."
        ),
        "findings": {
            # Dominance and Extremes
            "highest_sales_region": df_sorted.iloc[0]["region"],
            "highest_sales_value": int(df_sorted.iloc[0]["sales"]),
            "lowest_sales_region": df_sorted.iloc[-1]["region"],
            
            # Market Concentration Metrics (New)
            "top_regions_for_dominance": top_regions["region"].tolist(),
            "top_n_concentration_ratio": f"{concentration_ratio:.1f}%", # e.g., Top 3 regions account for 85% of sales
            
            # Regional Disparity Metrics (New)
            "sales_disparity_ratio_highest_to_lowest": f"{disparity_ratio:.1f}x", # e.g., Highest region sells 15.5x the lowest
            
            "sales_unit": "units (raw sales volume)",
        },
    }

    return chart_path, analysis_summary






def plot_price_box_region(df: pd.DataFrame) -> tuple[str, dict]:
    """
    Description:
        Violin plot showing vehicle price distribution by region.

    Args:
        df (pd.DataFrame): Dataframe containing at least 'region' and 'price_usd' columns.

    Returns:
        tuple[str, dict]:
            chart_path: saved chart path
            analysis_summary: structured insights for LLM analysis
    """

    # ---- 1. Figure and Violin Data ----
    fig, ax = plt.subplots(figsize=(12, 6))

    regions = df["region"].unique()
    data = [df[df["region"] == r]["price_usd"].values for r in regions]

    violin = ax.violinplot(
        data,
        showmedians=True,
        showextrema=True,
    )

    # ---- 2. Color Styling (Set2 Palette) ----
    set2 = plt.get_cmap("Set2")
    for i, body in enumerate(violin["bodies"]):
        body.set_facecolor(set2.colors[i % len(set2.colors)])
        body.set_edgecolor("black")
        body.set_alpha(0.8)

    if "cmedians" in violin:
        violin["cmedians"].set_color("#444444")
        violin["cmedians"].set_linewidth(1.8)

    # x-axis ticks for violinplot positions (1..N)
    ax.set_xticks(np.arange(1, len(regions) + 1))
    ax.set_xticklabels(regions)

    # ---- 3. Axes Range and Common Formatting ----
    y_min = max(0, df["price_usd"].min() - 10000)
    y_max = df["price_usd"].max() + 10000
    ax.set_ylim(y_min, y_max)

    apply_common_format(
        ax,
        title="Car Price Distribution by Region (Violin Plot)",
        xlabel="Region",
        ylabel="Price (USD)",
    )

    fig.tight_layout()

    # ---- 4. Save Figure ----
    chart_path = os.path.join(PLOTS_DIR, "price_boxplot_by_region.png")
    fig.savefig(chart_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

    # ---- 5. Analysis Summary (Optimized & Fixed) ----
    
    # 使用字典形式的 agg 确保 IQR 列名被正确创建
    price_stats = (
        df.groupby("region")["price_usd"]
          .agg(
              median='median',
              mean='mean',
              min='min',
              max='max',
              IQR=lambda x: x.quantile(0.75) - x.quantile(0.25)  # 直接命名列为 'IQR'
          )
          .sort_values("median", ascending=False)
    )

    highest_median_region = price_stats.index[0]
    lowest_median_region = price_stats.index[-1]

    # Analyze Dispersion (IQR)
    highest_dispersion_region = price_stats["IQR"].idxmax()
    lowest_dispersion_region = price_stats["IQR"].idxmin()
    
    # Calculate median price disparity ratio
    median_ratio = (price_stats.loc[highest_median_region, "median"] / price_stats.loc[lowest_median_region, "median"])

    analysis_summary = {
        "insight_key": "Price_Distribution_By_Region",
        "description": (
            "Violin plot detailing the full distribution of vehicle prices by region. "
            "Analysis focuses on central tendency (median) and price dispersion (IQR) to assess market segments."
        ),
        "findings": {
            # 1. Central Tendency Insights (Median)
            "highest_median_price_region": highest_median_region,
            "highest_median_price_value": float(price_stats.loc[highest_median_region, "median"]),
            "lowest_median_price_region": lowest_median_region,
            "median_price_disparity_ratio": f"{median_ratio:.2f}x",

            # 2. Dispersion Insights (IQR)
            "highest_price_dispersion_region": highest_dispersion_region,
            "highest_dispersion_value_iqr": float(price_stats.loc[highest_dispersion_region, "IQR"]),
            "lowest_price_dispersion_region": lowest_dispersion_region,
            
            # 3. Full Data for LLM context
            "median_price_by_region": {
                region: float(row["median"])
                for region, row in price_stats.iterrows()
            },
            "IQR_by_region": {
                region: float(row["IQR"])
                for region, row in price_stats.iterrows()
            },
            "price_unit": "USD",
        },
    }

    return chart_path, analysis_summary


def plot_model_performance_bar(
    model_sales_df: pd.DataFrame,
    top_n: int = 3,
    bottom_n: int = 3
) -> tuple[str, dict]:
    """
    Description:
        Horizontal bar chart of model-level sales performance,
        highlighting top-performing and underperforming models.

    Args:
        model_sales_df (pd.DataFrame): Aggregated dataframe containing at least
            'model' and 'total_sales_volume' columns.
        top_n (int): Number of top-selling models to highlight.
        bottom_n (int): Number of lowest-selling models to highlight.

    Returns:
        tuple[str, dict]:
            chart_path: saved chart path
            analysis_summary: structured insights for LLM analysis
    """

    # ---- 1. Sort Data by Total Sales Volume ----
    df_desc = (
        model_sales_df
        .sort_values("total_sales_volume", ascending=False)
        .reset_index(drop=True)
    )

    n_models = len(df_desc)
    top_n = min(top_n, n_models)
    bottom_n = min(bottom_n, n_models - top_n) if n_models > top_n else 0

    # For barh plotting, sort ascending so smaller bars appear at the bottom
    df_plot = (
        df_desc
        .sort_values("total_sales_volume", ascending=True)
        .reset_index(drop=True)
    )

    models = df_plot["model"].tolist()
    values = df_plot["total_sales_volume"].tolist()
    y_pos = np.arange(len(models))

    # Recompute top / bottom indices under ascending order
    top_indices = list(range(len(df_plot) - top_n, len(df_plot)))
    bottom_indices = list(range(0, bottom_n))

    # ---- 2. Color Mapping (Set2 + Semantic Highlights) ----
    set2 = plt.get_cmap("Set2")

    default_color = set2.colors[0]   # baseline color for normal models
    top_color = set2.colors[2]       # highlight for top-performing models
    bottom_color = set2.colors[1]    # highlight for underperforming models

    bar_colors = []
    for i in range(len(df_plot)):
        if i in top_indices:
            bar_colors.append(top_color)
        elif i in bottom_indices:
            bar_colors.append(bottom_color)
        else:
            bar_colors.append(default_color)

    # ---- 3. Draw Horizontal Bar Chart ----
    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.barh(y_pos, values, color=bar_colors)

    # Value labels inside bars (right aligned)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() * 0.98,
            bar.get_y() + bar.get_height() / 2,
            f"{val:,}",
            va="center",
            ha="right",
            fontsize=10,
            color="white",
            fontweight="bold",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(models)

    apply_common_format(
        ax,
        title="Top-Performing and Underperforming Models",
        xlabel="Total Sales Volume (units)",
        ylabel="Model",
    )

    fig.tight_layout()

    # ---- 4. Save Figure ----
    chart_path = os.path.join(PLOTS_DIR, "top_bottom_models_barh.png")
    fig.savefig(chart_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---- 5. Analysis Summary (Optimized - Volume Only) ----
    
    df_volume_sorted = df_desc.sort_values("total_sales_volume", ascending=False)
    
    # 1. Calculate Total Sales Volume
    total_sales_volume_all = df_volume_sorted["total_sales_volume"].sum()
    
    # 2. Identify Top/Bottom Performers
    top_models_volume = df_volume_sorted.head(top_n)["model"].tolist()
    bottom_models_volume = df_volume_sorted.tail(bottom_n)["model"].tolist()
    
    # 3. Calculate Market Concentration (e.g., Top 3 Models' share)
    top_n_volume = df_volume_sorted.head(top_n)["total_sales_volume"].sum()
    volume_concentration_ratio = (top_n_volume / total_sales_volume_all) * 100 if total_sales_volume_all > 0 else 0
    
    # 4. Long-Tail Insight (Optional, but adds depth)
    # Percentage of models contributing to the long tail (e.g., models outside Top N)
    long_tail_models_count = len(df_volume_sorted) - top_n
    long_tail_volume_share = (total_sales_volume_all - top_n_volume) / total_sales_volume_all * 100

    analysis_summary = {
        "insight_key": "Model_Performance_Sales_Volume",
        "description": (
            "Horizontal bar chart ranking car models by total sales volume. "
            "Analysis focuses on identifying portfolio risk, market concentration, and 'long-tail' contribution."
        ),
        "findings": {
            # 1. Product Extremes
            "top_models_by_volume": top_models_volume,
            "bottom_models_by_volume": bottom_models_volume,
            
            # 2. Market Concentration Metrics (Crucial business insight)
            "concentration_ratio_top_n_volume": f"{volume_concentration_ratio:.1f}%",
            "concentration_N_models": top_n,
            
            # 3. Long-Tail Contribution (New)
            "long_tail_models_count": long_tail_models_count,
            "long_tail_volume_share": f"{long_tail_volume_share:.1f}%",
            
            # 4. Full Data for LLM context
            "model_sales_volume_ranking": {
                row["model"]: int(row["total_sales_volume"])
                for _, row in df_volume_sorted.iterrows()
            },
            "sales_unit": "units",
        },
    }

    return chart_path, analysis_summary





def plot_mileage_sales(df_mileage_impact: pd.DataFrame) -> tuple[str, dict]:
    """
    Description:
        Bar chart showing how different mileage ranges impact total sales volume.

    Args:
        df_mileage_impact (pd.DataFrame): Aggregated dataframe containing
            'mileage_group' and 'total_sales' columns.

    Returns:
        tuple[str, dict]:
            chart_path: saved chart path
            analysis_summary: structured insights for LLM analysis
    """

    # ---- 1. Enforce Mileage Group Order (Low → Mid → High) ----
    ordered_labels = ["Low Mileage", "Mid Mileage", "High Mileage"]
    df_mileage_impact["mileage_group"] = pd.Categorical(
        df_mileage_impact["mileage_group"],
        categories=ordered_labels,
        ordered=True,
    )
    df_mileage_impact = df_mileage_impact.sort_values("mileage_group")

    groups = df_mileage_impact["mileage_group"].astype(str).tolist()
    sales = df_mileage_impact["total_sales"].tolist()

    # ---- 2. Plot Bars with Semantic Colors (Low=Green, Mid=Yellow, High=Red) ----
    fig, ax = plt.subplots(figsize=(9, 5))

    set2 = plt.get_cmap("Set2")
    mileage_colors = [set2(4), set2(5), set2(1)]

    bars = ax.bar(groups, sales, color=mileage_colors, alpha=0.85)

    # Value labels at the top of each bar
    ax.bar_label(bars, fmt="%.0f", fontsize=10)

    # Y-axis formatting (K / M)
    def formatter(val, _):
        if val >= 1e6:
            return f"{val/1e6:.1f}M"
        elif val >= 1e3:
            return f"{val/1e3:.0f}K"
        return f"{val:.0f}"

    ax.yaxis.set_major_formatter(FuncFormatter(formatter))
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    # Unified title and axis style
    apply_common_format(
        ax,
        title="Total Sales Volume by Mileage Range",
        xlabel="Mileage Group",
        ylabel="Total Sales Volume (units)",
    )

    fig.tight_layout()

    # ---- 3. Save Figure ----
    chart_path = os.path.join(PLOTS_DIR, "sales_by_mileage.png")
    fig.savefig(chart_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---- 4. Analysis Summary (Optimized) ----
    
    # Calculate total sales and add market share column
    total_sales_all = df_mileage_impact["total_sales"].sum()
    df_mileage_impact["market_share"] = (
        df_mileage_impact["total_sales"] / total_sales_all
    ) * 100
    
    # Identify top and bottom groups
    top_group_data = df_mileage_impact.loc[
        df_mileage_impact["total_sales"].idxmax()
    ]
    low_group_data = df_mileage_impact.loc[
        df_mileage_impact["total_sales"].idxmin()
    ]
    
    # Calculate sales disparity ratio
    top_sales = top_group_data["total_sales"]
    low_sales = low_group_data["total_sales"]
    sales_disparity_ratio = (top_sales / low_sales) if low_sales > 0 else float('inf')


    analysis_summary = {
        "insight_key": "Mileage_Impact_Analysis_Segmentation",
        "description": (
            "Bar chart showing how different mileage ranges impact used-car sales. "
            "Analysis focuses on quantifying the market share and dominance of key mileage segments."
        ),
        "findings": {
            # 1. Segment Dominance and Extremes
            "highest_sales_group": str(top_group_data["mileage_group"]),
            "highest_sales_share": f'{top_group_data["market_share"]:.1f}%',
            "lowest_sales_group": str(low_group_data["mileage_group"]),
            
            # 2. Market Structure Metric
            "sales_disparity_ratio_highest_to_lowest": f"{sales_disparity_ratio:.1f}x",
            
            # 3. Full Data for LLM context (including share)
            "segment_performance": [
                {
                    "mileage_group": str(row["mileage_group"]),
                    "total_sales": int(row["total_sales"]),
                    "market_share": f'{row["market_share"]:.1f}%'
                }
                for _, row in df_mileage_impact.iterrows()
            ],
            "total_sales_volume": int(total_sales_all),
            "sales_unit": "units",
        },
    }

    return chart_path, analysis_summary



def plot_fuel_type_by_region_facets(
    region_summary_df: pd.DataFrame
) -> tuple[str, dict]:
    """
    Description:
        Faceted bar charts showing fuel-type sales distribution within each region.

    Args:
        region_summary_df (pd.DataFrame): Aggregated dataframe containing
            'region', 'fuel_type', and 'total_sales' columns.

    Returns:
        tuple[str, dict]:
            chart_path: saved chart path
            analysis_summary: structured insights for LLM analysis
    """

    # ---- 1. Prepare regions & fuel types ----
    regions = sorted(region_summary_df["region"].unique().tolist())
    fuel_types = ["Diesel", "Petrol", "Electric", "Hybrid"]

    n_regions = len(regions)
    n_cols = 3
    n_rows = int(np.ceil(n_regions / n_cols))

    # ---- 2. Colors (Set2 palette) ----
    cmap = plt.get_cmap("Set2")
    colors = [cmap(i) for i in range(len(fuel_types))]
    color_map = {ft: colors[i] for i, ft in enumerate(fuel_types)}

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5 * n_cols, 4 * n_rows),
        squeeze=False,
    )

    # ---- 3. Y-axis formatter (K / M) ----
    def millions_formatter(val, _):
        if val >= 1e6:
            return f"{val/1e6:.1f}M"
        elif val >= 1e3:
            return f"{val/1e3:.0f}K"
        return f"{val:.0f}"

    # ---- 4. Collect summary records ----
    summary_records = []

    # ---- 5. Draw each facet (one region per subplot) ----
    for idx, region in enumerate(regions):
        r_df = region_summary_df[region_summary_df["region"] == region]

        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row][col]

        # Ensure consistent fuel_type order for all regions
        totals = []
        for ft in fuel_types:
            sub = r_df[r_df["fuel_type"] == ft]
            val = float(sub["total_sales"].iloc[0]) if not sub.empty else 0.0
            totals.append(val)

        x = np.arange(len(fuel_types))

        # ---- Bars for this region ----
        ax.bar(
            x,
            totals,
            color=[color_map[ft] for ft in fuel_types],
            alpha=0.85,
        )

        ax.set_xticks(x)
        ax.set_xticklabels(fuel_types, rotation=25)

        # Keep your original y-axis scaling logic
        ax.set_ylim(
            max(region_summary_df["total_sales"]) * 0.8,
            max(region_summary_df["total_sales"]) * 1.03,
        )
        ax.yaxis.set_major_formatter(FuncFormatter(millions_formatter))
        ax.grid(axis="y", linestyle="--", alpha=0.3)

        # Apply unified axis style (title only, no labels inside facets)
        apply_common_format(
            ax,
            title=f"{region}",
            xlabel=None,
            ylabel=None,
        )

        # Summary: top fuel type within each region
        if totals:
            max_idx = int(np.argmax(totals))
            summary_records.append(
                {
                    "region": region,
                    "top_fuel_type": fuel_types[max_idx],
                    "top_sales": totals[max_idx],
                }
            )

    # ---- 6. Remove unused subplots (if any) ----
    for j in range(idx + 1, n_rows * n_cols):
        fig.delaxes(axes[j // n_cols][j % n_cols])

    # ---- 7. Figure-level title ----
    fig.suptitle(
        "Fuel Type Sales by Region (Facet Comparison)",
        fontsize=20,
        fontweight="bold",
        y=0.98,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    # ---- 8. Save figure ----
    chart_path = os.path.join(PLOTS_DIR, "fuel_type_by_region_facets.png")
    fig.savefig(chart_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---- 9. Analysis summary for LLM (Optimized) ----
    
    # 1. Calculate Market Share and Dominance Metrics
    
    # Group by region to get regional totals
    regional_totals = region_summary_df.groupby('region')['total_sales'].sum().rename('regional_total_sales')
    
    df_metrics = region_summary_df.merge(regional_totals, on='region')
    # Calculate market share for each fuel type within its region
    df_metrics['market_share'] = (df_metrics['total_sales'] / df_metrics['regional_total_sales']) * 100
    
    # 2. Summarize Dominance and NEV Share for each region
    regional_fuel_summary = {}
    new_energy_types = ["Electric", "Hybrid"]
    
    # Use regions list from step 1
    for region in regions:
        r_df = df_metrics[df_metrics['region'] == region]
        
        # Dominance: top fuel type's share
        top_row = r_df.loc[r_df['total_sales'].idxmax()]
        dominance_share = float(top_row['market_share'])
        
        # NEV Share (Electric + Hybrid)
        nev_sales = r_df[r_df['fuel_type'].isin(new_energy_types)]['total_sales'].sum()
        nev_share = (nev_sales / top_row['regional_total_sales']) * 100 if top_row['regional_total_sales'] > 0 else 0
        
        regional_fuel_summary[region] = {
            "dominance_fuel": str(top_row['fuel_type']),
            "dominance_share": f"{dominance_share:.1f}%", # Top fuel type's percentage of regional sales
            "nev_sales_share": f"{nev_share:.1f}%",      # Combined NEV percentage of regional sales
            "regional_total_sales": int(top_row['regional_total_sales'])
        }

    # 3. Identify Extremes Across Regions for NEV Adoption
    df_temp = pd.DataFrame.from_dict(regional_fuel_summary, orient='index')
    df_temp['nev_sales_share_float'] = df_temp['nev_sales_share'].str.rstrip('%').astype(float)
    
    highest_nev_region = df_temp['nev_sales_share_float'].idxmax()
    lowest_nev_region = df_temp['nev_sales_share_float'].idxmin()

    analysis_summary = {
        "insight_key": "Regional_FuelType_Segmentation",
        "description": (
            "Facet bar charts comparing the internal market share of fuel types within each region. "
            "Analysis focuses on fuel type dominance and New Energy Vehicle (NEV) adoption rates to highlight regional market maturity."
        ),
        "findings": {
            # Key Cross-Regional Insights (New Energy Adoption) - NEW
            "highest_nev_share_region": highest_nev_region,
            "highest_nev_share_value": f"{df_temp.loc[highest_nev_region, 'nev_sales_share_float']:.1f}%",
            "lowest_nev_share_region": lowest_nev_region,
            
            # Regional-Level Summaries (Dominance and Share) - NEW
            "regional_fuel_segmentation_summary": regional_fuel_summary,

            # Raw Data and Structure
            "all_fuel_types": fuel_types,
            "new_energy_types": new_energy_types,
            "sales_data_records": df_metrics.to_dict("records"), # Includes market_share for full context
        },
    }

    return chart_path, analysis_summary



def plot_fuel_type_trend_grouped(
    fuel_trend_df: pd.DataFrame,
) -> tuple[str, dict]:
    """
    Description:
        Grouped bar + line chart showing annual sales trends for each fuel type.

    Args:
        fuel_trend_df (pd.DataFrame): Aggregated dataframe containing
            'year', 'fuel_type', and 'total_sales' columns.

    Returns:
        tuple[str, dict]:
            chart_path: saved chart path
            analysis_summary: structured insights for LLM analysis
    """

    # ---- 1. Prepare year × fuel_type matrix ----
    years = sorted(fuel_trend_df["year"].dropna().unique().tolist())
    fuel_types = sorted(fuel_trend_df["fuel_type"].dropna().unique().tolist())

    sales_matrix = pd.DataFrame(0.0, index=years, columns=fuel_types)

    for _, row in fuel_trend_df.iterrows():
        y = int(row["year"])
        ft = row["fuel_type"]
        sales_matrix.loc[y, ft] = row["total_sales"]

    # ---- 2. Grouped bar chart + trend lines ----
    fig, ax = plt.subplots(figsize=(11, 6))

    x = np.arange(len(years))
    n_ft = len(fuel_types)
    bar_width = 0.8 / n_ft

    # Use Set2 color scheme (aligned with global theme)
    cmap = plt.get_cmap("Set2")
    colors = [cmap(i) for i in range(n_ft)]

    for i, ft in enumerate(fuel_types):
        y_vals = sales_matrix[ft].values

        # Horizontal shift per fuel type to form grouped bars
        bar_positions = x - 0.8 / 2 + bar_width / 2 + i * bar_width

        # Bars: slightly transparent, so lines are visible on top
        ax.bar(
            bar_positions,
            y_vals,
            width=bar_width,
            color=colors[i],
            alpha=0.6,
            label=f"{ft} (Bar)" if i == 0 else None,  # placeholder, filtered out in legend
        )

        # Lines: use year centers for trend curves
        ax.plot(
            x,
            y_vals,
            marker="o",
            linestyle="-",
            linewidth=2,
            color=colors[i],
            label=ft,
        )

    # X-axis: years
    ax.set_xticks(x)
    ax.set_xticklabels(years)

    # Keep your original y-axis range
    ax.set_ylim(2_000_000, fuel_trend_df["total_sales"].max() * 1.1)

    # ---- 3. Y-axis formatting (K / M) ----
    def millions_formatter(val, _):
        if val >= 1e6:
            return f"{val/1e6:.1f}M"
        elif val >= 1e3:
            return f"{val/1e3:.0f}K"
        return f"{val:.0f}"

    ax.yaxis.set_major_formatter(FuncFormatter(millions_formatter))
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    # ---- 4. Apply unified axis style ----
    apply_common_format(
        ax,
        title="Fuel Type Sales in Used-Car Market by Year\n(Grouped Bars + Trend Lines)",
        xlabel="Year",
        ylabel="Total Sales Volume (units)",
    )

    # Legend: keep only pure fuel-type labels (hide "(Bar)")
    handles, labels = ax.get_legend_handles_labels()
    filtered = [(h, l) for h, l in zip(handles, labels) if "(Bar)" not in l]
    if filtered:
        handles, labels = zip(*filtered)
        ax.legend(
            handles,
            labels,
            title="Fuel Type",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
        )
    else:
        ax.legend()

    fig.tight_layout()

    chart_path = os.path.join(PLOTS_DIR, "fuel_type_grouped_bar_trend.png")
    fig.savefig(chart_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---- 5. Analysis summary (Optimized) ----

    latest_year = int(max(years))
    earliest_year = int(min(years))
    n_years = len(years)

    # 1. Calculate Market Share for all years
    annual_total_sales = (
        fuel_trend_df.groupby("year")["total_sales"].sum().rename("annual_total_sales")
    )
    df_metrics = fuel_trend_df.merge(annual_total_sales, on="year")
    df_metrics["market_share"] = (
        df_metrics["total_sales"] / df_metrics["annual_total_sales"]
    ) * 100

    # 2. Analyze Share Shift and CAGR
    fuel_share_shift = {}
    fuel_cagr = {}
    
    # NEV analysis (use filtered fuel_list from original code)
    new_energy_types = [
        ft
        for ft in fuel_types
        if any(kw in str(ft) for kw in ["Electric", "Hybrid", "Plug-in"])
    ]

    for ft in fuel_types:
        ft_df = df_metrics[df_metrics["fuel_type"] == ft].sort_values("year")

        if not ft_df.empty:
            # Market Share Shift
            first_share = (
                ft_df.loc[ft_df["year"] == earliest_year, "market_share"].iloc[0]
                if not ft_df[ft_df["year"] == earliest_year].empty else 0
            )
            latest_share = (
                ft_df.loc[ft_df["year"] == latest_year, "market_share"].iloc[0]
                if not ft_df[ft_df["year"] == latest_year].empty else 0
            )
            fuel_share_shift[ft] = latest_share - first_share

            # CAGR (if sales > 0)
            first_sales = (
                ft_df.loc[ft_df["year"] == earliest_year, "total_sales"].iloc[0]
                if not ft_df[ft_df["year"] == earliest_year].empty else 0
            )
            latest_sales = (
                ft_df.loc[ft_df["year"] == latest_year, "total_sales"].iloc[0]
                if not ft_df[ft_df["year"] == latest_year].empty else 0
            )
            
            cagr = 0.0
            if first_sales > 0 and n_years > 1:
                 # CAGR formula: ((End/Start)^(1/(N-1)) - 1) * 100
                 cagr = ((latest_sales / first_sales) ** (1 / (n_years - 1)) - 1) * 100
            
            fuel_cagr[ft] = cagr
    
    # 3. Identify Top/Bottom Share Gainers
    share_series = pd.Series(fuel_share_shift)
    top_share_gainer = share_series.idxmax()
    top_share_loser = share_series.idxmin()

    analysis_summary = {
        "insight_key": "FuelType_MarketShare_Evolution",
        "description": (
            "Grouped bar chart showing the absolute sales trend of each fuel type. "
            "Analysis focuses on the evolution of market share and the relative growth rate (CAGR) of new energy types."
        ),
        "findings": {
            "years": years,
            "fuel_types": fuel_types,
            
            # Market Share Shift Analysis (NEW)
            "top_market_share_gainer": top_share_gainer,
            "top_market_share_loser": top_share_loser,
            "top_gainer_shift": f"{fuel_share_shift[top_share_gainer]:.1f}pp", # pp = percentage point
            
            # Growth Metrics (NEW)
            "fuel_type_cagr": {ft: f"{fuel_cagr[ft]:.1f}%" for ft in fuel_types},
            
            # New Energy Focus
            "new_energy_types": new_energy_types,
            "new_energy_cagr": {ft: f"{fuel_cagr[ft]:.1f}%" for ft in new_energy_types},
            
            # Raw Data (Market Share included for full context)
            "market_share_records": df_metrics.to_dict("records"),
        },
    }

    return chart_path, analysis_summary



def plot_transmission_trend_grouped(trans_trend_df: pd.DataFrame):
    """
    Grouped bar chart with trend lines for transmission-type sales by year.

    Returns:
        tuple[str, dict]:
            chart_path: saved chart path
            analysis_summary: structured insights for LLM analysis
    """

    # ---- 1. Prepare year × transmission matrix ----
    years = sorted(trans_trend_df["year"].dropna().unique().tolist())
    transmissions = sorted(trans_trend_df["transmission"].dropna().unique().tolist())

    sales_matrix = pd.DataFrame(
        0.0,
        index=years,
        columns=transmissions
    )

    for _, row in trans_trend_df.iterrows():
        y = int(row["year"])
        tr = row["transmission"]
        sales_matrix.loc[y, tr] = row["total_sales"]

    # ---- 2. Plot grouped bars + trend lines ----
    fig, ax = plt.subplots(figsize=(11, 6))

    x = np.arange(len(years))
    n_tr = len(transmissions)
    bar_width = 0.8 / n_tr

    cmap = plt.get_cmap("Set2")
    colors = [cmap(i) for i in range(n_tr)]

    for i, tr in enumerate(transmissions):
        y_vals = sales_matrix[tr].values
        bar_positions = x - 0.8 / 2 + bar_width / 2 + i * bar_width

        # Bars (slightly transparent to blend with trend lines)
        ax.bar(
            bar_positions,
            y_vals,
            width=bar_width,
            color=colors[i],
            alpha=0.6,
            label=f"{tr} (Bar)" if i == 0 else None
        )

        # Trend lines
        ax.plot(
            x,
            y_vals,
            marker="o",
            linewidth=2,
            color=colors[i],
            label=tr
        )

    # ---- 3. Axis formatting ----
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.set_ylim(
        max(trans_trend_df["total_sales"]) * 0.8,
        max(trans_trend_df["total_sales"]) * 1.03
    )

    def millions_formatter(val, _):
        if val >= 1e6:
            return f"{val/1e6:.1f}M"
        elif val >= 1e3:
            return f"{val/1e3:.0f}K"
        return f"{val:.0f}"

    ax.yaxis.set_major_formatter(FuncFormatter(millions_formatter))
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    # Unified title/axis styling
    apply_common_format(
        ax,
        title="Transmission Sales in Used-Car Market by Year\n(Grouped Bars + Trend Lines)",
        xlabel="Year",
        ylabel="Total Sales Volume (units)",
    )

    # ---- Legend: remove "(Bar)" helper labels ----
    handles, labels = ax.get_legend_handles_labels()
    filtered = [(h, l) for h, l in zip(handles, labels) if "(Bar)" not in l]
    if filtered:
        handles, labels = zip(*filtered)
        ax.legend(
            handles,
            labels,
            title="Transmission",
            bbox_to_anchor=(1.02, 1),
            loc="upper left"
        )
    else:
        ax.legend()

    fig.tight_layout()

    # ---- 4. Save chart ----
    chart_path = os.path.join(PLOTS_DIR, "transmission_trend_grouped_bar_line.png")
    fig.savefig(chart_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---- 5. Analysis Summary (Optimized) ----
    
    latest_year = int(max(years))
    earliest_year = int(min(years))
    n_years = len(years)

    # 1. Calculate Market Share for all years
    annual_total_sales = (
        trans_trend_df.groupby("year")["total_sales"].sum().rename("annual_total_sales")
    )
    df_metrics = trans_trend_df.merge(annual_total_sales, on="year")
    df_metrics["market_share"] = (
        df_metrics["total_sales"] / df_metrics["annual_total_sales"]
    ) * 100

    # 2. Analyze Share Shift and CAGR
    trans_share_shift = {}
    trans_cagr = {}

    for tr in transmissions:
        tr_df = df_metrics[df_metrics["transmission"] == tr].sort_values("year")

        if not tr_df.empty:
            # Market Share Shift (Latest Year Share - Earliest Year Share)
            first_share_row = tr_df[tr_df["year"] == earliest_year]
            latest_share_row = tr_df[tr_df["year"] == latest_year]
            
            first_share = float(first_share_row["market_share"].iloc[0]) if not first_share_row.empty else 0.0
            latest_share = float(latest_share_row["market_share"].iloc[0]) if not latest_share_row.empty else 0.0
            
            trans_share_shift[tr] = latest_share - first_share

            # CAGR (Compound Annual Growth Rate)
            first_sales = float(first_share_row["total_sales"].iloc[0]) if not first_share_row.empty else 0.0
            latest_sales = float(latest_share_row["total_sales"].iloc[0]) if not latest_share_row.empty else 0.0
            
            cagr = 0.0
            if first_sales > 0 and n_years > 1:
                 # CAGR formula: ((End/Start)^(1/(N-1)) - 1) * 100
                 cagr = ((latest_sales / first_sales) ** (1 / (n_years - 1)) - 1) * 100
            
            trans_cagr[tr] = cagr
    
    # 3. Identify Top/Bottom Share Movers
    share_series = pd.Series(trans_share_shift)
    top_share_gainer = share_series.idxmax()
    top_share_loser = share_series.idxmin()
    
    # Top absolute sales in latest year (for context)
    latest_rows = df_metrics[df_metrics["year"] == latest_year]
    top_latest_sales = latest_rows.loc[latest_rows["total_sales"].idxmax()]


    analysis_summary = {
        "insight_key": "Transmission_MarketShare_Shift",
        "description": (
            "Grouped chart showing the absolute sales trend of transmission types. "
            "Analysis focuses on quantifying the market share shift and preference evolution over the years."
        ),
        "findings": {
            "years": years,
            "transmissions": transmissions,
            
            # 市场支配地位 (Dominance)
            "dominant_transmission_latest_year": str(top_latest_sales["transmission"]),
            "dominant_share_latest_year": f"{top_latest_sales['market_share']:.1f}%",
            
            # 市场份额转变分析 (Market Shift Analysis)
            "top_market_share_gainer": top_share_gainer,
            "top_gainer_shift": f"{trans_share_shift[top_share_gainer]:.1f}pp", # pp = percentage point
            "top_market_share_loser": top_share_loser,
            "top_loser_shift": f"{trans_share_shift[top_share_loser]:.1f}pp",
            
            # 增长指标 (Growth Metrics)
            "transmission_cagr": {tr: f"{trans_cagr[tr]:.1f}%" for tr in transmissions},
            
            # 原始数据 (包含市场份额供完整参考)
            "market_share_records": df_metrics.to_dict("records"),
        },
    }

    return chart_path, analysis_summary



def plot_color_sales_bar(color_df: pd.DataFrame):
    """
    Bar chart showing total sales volume by vehicle color (market preference analysis).

    Returns:
        tuple[str, dict]:
            chart_path: saved chart path
            analysis_summary: structured insights for LLM analysis
    """

    # ---- 1. Prepare data ----
    colors = color_df["color"].tolist()
    sales = color_df["total_sales"].tolist()

    # ---- 2. Plot bar chart ----
    fig, ax = plt.subplots(figsize=(9, 5))

    cmap = plt.get_cmap("Set2")
    bar_colors = [
        cmap(i % len(cmap.colors)) if hasattr(cmap, "colors")
        else cmap(i / max(1, len(colors) - 1))
        for i in range(len(colors))
    ]

    bars = ax.bar(
        colors,
        sales,
        color=bar_colors,
        alpha=0.85
    )

    # Add numerical labels
    ax.bar_label(bars, fmt="%.0f", fontsize=10)

    # ---- 3. Keep your custom Y-range (do not modify) ----
    ax.set_ylim(12000000, 15000000)

    # Y-axis formatting (K/M style)
    def formatter(val, _):
        if val >= 1e6:
            return f"{val/1e6:.1f}M"
        elif val >= 1e3:
            return f"{val/1e3:.0f}K"
        return f"{val:.0f}"

    ax.yaxis.set_major_formatter(FuncFormatter(formatter))
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    # Unified formatting (title, labels, font)
    apply_common_format(
        ax,
        title="Total Sales Volume by Color (Market Preference Analysis)",
        xlabel="Color",
        ylabel="Total Sales Volume (units)",
    )

    fig.tight_layout()

    # ---- 4. Save chart ----
    chart_path = os.path.join(PLOTS_DIR, "color_sales_bar.png")
    fig.savefig(chart_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---- 5. Analysis Summary (Optimized) ----
    
    # Ensure data is sorted by total_sales descending
    df_sorted = color_df.sort_values("total_sales", ascending=False).reset_index(drop=True)
    total_sales_all = df_sorted["total_sales"].sum()
    
    top_row = df_sorted.iloc[0]
    bottom_row = df_sorted.iloc[-1]
    
    # 1. Calculate Market Share for Top Color
    top_sales = float(top_row["total_sales"])
    top_color_share = (top_sales / total_sales_all) * 100
    
    # 2. Calculate Disparity Ratio (Top vs. Bottom)
    bottom_sales = float(bottom_row["total_sales"])
    disparity_ratio = (top_sales / bottom_sales) if bottom_sales > 0 else float('inf')
    
    # 3. Calculate Top N Concentration (e.g., Top 3)
    N_concentration = min(3, len(df_sorted))
    top_n_sales = df_sorted["total_sales"].head(N_concentration).sum()
    concentration_ratio = (top_n_sales / total_sales_all) * 100
    
    # Create records including market share for LLM context
    df_sorted['market_share'] = (df_sorted['total_sales'] / total_sales_all) * 100
    
    analysis_summary = {
        "insight_key": "Color_Sales_Preference_Concentration",
        "description": (
            "Bar chart showing aggregated total sales for each vehicle color. "
            "Analysis focuses on quantifying the market preference concentration and sales disparity between color options."
        ),
        "findings": {
            # 1. Dominance and Share
            "top_color": str(top_row["color"]),
            "top_color_sales_volume": int(top_sales),
            "top_color_market_share": f"{top_color_share:.1f}%", # Top color's percentage of total sales
            
            # 2. Market Concentration Metrics
            "concentration_n": N_concentration,
            "top_n_concentration_ratio": f"{concentration_ratio:.1f}%", # Top N colors' combined market share
            
            # 3. Disparity Metric
            "least_color": str(bottom_row["color"]),
            "preference_disparity_ratio_top_to_bottom": f"{disparity_ratio:.1f}x", # How many times more popular is the top color
            
            # 4. Full Data for LLM context (including share)
            "color_sales_with_share": [
                {
                    "color": row["color"],
                    "total_sales": int(row["total_sales"]),
                    "market_share": f'{row["market_share"]:.1f}%'
                }
                for _, row in df_sorted.iterrows()
            ],
        },
    }

    return chart_path, analysis_summary



def plot_engine_size_sales_bar(eng_df: pd.DataFrame):
    """
    Bar chart showing total sales volume across engine-size categories
    (ordered from small to large displacement).

    Returns:
        tuple[str, dict]:
            chart_path: saved chart path
            analysis_summary: structured insights for LLM analysis
    """

    # ---- 1. Enforce ordering: small → large engine displacement ----
    ordered_labels = ["<=1.5L", "1.6–2.0L", "2.1–3.0L", "3.1–4.0L", ">4.0L"]
    eng_df["engine_bin"] = pd.Categorical(
        eng_df["engine_bin"],
        categories=ordered_labels,
        ordered=True
    )
    eng_df = eng_df.sort_values("engine_bin")

    bins = eng_df["engine_bin"].astype(str).tolist()
    sales = eng_df["total_sales"].tolist()

    # ---- 2. Plot bar chart (Set2 color scheme) ----
    fig, ax = plt.subplots(figsize=(9, 5))

    cmap = plt.get_cmap("Set2")
    bar_colors = [
        cmap(i % len(cmap.colors)) if hasattr(cmap, "colors")
        else cmap(i / max(1, len(bins) - 1))
        for i in range(len(bins))
    ]

    bars = ax.bar(bins, sales, color=bar_colors, alpha=0.85)
    ax.bar_label(bars, fmt="%.0f", fontsize=10)

    # Y-axis formatter (K/M format)
    def formatter(val, _):
        if val >= 1e6:
            return f"{val/1e6:.1f}M"
        elif val >= 1e3:
            return f"{val/1e3:.0f}K"
        return f"{val:.0f}"

    ax.yaxis.set_major_formatter(FuncFormatter(formatter))
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    # Apply unified formatting (title, labels, style)
    apply_common_format(
        ax,
        title="Total Sales by Engine Size (Market Preference Analysis)",
        xlabel="Engine Size Category",
        ylabel="Total Sales Volume (units)",
    )

    fig.tight_layout()

    # ---- 3. Save output ----
    chart_path = os.path.join(PLOTS_DIR, "engine_size_sales_bar.png")
    fig.savefig(chart_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---- 4. Analysis Summary (Optimized) ----
    
    # 确保数据是按引擎排量类别排序的（保持原始顺序）
    df_sorted = eng_df.copy()
    total_sales_all = df_sorted["total_sales"].sum()
    
    # 计算 Market Share
    df_sorted["market_share"] = (
        df_sorted["total_sales"] / total_sales_all
    ) * 100
    
    # 根据销量重新排序，以确定 Top/Bottom
    df_sales_sorted = df_sorted.sort_values("total_sales", ascending=False)
    
    top_row = df_sales_sorted.iloc[0]
    bottom_row = df_sales_sorted.iloc[-1]
    
    # 1. 量化主流地位 (Top Bin)
    top_bin = str(top_row["engine_bin"])
    top_val = float(top_row["total_sales"])
    top_share = float(top_row["market_share"])
    
    # 2. 量化集中度 (Top N, e.g., Top 2)
    N_concentration = min(2, len(df_sales_sorted))
    top_n_sales = df_sales_sorted["total_sales"].head(N_concentration).sum()
    concentration_ratio = (top_n_sales / total_sales_all) * 100
    
    # 3. 差距比率 (Top vs. Bottom)
    bottom_val = float(bottom_row["total_sales"])
    disparity_ratio = (top_val / bottom_val) if bottom_val > 0 else float('inf')


    analysis_summary = {
        "insight_key": "EngineSize_Sales_Distribution_Concentration",
        "description": (
            "Bar chart comparing sales performance across engine-size categories. "
            "Analysis focuses on identifying the mainstream displacement market segment and quantifying sales concentration."
        ),
        "definition": {
            "<=1.5L": "small engines",
            "1.6–2.0L": "compact mainstream engines",
            "2.1–3.0L": "mid-size or premium engines",
            "3.1–4.0L": "high-displacement engines",
            ">4.0L": "performance or luxury engines",
        },
        "findings": {
            # 1. 支配地位和主流排量
            "mainstream_engine_bin": top_bin,
            "mainstream_bin_market_share": f"{top_share:.1f}%", # 主流排量占总销量的百分比
            
            # 2. 集中度指标
            "concentration_n": N_concentration,
            "top_n_concentration_ratio": f"{concentration_ratio:.1f}%", # 前N个排量区间的总市场份额
            
            # 3. 差距指标
            "least_popular_bin": str(bottom_row["engine_bin"]),
            "sales_disparity_ratio_top_to_bottom": f"{disparity_ratio:.1f}x", # 最畅销与最不畅销排量的倍数差距
            
            # 4. 完整数据 (包含市场份额)
            "engine_bin_sales_with_share": [
                {
                    "category": str(row["engine_bin"]),
                    "total_sales": int(row["total_sales"]),
                    "market_share": f'{row["market_share"]:.1f}%'
                }
                for _, row in df_sorted.iterrows()
            ],
        }
    }

    return chart_path, analysis_summary


def plot_engine_size_sales_bar(eng_df: pd.DataFrame) -> tuple[str, dict]:
    """
    Plot total sales volume across engine-size categories
    (ordered from small to large displacement).

    Args:
        eng_df (pd.DataFrame):
            Aggregated dataframe with columns:
            - engine_bin: engine-size category
            - total_sales: aggregated sales volume

    Returns:
        tuple[str, dict]:
            chart_path: saved chart path
            analysis_summary: structured insights for LLM analysis
    """

    # ---- 1. Enforce sorted engine-size order: small → large ----
    ordered_labels = ["<=1.5L", "1.6–2.0L", "2.1–3.0L", "3.1–4.0L", ">4.0L"]
    eng_df["engine_bin"] = pd.Categorical(
        eng_df["engine_bin"],
        categories=ordered_labels,
        ordered=True
    )
    eng_df = eng_df.sort_values("engine_bin")

    bins = eng_df["engine_bin"].astype(str).tolist()
    sales = eng_df["total_sales"].tolist()

    # ---- 2. Plot bar chart (Set2 color scheme) ----
    fig, ax = plt.subplots(figsize=(9, 5))

    cmap = plt.get_cmap("Set2")
    bar_colors = [
        cmap(i % len(cmap.colors)) if hasattr(cmap, "colors")
        else cmap(i / max(1, len(bins) - 1))
        for i in range(len(bins))
    ]

    bars = ax.bar(bins, sales, color=bar_colors, alpha=0.85)
    ax.bar_label(bars, fmt="%.0f", fontsize=10)

    # ---- 3. Y-axis formatting (K/M units) ----
    def formatter(val, _):
        if val >= 1e6:
            return f"{val/1e6:.1f}M"
        elif val >= 1e3:
            return f"{val/1e3:.0f}K"
        return f"{val:.0f}"

    ax.yaxis.set_major_formatter(FuncFormatter(formatter))
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    # ---- 4. Apply unified plot style ----
    apply_common_format(
        ax,
        title="Total Sales by Engine Size (Market Preference Analysis)",
        xlabel="Engine Size Category",
        ylabel="Total Sales Volume (units)",
    )

    fig.tight_layout()

    # ---- 5. Save figure ----
    chart_path = os.path.join(PLOTS_DIR, "engine_size_sales_bar.png")
    fig.savefig(chart_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---- 6. Analysis Summary (Optimized) ----
    
    # 确保数据是按销量降序排列，以便确定 Top/Bottom
    df_sales_sorted = eng_df.sort_values("total_sales", ascending=False).reset_index(drop=True)
    total_sales_all = df_sales_sorted["total_sales"].sum()
    
    # 计算 Market Share
    df_sales_sorted["market_share"] = (
        df_sales_sorted["total_sales"] / total_sales_all
    ) * 100
    
    top_row = df_sales_sorted.iloc[0]
    bottom_row = df_sales_sorted.iloc[-1]
    
    # 1. 量化主流地位 (Top Bin)
    top_bin = str(top_row["engine_bin"])
    top_val = float(top_row["total_sales"])
    top_share = float(top_row["market_share"])
    
    # 2. 量化集中度 (Top N, e.g., Top 2)
    # Concentration on the top two most popular engine sizes
    N_concentration = min(2, len(df_sales_sorted))
    top_n_sales = df_sales_sorted["total_sales"].head(N_concentration).sum()
    concentration_ratio = (top_n_sales / total_sales_all) * 100
    
    # 3. 差距比率 (Top vs. Bottom)
    bottom_val = float(bottom_row["total_sales"])
    disparity_ratio = (top_val / bottom_val) if bottom_val > 0 else float('inf')


    analysis_summary = {
        "insight_key": "EngineSize_Sales_Distribution_Concentration",
        "description": (
            "Bar chart comparing sales performance across engine-size categories. "
            "Analysis focuses on identifying the **mainstream displacement market segment** and quantifying sales concentration."
        ),
        "definition": {
            "<=1.5L": "small engines",
            "1.6–2.0L": "compact mainstream engines",
            "2.1–3.0L": "mid-size or premium engines",
            "3.1–4.0L": "high-displacement engines",
            ">4.0L": "performance or luxury engines",
        },
        "findings": {
            # 1. 支配地位和主流排量 (Dominance)
            "mainstream_engine_bin": top_bin,
            "mainstream_bin_market_share": f"{top_share:.1f}%", # 主流排量占总销量的百分比
            
            # 2. 集中度指标 (Concentration)
            "concentration_n": N_concentration,
            "top_n_concentration_ratio": f"{concentration_ratio:.1f}%", # 前N个排量区间的总市场份额
            
            # 3. 差距指标 (Disparity)
            "least_popular_bin": str(bottom_row["engine_bin"]),
            "sales_disparity_ratio_top_to_bottom": f"{disparity_ratio:.1f}x", # 最畅销与最不畅销排量的倍数差距
            
            # 4. 完整数据 (包含市场份额供 LLM 撰写分析)
            "engine_bin_sales_with_share": [
                {
                    "category": str(row["engine_bin"]),
                    "total_sales": int(row["total_sales"]),
                    "market_share": f'{row["market_share"]:.1f}%'
                }
                for _, row in df_sales_sorted.iterrows()
            ],
        }
    }

    return chart_path, analysis_summary
