import argparse
import os
import logging
import json

from src.data_loader import load_data
from src.data_processor import sales_volume_value_year, sales_region_year, total_sales_volume_region, total_sales_value_region, region_price, model_performance, mileage_sales, analyze_fuel_type_by_region, analyze_fuel_type_trend, analyze_transmission_trend, analyze_color_sales, analyze_engine_size_sales
from src.visualization_module import plot_sales_volume_value_year, plot_region_annual_sales_radar, plot_total_sales_region_3d, plot_price_box_region, plot_model_performance_bar,plot_mileage_sales, plot_fuel_type_by_region_facets, plot_fuel_type_trend_grouped, plot_transmission_trend_grouped, plot_color_sales_bar, plot_engine_size_sales_bar
from src.analysis_module import catboost_forecast_top3
from src.llm_orchestrator import LLMOrchestrator
from src.report_generator import generate_full_report


logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)

logging.info(">>> Starting BMW used-car analytics data pipeline...")


def main(file_path: str):
    """
    Runs the complete automated report generation workflow.

    Args:
        file_path (str): The path to the input data file.
    """
    orchestrator = LLMOrchestrator()

    logging.info(f"--- [Step 1] Loading and cleaning data from {file_path} ---")

    # ---- 1. Load & clean data ----
    try:
        df_clean = load_data(file_path)
        logging.info(">>> [Step 1] Data loaded and cleaned successfully.")
    except FileNotFoundError:
        logging.error(f"Error: File not found at {file_path}")
        return
    except Exception as e:
        logging.error(f"Data loading failed: {e}")
        return

    # ---- 2. Main data analysis ----
    logging.info(f"--- [Step 2] Starting data analysis pipeline ---")

    report_data: list[dict] = []

    # =========================================================
    # SECTION 1 — Trends by year
    # =========================================================
    
    section_title_trends = "Annual Performance and Market Structure Dynamics"

    # 1.1 Annual sales trends
    subsection_title = "Comprehensive Annual Sales Volume and Revenue Trend"
    df_sales_volume_value_year = sales_volume_value_year(df_clean)
    chart_path_sales_volume_value_year, summary_sales_volume_value_year = plot_sales_volume_value_year(
        df_sales_volume_value_year
    )
    narrative_1 = orchestrator.generate_chart_narrative(
        section_title=subsection_title,
        data_summary=summary_sales_volume_value_year,
    )
    report_data.append({
        "section_title": section_title_trends,
        "subsection_title": subsection_title,
        "chart_path": chart_path_sales_volume_value_year,
        "narrative": narrative_1,
    })
    logging.info(">>> [1.1 Annual sales trends] Data processed, chart generated, and narrative created.")

    # 1.2 Fuel type trend by year
    subsection_title = "Annual Shift in Fuel Type Market Preference"
    df_fuel_type_year = analyze_fuel_type_trend(df_clean)
    chart_path_fuel_type_year, summary_fuel_type_year = plot_fuel_type_trend_grouped(df_fuel_type_year)
    narrative_2 = orchestrator.generate_chart_narrative(
        section_title=subsection_title,
        data_summary=summary_fuel_type_year,
    )
    report_data.append({
        "section_title": section_title_trends,
        "subsection_title": subsection_title,
        "chart_path": chart_path_fuel_type_year,
        "narrative": narrative_2,
    })
    logging.info(">>> [1.2 Fuel type trend by year] Data processed, chart generated, and narrative created.")

    # 1.3 Transmission type (manual/automatic) trend by year
    subsection_title = "Annual Shift in Transmission Type Market Preference"
    df_transmission_trend_year = analyze_transmission_trend(df_clean)
    chart_path_transmission_type_year, summary_transmission_type_year = plot_transmission_trend_grouped(
        df_transmission_trend_year
    )
    narrative_3 = orchestrator.generate_chart_narrative(
        section_title=subsection_title,
        data_summary=summary_transmission_type_year,
    )
    report_data.append({
        "section_title": section_title_trends,
        "subsection_title": subsection_title,
        "chart_path": chart_path_transmission_type_year,
        "narrative": narrative_3,
    })
    logging.info(">>> [1.3 Transmission trend by year] Data processed, chart generated, and narrative created.")

    # =========================================================
    # SECTION 2 — Regional Market Performance & Pricing Strategy
    # =========================================================
    section_title_market = "Market segmentation by region"

    # 2.1 Regional annual sales distribution (radar)
    subsection_title = "Comparative Analysis of Regional Market Performance Patterns"
    df_region_sales = sales_region_year(df_clean)
    chart_path_region_sales, summary_region_sales = plot_region_annual_sales_radar(df_region_sales)
    narrative_4 = orchestrator.generate_chart_narrative(
        section_title=subsection_title,
        data_summary=summary_region_sales
    )
    report_data.append({
        "section_title": section_title_market,
        "subsection_title": subsection_title,
        "chart_path": chart_path_region_sales,
        "narrative": narrative_4,
    })
    logging.info(">>> [2.1 Regional annual sales distribution] Data processed, chart generated, and narrative created.")

    # 2.2 Total sales volume by region (3D map)
    subsection_title = "Regional Concentration Analysis of Total Sales Volume"
    df_total_sales_volume_region = total_sales_volume_region(df_clean)
    total_sales_region_plot_path, total_sales_region_summary = plot_total_sales_region_3d(
        df_total_sales_volume_region
    )
    narrative_5 = orchestrator.generate_chart_narrative(
        section_title=subsection_title,
        data_summary=total_sales_region_summary,
    )
    report_data.append({
        "section_title": section_title_market,
        "subsection_title": subsection_title,
        "chart_path": total_sales_region_plot_path,
        "narrative": narrative_5,
    })
    logging.info(">>> [2.2 Total sales volume by region] Data processed, chart generated, and narrative created.")

    # 2.3 Price distribution by region (violin)
    subsection_title = "Full Distribution and Dispersion of Used Car Transaction Prices by Region"
    df_region_price = region_price(df_clean)  # almost all rows; only used for chart
    chart_path_price_box_region, summary_price_box_region = plot_price_box_region(df_region_price)
    narrative_6 = orchestrator.generate_chart_narrative(
        section_title=subsection_title,
        data_summary=summary_price_box_region,
    )
    report_data.append({
        "section_title": section_title_market,
        "subsection_title": subsection_title,
        "chart_path": chart_path_price_box_region,
        "narrative": narrative_6,
    })
    logging.info(">>> [2.3 Price distribution by region] Data processed, chart generated, and narrative created.")

    # 2.4 Fuel type distribution by region (facets)
    subsection_title = "Regional Differentiation of Fuel Type Preferences"
    df_fuel_type_region = analyze_fuel_type_by_region(df_clean)
    chart_path_fuel_type_region, summary_fuel_type_region = plot_fuel_type_by_region_facets(df_fuel_type_region)
    narrative_7 = orchestrator.generate_chart_narrative(
        section_title=subsection_title,
        data_summary=summary_fuel_type_region,
    )
    report_data.append({
        "section_title": section_title_market,
        "subsection_title": subsection_title,
        "chart_path": chart_path_fuel_type_region,
        "narrative": narrative_7,
    })
    logging.info(">>> [2.4 Fuel type distribution by region] Data processed, chart generated, and narrative created.")

    # =========================================================
    # SECTION 3 — Product and consumer preferences
    # =========================================================
    section_title_pref = "Product Competitiveness & Consumer Preference Insights"

    # 3.1 Model performance ranking
    subsection_title = "Ranking of Model Sales Volume"
    df_model_performance = model_performance(df_clean)
    model_performance_plot_path, model_performance_summary = plot_model_performance_bar(df_model_performance)
    narrative_8 = orchestrator.generate_chart_narrative(
        section_title=subsection_title,
        data_summary=model_performance_summary,
    )
    report_data.append({
        "section_title": section_title_pref,
        "subsection_title": subsection_title,
        "chart_path": model_performance_plot_path,
        "narrative": narrative_8,
    })
    logging.info(">>> [3.1 Model performance ranking] Data processed, chart generated, and narrative created.")

    # 3.2 Mileage impact on sales
    subsection_title = "Impact of Mileage on Sales Volume"
    df_mileage_sales = mileage_sales(df_clean)
    chart_path_mileage_sales, summary_mileage_sales = plot_mileage_sales(df_mileage_sales)
    narrative_9 = orchestrator.generate_chart_narrative(
        section_title=subsection_title,
        data_summary=summary_mileage_sales,
    )
    report_data.append({
        "section_title": section_title_pref,
        "subsection_title": subsection_title,
        "chart_path": chart_path_mileage_sales,
        "narrative": narrative_9,
    })
    logging.info(">>> [3.2 Mileage impact on sales] Data processed, chart generated, and narrative created.")

    # 3.3 Color preference analysis
    subsection_title = "Impact of Exterior Color on Sales and Hot Market Colors"
    df_color_sales = analyze_color_sales(df_clean)
    chart_path_color_sales, summary_color_sales = plot_color_sales_bar(df_color_sales)
    narrative_10 = orchestrator.generate_chart_narrative(
        section_title=subsection_title,
        data_summary=summary_color_sales,
    )
    report_data.append({
        "section_title": section_title_pref,
        "subsection_title": subsection_title,
        "chart_path": chart_path_color_sales,
        "narrative": narrative_10,
    })
    logging.info(">>> [3.3 Color preference analysis] Data processed, chart generated, and narrative created.")

    # 3.4 Engine-size market preference
    subsection_title = "Demand Structure of Engine Size"
    df_engine_size_sales = analyze_engine_size_sales(df_clean)
    chart_path_engine_size_sales, summary_engine_size_sales = plot_engine_size_sales_bar(df_engine_size_sales)
    narrative_11 = orchestrator.generate_chart_narrative(
        section_title=subsection_title,
        data_summary=summary_engine_size_sales,
    )
    report_data.append({
        "section_title": section_title_pref,
        "subsection_title": subsection_title,
        "chart_path": chart_path_engine_size_sales,
        "narrative": narrative_11,
    })
    logging.info(">>> [3.4 Engine-size market preference] Data processed, chart generated, and narrative created.")
    
    

    # =========================================================
    # SECTION 4 — Sales Forecast and Strategic Segment Growth (2025)
    # =========================================================
    section_title_forecast = "Sales Forecast and Strategic Segment Growth (2025)"
    # subsection_title = "Strategic Top Segment Prediction (2024 Actual vs 2025 Forecast)"
    metrics_2024, total_2025, combined_top3_df, md_combined_top3, summary_forecast = catboost_forecast_top3(df_clean)
    narrative_12 = orchestrator.generate_chart_narrative(
        section_title=subsection_title,
        data_summary=summary_forecast,
    )
    report_data.append({
        "section_title": section_title_forecast,
        # "subsection_title": subsection_title,
        "chart_path": "", 
        "narrative": narrative_12,
        "table_content": md_combined_top3, 
    })
    logging.info(">>> [4.1 Sales forecase and strategic] Data processed, table generated, and narrative created.")

    # =========================================================
    # SECTION 5 — Summary
    # =========================================================
    section_title_summary = "Summary"
    all_narratives = [narrative_1, narrative_2, narrative_3, narrative_4, narrative_5, narrative_6, narrative_7, narrative_8, narrative_9, narrative_10, narrative_11, narrative_12]
    narrative_summary = orchestrator.generate_executive_summary(
        all_narratives = all_narratives
    )
    report_data.append({
        "section_title": section_title_summary,
        "narrative": narrative_summary,
    })
    logging.info(">>> [5 Summary] Narrative created.")

    logging.info("--- [Step 3] Reporting data generation completed ---")
    # temp save report data 
    with open("reports/temp_report_data.json", "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=4, ensure_ascii=False)
    logging.info(">>> [Step 4] Report data saved to reports/temp_report_data.json")

    return report_data



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated LLM-Powered Business Reporting.")
    parser.add_argument(
        '--file', 
        type=str, 
        default='data/raw/BMW_sales_data_(2020-2024).xlsx', 
        help='Path to input data file'
    )
    args = parser.parse_args()
    # run main function 
    # report_data = main(args.file)
    generate_full_report()
