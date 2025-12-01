import pandas as pd
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)

OUTPUT_CSV = Path("data/processed/bmw_sales_2020_2024_clean.csv")

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names:
    - strip leading/trailing spaces
    - lowercase
    - replace spaces with underscores
    - remove non-alphanumeric characters (keep letters, digits, and underscore)
    """
    df = df.copy()
    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace(r"[^\w]", "", regex=True)
    )
    return df


def convert_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Try to convert object columns that look like numbers into numeric types:
    - remove commas and common currency symbols
    - convert to float
    - if more than 80% of values can be converted, treat the whole column as numeric
    """
    df = df.copy()
    obj_cols = df.select_dtypes(include=["object"]).columns

    for col in obj_cols:
        raw = df[col].astype(str).str.strip()

        cleaned = (
            raw.str.replace(",", "", regex=False)
               .str.replace("$", "", regex=False)
               .str.replace("€", "", regex=False)
               .str.replace("£", "", regex=False)
               .str.replace("¥", "", regex=False)
        )

        numeric = pd.to_numeric(cleaned, errors="coerce")
        success_ratio = numeric.notna().mean()

        if success_ratio > 0.8:
            df[col] = numeric

    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simple missing value handling:
    - drop rows that are completely empty
    - numeric columns:
        - if column name looks like quantity/sales -> fill with 0
        - otherwise -> fill with median
    - object/category columns -> fill with 'Unknown'
    """
    df = df.copy()

    df = df.dropna(how="all")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=["object"]).columns

    for col in numeric_cols:
        lname = col.lower()
        if any(k in lname for k in ["qty", "quantity", "volume", "units", "sales"]):
            df[col] = df[col].fillna(0)
        else:
            median = df[col].median()
            df[col] = df[col].fillna(median)

    for col in cat_cols:
        df[col] = df[col].fillna("Unknown")

    return df


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop duplicate rows.
    """
    df = df.copy()
    df = df.drop_duplicates()
    return df


def clean_workflow(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Run the full cleaning pipeline on a single sheet.
    """
    logging.info(">>> Running cleaning workflow...")

    df = df_raw.copy()

    # 1. normalize column names
    df = normalize_columns(df)

    # 2. drop completely empty rows
    df = df.dropna(how="all")

    # 3. type conversion
    df = convert_numeric(df)

    # 4. missing value handling
    df = handle_missing_values(df)

    # 5. drop duplicates
    df = drop_duplicates(df)

    logging.info(">>> Cleaning workflow completed.")
    return df


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load and clean data from a CSV or Excel file.

    Args:
        file_path (str): The path to the CSV/XLSX file.

    Returns:
        pd.DataFrame: The cleaned DataFrame after applying `clean_workflow`.
    """

    file_path = Path(file_path)
    ext = file_path.suffix.lower()

    # 1. extention check 

    if ext == ".csv":
        logging.info(f">>> Loading CSV file: {file_path}")
        df_raw = pd.read_csv(file_path)

    elif ext in [".xlsx", ".xls"]:
        logging.info(f">>> Loading Excel file (first sheet only): {file_path}")
        df_raw = pd.read_excel(file_path, sheet_name=0)

    else:
        logging.error(f">>> Unsupported file type: {ext}")
        raise ValueError(f"Unsupported file type: {ext}")

    logging.info(f">>> Loaded rows: {len(df_raw)}")

    # 2. run the clean workflow 
    df_clean = clean_workflow(df_raw)
    logging.info(f">>> Cleaned rows: {len(df_clean)}")

    # 3. Save output to CSV
    df_clean.to_csv(OUTPUT_CSV, index=False)

    logging.info(f">>> Cleaning finished. Saved to: {OUTPUT_CSV.resolve()}")

    return df_clean
