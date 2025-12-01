# Automated LLM Report Generation

This repository contains the complete pipeline for data analysis and modeling on [e.g., BMW Sales Data (2020-2024)], culminating in automated report generation.
- **Input:** Structured dataset (Excel / CSV)  
- **Output:** Complete business report in: `.md` (Markdown) or `.html` (HTML) or `.pdf` (PDF)

## Quick entries:
Final reports: `reports/final_report.pdf`

Executive summary: `executive_summary/executive_summary.pdf`

---


## 📂 Repository Structure

This project adopts a structured layout based on the data science workflow, organizing components by their function:

| Folder/File | Purpose |
| :--- | :--- |
| **`reports/`** | **Final Outputs.** Stores the final reports (`final_report.pdf`, `.html`, `.md`) and all generated figures (`report_figures/`). |
| **`executive_summary/`**| **Executive summary.** Contains the required deliverables "executive summary" PDF. |
| **`data/`** | **Data Storage.** Separated into `raw/` (for immutable source files) and `processed/` (for clean, analysis-ready data). |
| **`src/`** | **Source Code.** Contains all reusable Python modules (`data_loader.py`, `analysis_module.py`, `report_generator.py`, etc.) that execute the pipeline logic. |
| **`notebooks/`** | **ML Prototypes.** Used for exploratory of multiple ML forecast models before integrating code into `src/`. |
| **`tmp/`** | **ML Model Storage.** Stores the training logging of Catboost model. |
| **`external_src/`**| **External Resources.** Stores non-code assets, such as geographical shapefiles used for specialized visualization. |
| **`main.py`** | **Project Entry Point.** The central script to run the entire data pipeline and generate all outputs. |
| **`environment.yml`** | **Environment Configuration.** Specifies all necessary Python dependencies and versions for project reproducibility using Conda. |

---

## 🚀 Getting Started
### 1. Clone the Repository

```bash
# Clone the repository
git clone https://github.com/lixiaoqingnnz/Data_Analysis_Li_Xiaoqing.git

# Enter the project directory
cd Data_Analysis_Li_Xiaoqing
```


### 2. Environment Preparation (System Prerequisites)

To ensure successful conversion of Markdown (`.md`) files to PDF, the following **non-Python** system dependencies must be installed:

**macOS**
```bash
brew install pandoc
brew install wkhtmltopdf
brew install libomp
```
**Ubuntu**
```bash
sudo apt-get install pandoc
sudo apt-get install wkhtmltopdf
sudo apt-get install libomp-dev
```

### 3. Conda Environment Setup

This project uses Conda for managing Python dependencies. Follow these steps to set up and activate the project environment:
**Create and Activate Environment:** Use the provided `environment.yml` file to create a dedicated Conda environment.

```bash
conda env create -f environment.yml

conda activate dataanalysis_clean
```

### 4. OPENROUTER API Key Setting

Replace the API key in `.env` with your actual OPENROUTER API key.

### 5. Project Execution

Once the environment is activated, you can execute the entire data pipeline (loading, processing, analyzing, and reporting) by running the main script:

```bash
python main.py --file data/raw/BMW_sales_data_(2020-2024).xlsx
```

You should be able to see your final reports in `reports/final_report.pdf`