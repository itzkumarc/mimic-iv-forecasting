# MIMIC-IV Time-Series Forecasting Pipeline

A high-resolution, end-to-end clinical time-series forecasting pipeline built on the MIMIC-IV (v2.2 demo) dataset. This project benchmarks multiple model architectures — **Prophet**, **Stacked Attention-LSTM**, **GRU/TFT**, and a **Weighted Ensemble** — for predicting **Mean Arterial Pressure (MAP)** at a 6-hour forecast horizon in an ICU setting.

---

## Project Structure

```
.
├── src/
│   ├── 01_build_dataset.py      # Raw MIMIC-IV → Hourly time-series dataset
│   ├── 02_qc_plots.py           # Quality control and EDA visualizations
│   ├── 03_model_prophet.py      # Facebook Prophet baseline model
│   ├── 04_model_lstm.py         # 2-layer stacked Attention-LSTM
│   ├── 05_model_tft.py          # Temporal Fusion Transformer / GRU model
│   ├── 06_evaluate_and_report.py# Model evaluation and metrics generation
│   └── 07_model_ensemble.py     # Weighted predictive ensemble
├── outputs/
│   ├── qc/                      # Quality control plots
│   ├── eda/                     # Exploratory data analysis plots
│   └── results/                 # Model results, metrics, leaderboard
├── Task11_MIMIC_Forecasting_Report.tex  # LaTeX source for the technical report
├── Task11_MIMIC_Forecasting_Report.md   # Markdown analysis report
├── Task11_MIMIC_Forecasting_Analysis.ipynb  # Standalone Jupyter notebook
├── requirements.txt
└── README.md
```

---

## Data

This pipeline uses the **MIMIC-IV Clinical Database Demo v2.2** from PhysioNet.

> **Important:** The raw MIMIC-IV data files are **NOT** included in this repository (restricted clinical data). You must apply for access and download the dataset from:
> [https://physionet.org/content/mimic-iv-demo/](https://physionet.org/content/mimic-iv-demo/)

Once downloaded, place the data directory at:
```
data/mimic-iv-clinical-database-demo-2.2/
├── hosp/
│   ├── patients.csv
│   ├── admissions.csv
│   └── labevents.csv
└── icu/
    ├── icustays.csv
    ├── chartevents.csv
    └── inputevents.csv
```

---

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Full Pipeline
Run each script in order:
```bash
python src/01_build_dataset.py      # Build analysis-ready dataset
python src/02_qc_plots.py           # Generate QC and EDA plots
python src/03_model_prophet.py      # Train Prophet baseline
python src/04_model_lstm.py         # Train Attention-LSTM
python src/05_model_tft.py          # Train GRU/TFT model
python src/06_evaluate_and_report.py# Evaluate all models
python src/07_model_ensemble.py     # Build weighted ensemble
```

---

## Models and Results

| Model Architecture | MAE | RMSE | R² | MAPE |
|---|---|---|---|---|
| Facebook Prophet (Baseline) | 11.64 | 15.65 | 0.497 | 14.64% |
| Stacked Attention-LSTM | 11.72 | 14.95 | 0.517 | — |
| GRU / TFT | 12.04 | 15.39 | 0.487 | — |
| **Weighted Ensemble** | **11.40** | **14.43** | **0.540** | **13.90%** |

---

## Clinical Significance

An R² of ~0.50 in ICU hemodynamic forecasting is considered **highly significant**. In the intensive care setting, the remaining variance is primarily explained by unmeasured physiological factors (e.g., autonomic reflexes) and spontaneous variation — sources of noise that no non-invasive EHR-based model can fully capture.

---

