# ABxpress - AI assisted A/B test analyzer

ABxpress analyzes A/B test CSVs where each row is a user or session and variants are control vs treatment. It computes lift, confidence intervals, p values, power, and plots. It also includes segment filtering and a sample size calculator.

## Demo in 30 seconds
1. `pip install -r requirements.txt`
2. `streamlit run app.py`
3. In the sidebar: Load sample data - or upload your CSV - choose columns - view results.

## Expected CSV format
- One row per user or session
- Required: a variant column with control and treatment, a metric column
- Optional: a date column for time series, a segment column for slicing

Example:

| user_id | variant   | converted | revenue | date       | platform |
|--------:|-----------|-----------|---------|------------|----------|
| 1       | control   | 0         | 0.00    | 2025-01-01 | PS5      |
| 2       | treatment | 1         | 5.99    | 2025-01-01 | PS5      |

## How to use
- Upload your CSV or click Load sample data
- Pick Variant and Metric from dropdowns
- Metric type can be Auto detect or manual (binary or numeric)
- Optional: add Date and Segment if you have them
- Review KPIs, charts, Segment view
- Use the Sample size calculator for planning

## What the metrics mean
- p value - smaller means stronger evidence against no effect (0.05 is a common cutoff)
- 95 percent CI for lift - plausible range of the difference
- Power - chance to detect the effect if it is real
- Absolute lift - treatment minus control
- Relative lift - absolute lift divided by control

## Features
- Column auto suggestions
- Dropdowns populated from your file
- Results and charts update on every change
- Segment view for any category
- Sample size calculator for binary metrics
- Session safe file handling so the app does not reset on interactions

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```