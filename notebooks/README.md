# Notebooks

This folder contains the two required ML notebooks:

1. `demand_forecasting.ipynb`  
   Interactive regression notebook for predicting delivery demand.

2. `anomaly_classifier.ipynb`  
   Interactive classification notebook for detecting drone anomalies.

Both notebooks use the same functions as the Streamlit app from `src/ml_pipeline.py`, so the notebook results and dashboard results stay consistent.

## How to open

From the project root:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

Then open the files inside this `notebooks/` folder.
