# AeroNet Lite — Interactive Drone Delivery AI Simulator

AeroNet Lite is a beginner-friendly, interactive AI semester project that demonstrates:

- **CSP layout validation** for city planning constraints
- **Fleet selection** using brute force or a small Genetic Algorithm
- **A\* path planning** for drone delivery routes
- **Real-time no-fly disruption handling** and rerouting
- **Demand forecasting** using regression
- **Anomaly detection** using classification
- **20-step integrated simulation** with visual event logs

The project is built as a **Streamlit dashboard** so it is easy to show in viva/demo.

---

## 1. Project structure

```text
aeronet_lite_interactive/
  data/
    raw/
    processed/
  src/
    app.py                  # Main interactive dashboard
    main.py                 # Console fallback demo
    grid_model.py           # Shared 10x10 grid model
    layout_validator.py     # CSP validation rules R1-R4
    fleet_selector.py       # Brute force + Genetic Algorithm fleet planning
    astar_planner.py        # A* path planner
    delivery_simulator.py   # 20-step simulation and rerouting
    ml_pipeline.py          # Demand regression + anomaly classifier
    visualization.py        # Plotly grid, heatmap, charts
  notebooks/
    demand_forecasting.ipynb
    anomaly_classifier.ipynb
  report/
  smoke_test.py
  requirements.txt
  run_app.bat
  run_app.sh
```

---

## 2. How to run on Windows

Open PowerShell or Command Prompt inside the project folder:

```bat
run_app.bat
```

That script creates a virtual environment, installs dependencies, and launches the dashboard.

Manual method:

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run src\app.py
```

---

## 3. How to run on Linux / WSL / macOS

```bash
chmod +x run_app.sh
./run_app.sh
```

Manual method:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run src/app.py
```

---

## 4. Dashboard tabs

### Dashboard
Shows the current 10x10 city grid, hubs, charging pads, no-fly cells, zone map, and demand heatmap.

### CSP Validator
Checks the four required layout rules:

- **R1:** Industrial cells cannot touch Schools or Hospitals directly.
- **R2:** Every Residential cell must be within 3 Manhattan cells of a Drone Hub.
- **R3:** Every Drone Hub must have a Charging Pad within 2 cells.
- **R4:** At least one Hospital must have a Medical Pickup point within 1 cell.

### Fleet Selector
Selects Light and Heavy drones under a budget.

Fitness function:

```text
score = (0.75 × coverage_percentage) - (0.25 × budget_used_percentage)
```

### A* Planner
Plans:

```text
hub → pickup → drop-off → hub
```

The planner avoids cells marked as `no_fly = True`.

### Disruption Handler
Blocks a selected cell and compares the route before and after the no-fly event.

### ML Pipeline
Includes:

- Demand forecasting using Linear Regression or Random Forest Regressor
- Anomaly classification using Decision Tree or Random Forest Classifier
- Accuracy, MAE, RMSE, and confusion matrix outputs

### 20-Step Simulation
Runs the final integrated scenario:

1. Validate layout
2. Select fleet
3. Initialize drones
4. Generate deliveries
5-6. Assign deliveries
7-10. Move drones
11. Activate no-fly cell
12-14. Reroute and continue
15-17. Forecast demand and add delivery
18. Detect anomaly
19. Force return to hub
20. Print final summary


---

## 5A. Notebooks

The `notebooks/` folder now contains two interactive Jupyter notebooks:

- `demand_forecasting.ipynb` — regression model, feature explorer, actual-vs-predicted plot, and manual demand prediction sliders.
- `anomaly_classifier.ipynb` — classification model, feature scatter explorer, confusion matrix, and manual anomaly prediction sliders.

Open them with:

```bash
jupyter notebook
```

## 5. Smoke test

To verify core modules:

```bash
python smoke_test.py
```

Expected output:

```text
All smoke tests passed.
```

---

## 6. Viva explanation

The honest explanation:

> This is not a real drone deployment platform. It is an academic simulation that integrates multiple AI concepts in a clean, visual, and explainable way.

Strong viva points:

- CSP rules are transparent and easy to inspect.
- A* uses Manhattan distance, which is admissible for 4-direction grid movement.
- Commercial cells have lower travel cost, making path cost more realistic.
- Rerouting is triggered only when a new no-fly cell intersects a drone's future path.
- ML models are intentionally simple and explainable.
- Synthetic anomaly labels are acceptable because the project brief allows simple synthetic logic for a 2-week project.

---

## 7. Limitations

- The grid is a simplified 10x10 model.
- Weather, battery, payload, and airspace effects are simplified.
- The ML datasets are synthetic by default to keep the project runnable offline.
- No real drone hardware control is included.

---

## 8. Best demo flow

1. Open **Dashboard** and explain the 10x10 grid.
2. Open **CSP Validator** and show all rules passing.
3. Load the invalid grid from the sidebar and show failed rules.
4. Open **Fleet Selector** and adjust budget.
5. Open **A\* Planner** and show route changes after marking cells as no-fly.
6. Open **Disruption Handler** and compare before/after rerouting.
7. Open **ML Pipeline** and show MAE/RMSE and confusion matrix.
8. Open **20-Step Simulation** and run all steps.

