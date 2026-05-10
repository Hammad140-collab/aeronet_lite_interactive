"""Interactive Streamlit dashboard for AeroNet Lite."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Make both `streamlit run src/app.py` and `python src/app.py` imports work.
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from astar_planner import astar, delivery_route
from delivery_simulator import deliveries_as_rows, drones_as_rows, initial_state, run_all_steps, run_next_step
from fleet_selector import brute_force_fleet, genetic_fleet
from grid_model import (
    ZONES,
    clone_grid,
    coordinates_as_options,
    create_sample_grid,
    find_cells,
    grid_to_table,
    parse_coord,
    set_flag,
    set_zone,
)
from layout_validator import layout_is_valid, validate_layout, validation_report_text
from ml_pipeline import predict_anomaly, predict_demand, train_anomaly_model, train_demand_model
from visualization import (
    confusion_matrix_figure,
    demand_heatmap_figure,
    fleet_score_figure,
    ga_history_figure,
    zone_map_figure,
)

st.set_page_config(page_title="AeroNet Lite", page_icon="🚁", layout="wide")

st.markdown(
    """
    <style>
    .main .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
    .metric-card {border: 1px solid #e7e7e7; border-radius: 16px; padding: 14px; background: #fafafa;}
    .small-note {font-size: 0.9rem; color: #555;}
    </style>
    """,
    unsafe_allow_html=True,
)


def ensure_grid() -> None:
    if "grid" not in st.session_state:
        st.session_state.grid = create_sample_grid(valid=True)
    if "last_route" not in st.session_state:
        st.session_state.last_route = []


def reset_grid(valid: bool = True) -> None:
    st.session_state.grid = create_sample_grid(valid=valid)
    st.session_state.last_route = []


def sidebar_controls() -> None:
    st.sidebar.title("AeroNet Lite Controls")
    st.sidebar.caption("Use this panel to edit the city and create demo situations.")
    if st.sidebar.button("Reset to valid demo grid", use_container_width=True):
        reset_grid(valid=True)
    if st.sidebar.button("Load intentionally invalid grid", use_container_width=True):
        reset_grid(valid=False)

    st.sidebar.divider()
    st.sidebar.subheader("Edit one cell")
    coord_text = st.sidebar.selectbox("Cell coordinate", coordinates_as_options(), index=22)
    coord = parse_coord(coord_text)
    zone = st.sidebar.selectbox("Zone", ZONES, index=ZONES.index(st.session_state.grid[coord[0]][coord[1]]["zone"]))
    no_fly = st.sidebar.checkbox("No-fly", bool(st.session_state.grid[coord[0]][coord[1]]["no_fly"]))
    hub = st.sidebar.checkbox("Drone hub", bool(st.session_state.grid[coord[0]][coord[1]]["is_hub"]))
    charging = st.sidebar.checkbox("Charging pad", bool(st.session_state.grid[coord[0]][coord[1]]["is_charging"]))
    pickup = st.sidebar.checkbox("Medical pickup", bool(st.session_state.grid[coord[0]][coord[1]]["is_medical_pickup"]))
    if st.sidebar.button("Apply cell edit", use_container_width=True):
        set_zone(st.session_state.grid, coord, zone)
        set_flag(st.session_state.grid, coord, "no_fly", no_fly)
        set_flag(st.session_state.grid, coord, "is_hub", hub)
        set_flag(st.session_state.grid, coord, "is_charging", charging)
        set_flag(st.session_state.grid, coord, "is_medical_pickup", pickup)
        st.sidebar.success(f"Updated cell {coord}.")


def dashboard_tab() -> None:
    st.subheader("City dashboard")
    c1, c2, c3, c4 = st.columns(4)
    hubs = find_cells(st.session_state.grid, flag="is_hub")
    no_fly = find_cells(st.session_state.grid, flag="no_fly")
    charging = find_cells(st.session_state.grid, flag="is_charging")
    total_demand = sum(float(row["demand"]) for row in grid_to_table(st.session_state.grid))
    c1.metric("Grid size", "10 x 10")
    c2.metric("Hubs", len(hubs))
    c3.metric("No-fly cells", len(no_fly))
    c4.metric("Total demand", f"{total_demand:.1f}")

    left, right = st.columns([1.15, 0.85])
    with left:
        st.plotly_chart(zone_map_figure(st.session_state.grid, st.session_state.last_route), use_container_width=True)
    with right:
        st.plotly_chart(demand_heatmap_figure(st.session_state.grid), use_container_width=True)

    with st.expander("Show full grid table"):
        st.dataframe(pd.DataFrame(grid_to_table(st.session_state.grid)), use_container_width=True, height=330)


def validator_tab() -> None:
    st.subheader("Module 1 — CSP Layout Validator")
    results = validate_layout(st.session_state.grid)
    valid = layout_is_valid(results)
    st.metric("Layout validity", "PASSED" if valid else "FAILED")

    cols = st.columns(4)
    for col, result in zip(cols, results):
        col.metric(f"{result.rule_id}", "Pass" if result.passed else "Fail")
        col.caption(result.title)

    for result in results:
        with st.expander(f"{result.rule_id}: {result.title} — {'Passed' if result.passed else 'Failed'}", expanded=not result.passed):
            if result.messages:
                for msg in result.messages:
                    st.write("-", msg)
            if result.suggestions:
                st.warning("Suggested fixes:")
                for suggestion in result.suggestions:
                    st.write("-", suggestion)
            if not result.messages and not result.suggestions:
                st.success("No issues found.")

    st.code(validation_report_text(results), language="text")


def fleet_tab() -> None:
    st.subheader("Module 2 — Fleet Selector")
    left, right = st.columns([0.35, 0.65])
    with left:
        budget = st.slider("Budget", min_value=2000, max_value=15000, value=7000, step=500)
        demand_units = st.slider("Estimated demand units", min_value=50, max_value=800, value=280, step=10)
        method = st.radio("Selection method", ["Brute force", "Genetic Algorithm"], horizontal=False)
    with right:
        if method == "Brute force":
            best, options = brute_force_fleet(budget, demand_units)
            st.success(f"Best fleet: {best.light_count} light + {best.heavy_count} heavy drones")
            st.dataframe(pd.DataFrame([opt.as_dict() for opt in options[:15]]), use_container_width=True)
            st.plotly_chart(fleet_score_figure(options), use_container_width=True)
        else:
            best, history = genetic_fleet(budget, demand_units)
            st.success(f"GA best fleet: {best.light_count} light + {best.heavy_count} heavy drones")
            st.json(best.as_dict())
            st.plotly_chart(ga_history_figure(history), use_container_width=True)

    st.info(
        "Fitness = 0.75 × demand coverage percentage − 0.25 × budget used percentage. "
        "This gives a simple defendable optimization objective."
    )


def planner_tab() -> None:
    st.subheader("Module 3 — A* Delivery Path Planner")
    col1, col2, col3, col4 = st.columns(4)
    options = coordinates_as_options()
    with col1:
        start_text = st.selectbox("Hub/start", options, index=22)
    with col2:
        pickup_text = st.selectbox("Pickup", options, index=19)
    with col3:
        drop_text = st.selectbox("Drop-off", options, index=79)
    with col4:
        return_to_hub = st.checkbox("Return to hub", value=True)

    start = parse_coord(start_text)
    pickup = parse_coord(pickup_text)
    dropoff = parse_coord(drop_text)

    mode = st.radio("Plan mode", ["Full delivery route", "Single A* segment"], horizontal=True)
    if mode == "Single A* segment":
        result = astar(start, dropoff, st.session_state.grid)
    else:
        result = delivery_route(start, pickup, dropoff, st.session_state.grid, return_to_hub=return_to_hub)

    st.session_state.last_route = result.path
    c1, c2, c3 = st.columns(3)
    c1.metric("Route cells", len(result.path))
    c2.metric("Route cost", "∞" if result.cost == float("inf") else result.cost)
    c3.metric("Explored nodes", result.explored_count)
    if result.path:
        st.success(result.message)
    else:
        st.error(result.message)
    st.plotly_chart(zone_map_figure(st.session_state.grid, result.path, title="A* Route Map"), use_container_width=True)
    with st.expander("Show route coordinates"):
        st.write(result.path)


def disruption_tab() -> None:
    st.subheader("Module 4 — Real-Time Disruption and Rerouting")
    st.write("Create a route, activate a no-fly cell, and rerun A* from the current position.")
    col1, col2, col3 = st.columns(3)
    options = coordinates_as_options()
    start = parse_coord(col1.selectbox("Current drone position", options, index=22, key="dis_start"))
    goal = parse_coord(col2.selectbox("Next target", options, index=79, key="dis_goal"))
    blocked = parse_coord(col3.selectbox("Cell to block", options, index=55, key="dis_block"))

    before = astar(start, goal, st.session_state.grid)
    temp_grid = clone_grid(st.session_state.grid)
    temp_grid[blocked[0]][blocked[1]]["no_fly"] = True
    after = astar(start, goal, temp_grid)

    c1, c2 = st.columns(2)
    with c1:
        st.caption("Before disruption")
        st.metric("Cost", before.cost if before.path else "No route")
        st.plotly_chart(zone_map_figure(st.session_state.grid, before.path, title="Before no-fly event"), use_container_width=True)
    with c2:
        st.caption(f"After blocking {blocked}")
        st.metric("New cost", after.cost if after.path else "No route")
        st.plotly_chart(zone_map_figure(temp_grid, after.path, title="After rerouting"), use_container_width=True)

    if st.button("Apply this no-fly cell to main grid"):
        st.session_state.grid[blocked[0]][blocked[1]]["no_fly"] = True
        st.session_state.last_route = after.path
        st.success(f"Blocked {blocked} in the main grid.")


def ml_tab() -> None:
    st.subheader("Module 5 — ML Pipeline")
    demand_tab, anomaly_tab = st.tabs(["Demand Forecasting", "Anomaly Classification"])

    with demand_tab:
        model_name = st.radio("Regression model", ["Random Forest", "Linear Regression"], horizontal=True)
        result = train_demand_model(model_name=model_name)
        m1, m2 = st.columns(2)
        m1.metric("MAE", result.mae)
        m2.metric("RMSE", result.rmse)
        st.dataframe(result.test_frame.head(20), use_container_width=True)

        st.markdown("#### Try your own demand prediction")
        c1, c2, c3, c4, c5 = st.columns(5)
        hour = c1.slider("Hour", 0, 23, 18)
        weekday = c2.slider("Weekday", 0, 6, 2)
        temp = c3.slider("Temperature", 5.0, 45.0, 31.0)
        weather = c4.selectbox("Weather", [0, 1, 2], format_func=lambda x: {0: "Clear", 1: "Cloudy", 2: "Rainy"}[x])
        density = c5.selectbox("Density level", [1, 2, 3], format_func=lambda x: {1: "Low", 2: "Medium", 3: "High"}[x])
        pred = predict_demand(result.model, hour, weekday, temp, weather, density)
        st.success(f"Predicted demand = {pred:.2f} units")

    with anomaly_tab:
        clf_name = st.radio("Classifier", ["Random Forest", "Decision Tree"], horizontal=True)
        result = train_anomaly_model(model_name=clf_name)
        st.metric("Accuracy", result.accuracy)
        st.plotly_chart(confusion_matrix_figure(result.confusion, result.labels), use_container_width=True)
        st.dataframe(result.sample_frame.head(20), use_container_width=True)

        st.markdown("#### Try your own anomaly classification")
        c1, c2, c3, c4 = st.columns(4)
        battery_drop = c1.slider("Battery drop", 0.0, 25.0, 4.0)
        speed = c2.slider("Speed", 0.0, 25.0, 8.0)
        route_deviation = c3.slider("Route deviation", 0.0, 8.0, 0.5)
        altitude_change = c4.slider("Altitude change", 0.0, 15.0, 1.0)
        label = predict_anomaly(result.model, battery_drop, speed, route_deviation, altitude_change)
        st.warning(f"Predicted status: {label}")


def simulation_tab() -> None:
    st.subheader("Integrated 20-Step Simulation")
    col1, col2, col3 = st.columns([0.25, 0.25, 0.5])
    with col1:
        budget = st.number_input("Budget", min_value=2000, max_value=15000, value=7000, step=500, key="sim_budget")
    with col2:
        demand_units = st.number_input("Demand units", min_value=50, max_value=800, value=280, step=10, key="sim_demand")
    with col3:
        st.caption("Run one step at a time for viva explanation, or run all steps for the final demo.")

    if "sim_state" not in st.session_state:
        st.session_state.sim_state = initial_state(budget, demand_units)

    b1, b2, b3 = st.columns(3)
    if b1.button("Reset simulation", use_container_width=True):
        st.session_state.sim_state = initial_state(budget, demand_units)
    if b2.button("Run next step", use_container_width=True):
        st.session_state.sim_state = run_next_step(st.session_state.sim_state)
    if b3.button("Run all 20 steps", use_container_width=True):
        st.session_state.sim_state = run_all_steps(budget, demand_units)

    state = st.session_state.sim_state
    st.metric("Current step", state.step)
    st.plotly_chart(zone_map_figure(state.grid, state.last_route, title="Simulation Map"), use_container_width=True)

    d1, d2 = st.columns(2)
    with d1:
        st.markdown("#### Drones")
        st.dataframe(pd.DataFrame(drones_as_rows(state.drones)), use_container_width=True, height=260)
    with d2:
        st.markdown("#### Deliveries")
        st.dataframe(pd.DataFrame(deliveries_as_rows(state.deliveries)), use_container_width=True, height=260)

    st.markdown("#### Event Log")
    st.code("\n".join(state.event_log[-30:]) if state.event_log else "No events yet. Click Run next step.", language="text")



ensure_grid()
sidebar_controls()

st.title("AeroNet Lite — Interactive Drone Delivery AI Simulator")
st.caption("CSP validation • Fleet planning • A* routing • Replanning • Demand forecasting • Anomaly detection")

tabs = st.tabs(
    [
        "Dashboard",
        "CSP Validator",
        "Fleet Selector",
        "A* Planner",
        "Disruption Handler",
        "ML Pipeline",
        "20-Step Simulation",
    ]
)

with tabs[0]:
    dashboard_tab()
with tabs[1]:
    validator_tab()
with tabs[2]:
    fleet_tab()
with tabs[3]:
    planner_tab()
with tabs[4]:
    disruption_tab()
with tabs[5]:
    ml_tab()
with tabs[6]:
    simulation_tab()
