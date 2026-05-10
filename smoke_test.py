"""Tiny smoke test to verify core modules before demo."""

from src.astar_planner import delivery_route
from src.delivery_simulator import run_all_steps
from src.fleet_selector import brute_force_fleet, genetic_fleet
from src.grid_model import create_sample_grid
from src.layout_validator import layout_is_valid, validate_layout
from src.ml_pipeline import train_anomaly_model, train_demand_model


def test_all():
    grid = create_sample_grid(valid=True)
    assert layout_is_valid(validate_layout(grid))

    best, options = brute_force_fleet(7000, 280)
    assert best.cost <= 7000 and options

    ga_best, history = genetic_fleet(7000, 280, generations=5)
    assert ga_best.cost <= 7000 and history

    route = delivery_route((2, 2), (1, 9), (0, 2), grid)
    assert route.path, route.message

    demand = train_demand_model()
    assert demand.mae > 0

    anomaly = train_anomaly_model()
    assert anomaly.accuracy > 0.7

    sim = run_all_steps()
    assert sim.step == 20
    assert sim.event_log
    print("All smoke tests passed.")


if __name__ == "__main__":
    test_all()
