"""Console entry point for AeroNet Lite.

Use this if Streamlit is not available. The full interactive version is src/app.py.
"""

from __future__ import annotations

from delivery_simulator import run_all_steps
from grid_model import create_sample_grid
from layout_validator import validate_layout, validation_report_text


def main() -> None:
    grid = create_sample_grid(valid=True)
    print("=== AeroNet Lite Console Demo ===")
    print(validation_report_text(validate_layout(grid)))
    print("\n=== 20-Step Simulation Event Log ===")
    state = run_all_steps()
    for event in state.event_log:
        print(event)


if __name__ == "__main__":
    main()
