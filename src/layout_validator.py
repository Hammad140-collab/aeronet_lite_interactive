"""CSP-style layout validator for AeroNet Lite."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

try:
    from .grid_model import Grid, find_cells, get_neighbors, manhattan
except ImportError:  # Allows running files directly during beginner debugging.
    from grid_model import Grid, find_cells, get_neighbors, manhattan


@dataclass
class RuleResult:
    rule_id: str
    title: str
    passed: bool
    messages: List[str]
    suggestions: List[str]


def check_industrial_safety(grid: Grid) -> RuleResult:
    messages: List[str] = []
    suggestions: List[str] = []
    for r, c in find_cells(grid, zone="Industrial"):
        for nr, nc in get_neighbors(r, c):
            neighbor_zone = str(grid[nr][nc]["zone"])
            if neighbor_zone in {"School", "Hospital"}:
                messages.append(
                    f"Industrial cell {(r, c)} is directly adjacent to {neighbor_zone} at {(nr, nc)}."
                )
                suggestions.append(
                    f"Move industrial cell {(r, c)} away or insert Open Field buffer near {(nr, nc)}."
                )
    return RuleResult("R1", "Industrial safety buffer", not messages, messages, suggestions)


def check_residential_coverage(grid: Grid) -> RuleResult:
    messages: List[str] = []
    suggestions: List[str] = []
    hubs = find_cells(grid, flag="is_hub")
    if not hubs:
        return RuleResult(
            "R2",
            "Residential hub coverage",
            False,
            ["No drone hub exists in the layout."],
            ["Add at least one hub near residential demand clusters."],
        )

    for coord in find_cells(grid, zone="Residential"):
        nearest = min(manhattan(coord, hub) for hub in hubs)
        if nearest > 3:
            messages.append(f"Residential cell {coord} is {nearest} cells away from the nearest hub.")
            suggestions.append(f"Add a hub within 3 Manhattan cells of {coord}, or convert it to Open Field.")
    return RuleResult("R2", "Residential hub coverage", not messages, messages, suggestions)


def check_hub_charging(grid: Grid) -> RuleResult:
    messages: List[str] = []
    suggestions: List[str] = []
    hubs = find_cells(grid, flag="is_hub")
    charging_pads = find_cells(grid, flag="is_charging")
    if not charging_pads:
        return RuleResult(
            "R3",
            "Hub charging access",
            False,
            ["No charging pad exists in the layout."],
            ["Place one charging pad within 2 cells of every hub."],
        )

    for hub in hubs:
        nearest = min(manhattan(hub, pad) for pad in charging_pads)
        if nearest > 2:
            messages.append(f"Hub {hub} has nearest charging pad {nearest} cells away.")
            suggestions.append(f"Add a charging pad within 2 Manhattan cells of hub {hub}.")
    return RuleResult("R3", "Hub charging access", not messages, messages, suggestions)


def check_medical_access(grid: Grid) -> RuleResult:
    hospitals = find_cells(grid, zone="Hospital")
    pickups = find_cells(grid, flag="is_medical_pickup")
    for hospital in hospitals:
        if any(manhattan(hospital, pickup) <= 1 for pickup in pickups):
            return RuleResult(
                "R4",
                "Medical pickup access",
                True,
                [f"Hospital {hospital} has a medical pickup point within 1 cell."],
                [],
            )
    return RuleResult(
        "R4",
        "Medical pickup access",
        False,
        ["No hospital has a medical pickup point within 1 Manhattan cell."],
        ["Place a medical pickup point beside at least one hospital."],
    )


def validate_layout(grid: Grid) -> List[RuleResult]:
    return [
        check_industrial_safety(grid),
        check_residential_coverage(grid),
        check_hub_charging(grid),
        check_medical_access(grid),
    ]


def layout_is_valid(results: List[RuleResult]) -> bool:
    return all(result.passed for result in results)


def validation_report_text(results: List[RuleResult]) -> str:
    lines = []
    lines.append(f"Layout validity = {layout_is_valid(results)}")
    for result in results:
        status = "PASSED" if result.passed else "FAILED"
        lines.append(f"{result.rule_id} {status}: {result.title}")
        for msg in result.messages:
            lines.append(f"  - {msg}")
        for suggestion in result.suggestions[:3]:
            lines.append(f"  Suggested fix: {suggestion}")
    return "\n".join(lines)
