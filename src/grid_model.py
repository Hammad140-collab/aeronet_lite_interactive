"""Shared 10x10 grid model for AeroNet Lite.

Beginner-friendly on purpose: each cell is a small dictionary so students can
inspect and modify the city without learning advanced Python first.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, Iterable, List, Tuple

Coord = Tuple[int, int]
Cell = Dict[str, object]
Grid = List[List[Cell]]

GRID_SIZE = 10

ZONES = [
    "Residential",
    "Commercial",
    "Hospital",
    "School",
    "Industrial",
    "Open Field",
]

ZONE_SYMBOLS = {
    "Residential": "R",
    "Commercial": "C",
    "Hospital": "H",
    "School": "S",
    "Industrial": "I",
    "Open Field": "O",
}

ZONE_COLORS = {
    "Residential": "#90EE90",
    "Commercial": "#87CEFA",
    "Hospital": "#FFB6C1",
    "School": "#FFD966",
    "Industrial": "#B7B7B7",
    "Open Field": "#F4F4F4",
}


def make_cell(row: int, col: int, zone: str = "Open Field") -> Cell:
    density_by_zone = {
        "Residential": 6500,
        "Commercial": 4200,
        "Hospital": 3500,
        "School": 3000,
        "Industrial": 1800,
        "Open Field": 500,
    }
    return {
        "row": row,
        "col": col,
        "zone": zone,
        "density": density_by_zone.get(zone, 500),
        "is_hub": False,
        "is_charging": False,
        "is_medical_pickup": False,
        "no_fly": False,
        "demand": 0.0,
    }


def empty_grid() -> Grid:
    return [[make_cell(r, c) for c in range(GRID_SIZE)] for r in range(GRID_SIZE)]


def in_bounds(row: int, col: int) -> bool:
    return 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE


def set_zone(grid: Grid, coord: Coord, zone: str) -> None:
    r, c = coord
    grid[r][c]["zone"] = zone
    density = {
        "Residential": 6500,
        "Commercial": 4200,
        "Hospital": 3500,
        "School": 3000,
        "Industrial": 1800,
        "Open Field": 500,
    }[zone]
    grid[r][c]["density"] = density


def set_flag(grid: Grid, coord: Coord, flag: str, value: bool = True) -> None:
    r, c = coord
    grid[r][c][flag] = value


def get_neighbors(row: int, col: int) -> List[Coord]:
    candidates = [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]
    return [(r, c) for r, c in candidates if in_bounds(r, c)]


def manhattan(a: Coord, b: Coord) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def find_cells(grid: Grid, *, zone: str | None = None, flag: str | None = None) -> List[Coord]:
    found: List[Coord] = []
    for row in grid:
        for cell in row:
            ok = True
            if zone is not None:
                ok = ok and cell["zone"] == zone
            if flag is not None:
                ok = ok and bool(cell.get(flag, False))
            if ok:
                found.append((int(cell["row"]), int(cell["col"])))
    return found


def iter_cells(grid: Grid) -> Iterable[Cell]:
    for row in grid:
        for cell in row:
            yield cell


def clone_grid(grid: Grid) -> Grid:
    return deepcopy(grid)


def create_sample_grid(valid: bool = True) -> Grid:
    """Create a defendable sample city.

    When valid=False, a few deliberate mistakes are injected so the CSP validator
    can show meaningful feedback during demo/viva.
    """
    grid = empty_grid()

    # Residential clusters around both hubs so R2 can pass.
    hubs = [(2, 2), (7, 7)]
    for hub in hubs:
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if manhattan((r, c), hub) <= 3:
                    set_zone(grid, (r, c), "Residential")

    # Commercial corridors lower travel cost in A*.
    for c in range(GRID_SIZE):
        set_zone(grid, (5, c), "Commercial")
    for r in range(GRID_SIZE):
        set_zone(grid, (r, 5), "Commercial")

    # Special facilities.
    set_zone(grid, (1, 8), "Hospital")
    set_zone(grid, (1, 9), "Commercial")
    set_flag(grid, (1, 9), "is_medical_pickup", True)

    set_zone(grid, (8, 1), "School")
    set_zone(grid, (8, 8), "Industrial")
    set_zone(grid, (8, 9), "Industrial")
    set_zone(grid, (9, 8), "Industrial")
    set_zone(grid, (9, 9), "Industrial")

    # Hubs and charging pads.
    for coord in hubs:
        set_zone(grid, coord, "Commercial")
        set_flag(grid, coord, "is_hub", True)
    for coord in [(2, 3), (7, 6)]:
        set_zone(grid, coord, "Commercial")
        set_flag(grid, coord, "is_charging", True)

    # No-fly cells for path planning demos, placed away from direct starts.
    for coord in [(0, 4), (4, 4), (6, 2)]:
        set_flag(grid, coord, "no_fly", True)

    # Demand = density influence + zone effect.
    for cell in iter_cells(grid):
        zone_boost = {
            "Residential": 20,
            "Commercial": 12,
            "Hospital": 18,
            "School": 10,
            "Industrial": 5,
            "Open Field": 1,
        }[str(cell["zone"])]
        cell["demand"] = round(float(cell["density"]) / 1000 + zone_boost, 2)

    if not valid:
        # R1 failure: industrial directly beside school.
        set_zone(grid, (8, 2), "Industrial")
        # R2 failure: isolated residential too far from all hubs.
        set_zone(grid, (0, 9), "Residential")
        # R3 failure: hub far from charging.
        set_flag(grid, (9, 0), "is_hub", True)
        set_zone(grid, (9, 0), "Commercial")
        # R4 failure option: remove medical pickup near hospital.
        set_flag(grid, (1, 9), "is_medical_pickup", False)

    return grid


def grid_to_table(grid: Grid) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for cell in iter_cells(grid):
        rows.append(
            {
                "row": cell["row"],
                "col": cell["col"],
                "zone": cell["zone"],
                "density": cell["density"],
                "demand": cell["demand"],
                "hub": cell["is_hub"],
                "charging": cell["is_charging"],
                "medical_pickup": cell["is_medical_pickup"],
                "no_fly": cell["no_fly"],
            }
        )
    return rows


def coordinates_as_options() -> List[str]:
    return [f"({r}, {c})" for r in range(GRID_SIZE) for c in range(GRID_SIZE)]


def parse_coord(text: str) -> Coord:
    cleaned = text.strip().replace("(", "").replace(")", "")
    r_s, c_s = cleaned.split(",")
    return int(r_s), int(c_s)
