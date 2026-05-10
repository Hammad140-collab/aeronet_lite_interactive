"""A* delivery path planner."""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

try:
    from .grid_model import Coord, Grid, get_neighbors, manhattan
except ImportError:
    from grid_model import Coord, Grid, get_neighbors, manhattan


@dataclass
class PathResult:
    path: List[Coord]
    cost: float
    message: str
    explored_count: int


def movement_cost(grid: Grid, coord: Coord) -> float:
    r, c = coord
    if grid[r][c]["zone"] == "Commercial":
        return 0.8
    return 1.0


def reconstruct(came_from: Dict[Coord, Coord], current: Coord) -> List[Coord]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def astar(start: Coord, goal: Coord, grid: Grid) -> PathResult:
    if grid[start[0]][start[1]]["no_fly"]:
        return PathResult([], float("inf"), f"Start {start} is blocked by no-fly status.", 0)
    if grid[goal[0]][goal[1]]["no_fly"]:
        return PathResult([], float("inf"), f"Goal {goal} is blocked by no-fly status.", 0)

    open_heap: List[Tuple[float, int, Coord]] = []
    heapq.heappush(open_heap, (manhattan(start, goal), 0, start))
    came_from: Dict[Coord, Coord] = {}
    g_score: Dict[Coord, float] = {start: 0.0}
    explored = 0
    counter = 0

    while open_heap:
        _, _, current = heapq.heappop(open_heap)
        explored += 1
        if current == goal:
            path = reconstruct(came_from, current)
            return PathResult(path, round(g_score[current], 2), "Path found using A*.", explored)

        for neighbor in get_neighbors(*current):
            nr, nc = neighbor
            if grid[nr][nc]["no_fly"]:
                continue
            tentative_g = g_score[current] + movement_cost(grid, neighbor)
            if tentative_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                counter += 1
                f_score = tentative_g + manhattan(neighbor, goal)
                heapq.heappush(open_heap, (f_score, counter, neighbor))

    return PathResult([], float("inf"), f"No safe path exists from {start} to {goal}.", explored)


def combine_segments(segments: List[PathResult]) -> PathResult:
    full_path: List[Coord] = []
    total_cost = 0.0
    explored = 0
    for seg in segments:
        explored += seg.explored_count
        if not seg.path:
            return PathResult([], float("inf"), seg.message, explored)
        total_cost += seg.cost
        if full_path:
            full_path.extend(seg.path[1:])
        else:
            full_path.extend(seg.path)
    return PathResult(full_path, round(total_cost, 2), "Complete multi-stop route found.", explored)


def delivery_route(hub: Coord, pickup: Coord, dropoff: Coord, grid: Grid, return_to_hub: bool = True) -> PathResult:
    points = [hub, pickup, dropoff]
    if return_to_hub:
        points.append(hub)
    segments: List[PathResult] = []
    for a, b in zip(points[:-1], points[1:]):
        segments.append(astar(a, b, grid))
    return combine_segments(segments)


def route_crosses(route: List[Coord], blocked: Coord) -> bool:
    return blocked in route


def next_target_from_route(route: List[Coord], current_position: Coord, final_goal: Optional[Coord] = None) -> Coord:
    if final_goal is not None:
        return final_goal
    if not route:
        return current_position
    if current_position in route:
        idx = route.index(current_position)
        return route[-1] if idx < len(route) - 1 else current_position
    return route[-1]
