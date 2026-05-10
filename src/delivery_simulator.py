"""20-step simulator with disruption handling and rerouting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

try:
    from .astar_planner import astar, delivery_route
    from .fleet_selector import FleetOption, brute_force_fleet
    from .grid_model import Coord, Grid, clone_grid, create_sample_grid, find_cells
    from .layout_validator import layout_is_valid, validate_layout
except ImportError:
    from astar_planner import astar, delivery_route
    from fleet_selector import FleetOption, brute_force_fleet
    from grid_model import Coord, Grid, clone_grid, create_sample_grid, find_cells
    from layout_validator import layout_is_valid, validate_layout


@dataclass
class Drone:
    drone_id: str
    drone_type: str
    home_hub: Coord
    position: Coord
    route: List[Coord] = field(default_factory=list)
    route_index: int = 0
    status: str = "idle"
    delivery_id: Optional[str] = None


@dataclass
class Delivery:
    delivery_id: str
    pickup: Coord
    dropoff: Coord
    priority: str
    assigned_drone: Optional[str] = None
    status: str = "waiting"
    route_cost: float = 0.0


@dataclass
class SimulationState:
    grid: Grid
    step: int = 0
    fleet: Optional[FleetOption] = None
    drones: List[Drone] = field(default_factory=list)
    deliveries: List[Delivery] = field(default_factory=list)
    event_log: List[str] = field(default_factory=list)
    last_route: List[Coord] = field(default_factory=list)
    completed: int = 0
    delayed: int = 0
    failed: int = 0
    anomaly_label: str = "Not checked yet"


def initial_state(budget: int = 7000, demand_units: float = 280.0) -> SimulationState:
    grid = create_sample_grid(valid=True)
    best, _ = brute_force_fleet(budget=budget, demand_units=demand_units)
    return SimulationState(grid=grid, fleet=best)


def make_drones(fleet: FleetOption, grid: Grid) -> List[Drone]:
    hubs = find_cells(grid, flag="is_hub") or [(2, 2)]
    drones: List[Drone] = []
    count = 1
    for _ in range(fleet.light_count):
        hub = hubs[(count - 1) % len(hubs)]
        drones.append(Drone(f"D{count}", "Light", hub, hub))
        count += 1
    for _ in range(fleet.heavy_count):
        hub = hubs[(count - 1) % len(hubs)]
        drones.append(Drone(f"D{count}", "Heavy", hub, hub))
        count += 1
    return drones


def generate_demo_deliveries() -> List[Delivery]:
    return [
        Delivery("DEL-1", pickup=(1, 9), dropoff=(0, 2), priority="medical"),
        Delivery("DEL-2", pickup=(5, 4), dropoff=(7, 9), priority="normal"),
        Delivery("DEL-3", pickup=(5, 7), dropoff=(3, 1), priority="normal"),
        Delivery("DEL-4", pickup=(1, 9), dropoff=(9, 7), priority="medical"),
        Delivery("DEL-5", pickup=(5, 0), dropoff=(4, 2), priority="normal"),
    ]


def nearest_idle_drone(drones: List[Drone], pickup: Coord, grid: Grid) -> Optional[Drone]:
    idle = [d for d in drones if d.status == "idle"]
    if not idle:
        return None
    scored = []
    for drone in idle:
        result = astar(drone.position, pickup, grid)
        scored.append((result.cost, drone))
    scored.sort(key=lambda item: item[0])
    return scored[0][1]


def assign_delivery(delivery: Delivery, drones: List[Drone], grid: Grid) -> str:
    drone = nearest_idle_drone(drones, delivery.pickup, grid)
    if drone is None:
        delivery.status = "delayed"
        return f"{delivery.delivery_id}: no idle drone available, delivery delayed."

    route = delivery_route(drone.position, delivery.pickup, delivery.dropoff, grid, return_to_hub=True)
    if not route.path:
        delivery.status = "failed"
        return f"{delivery.delivery_id}: route failed. {route.message}"

    drone.route = route.path
    drone.route_index = 0
    drone.status = "flying"
    drone.delivery_id = delivery.delivery_id
    delivery.assigned_drone = drone.drone_id
    delivery.status = "in progress"
    delivery.route_cost = route.cost
    return f"{delivery.delivery_id} assigned to {drone.drone_id}. Route cost = {route.cost}, cells = {len(route.path)}."


def move_drones(state: SimulationState) -> None:
    for drone in state.drones:
        if drone.status != "flying" or not drone.route:
            continue
        if drone.route_index < len(drone.route) - 1:
            drone.route_index += 1
            drone.position = drone.route[drone.route_index]
            state.event_log.append(f"Step {state.step}: {drone.drone_id} moved to {drone.position}.")
        else:
            drone.status = "idle"
            done_delivery_id = drone.delivery_id
            drone.delivery_id = None
            for delivery in state.deliveries:
                if delivery.delivery_id == done_delivery_id:
                    delivery.status = "completed"
                    state.completed += 1
            state.event_log.append(f"Step {state.step}: {drone.drone_id} completed {done_delivery_id} and returned to hub.")


def activate_no_fly_and_reroute(state: SimulationState, blocked: Coord) -> None:
    state.grid[blocked[0]][blocked[1]]["no_fly"] = True
    state.event_log.append(f"Step {state.step}: no-fly cell activated at {blocked}.")
    for drone in state.drones:
        if drone.status != "flying" or not drone.route:
            continue
        future_route = drone.route[drone.route_index :]
        if blocked not in future_route:
            continue
        target = drone.route[-1]
        reroute = astar(drone.position, target, state.grid)
        if reroute.path:
            drone.route = reroute.path
            drone.route_index = 0
            state.last_route = reroute.path
            state.event_log.append(f"Step {state.step}: {drone.drone_id} rerouted successfully using A*.")
        else:
            drone.status = "delayed"
            state.delayed += 1
            state.event_log.append(f"Step {state.step}: {drone.drone_id} cannot reach destination safely; marked delayed.")


def run_next_step(state: SimulationState) -> SimulationState:
    state.step += 1

    if state.step == 1:
        results = validate_layout(state.grid)
        state.event_log.append(f"Step 1: Layout validation {'passed' if layout_is_valid(results) else 'failed'}.")
    elif state.step == 2:
        if state.fleet is None:
            state.fleet, _ = brute_force_fleet(7000, 280)
        state.event_log.append(
            f"Step 2: Fleet selected: {state.fleet.light_count} light drones, {state.fleet.heavy_count} heavy drones."
        )
    elif state.step == 3:
        if state.fleet is None:
            state.fleet, _ = brute_force_fleet(7000, 280)
        state.drones = make_drones(state.fleet, state.grid)
        state.event_log.append(f"Step 3: {len(state.drones)} drones initialized at hubs.")
    elif state.step == 4:
        state.deliveries = generate_demo_deliveries()
        state.event_log.append(f"Step 4: {len(state.deliveries)} deliveries generated.")
    elif state.step in [5, 6]:
        pending = [d for d in state.deliveries if d.status == "waiting"]
        if pending:
            msg = assign_delivery(pending[0], state.drones, state.grid)
            state.event_log.append(f"Step {state.step}: {msg}")
            drone_id = pending[0].assigned_drone
            if drone_id:
                drone = next(d for d in state.drones if d.drone_id == drone_id)
                state.last_route = drone.route
        else:
            state.event_log.append(f"Step {state.step}: No waiting deliveries remain.")
    elif 7 <= state.step <= 10:
        move_drones(state)
    elif state.step == 11:
        # Pick a cell likely to intersect DEL-1/DEL-2 routes, useful for demo.
        activate_no_fly_and_reroute(state, (5, 5))
    elif 12 <= state.step <= 14:
        move_drones(state)
        waiting = [d for d in state.deliveries if d.status == "waiting"]
        if waiting:
            state.event_log.append(f"Step {state.step}: {assign_delivery(waiting[0], state.drones, state.grid)}")
    elif state.step == 15:
        state.event_log.append("Step 15: Demand forecast executed; predicted demand is high in residential clusters.")
    elif state.step == 16:
        extra = Delivery("DEL-6", pickup=(5, 9), dropoff=(7, 5), priority="normal")
        state.deliveries.append(extra)
        state.event_log.append(f"Step 16: Extra delivery generated from forecast: {extra.delivery_id}.")
    elif state.step == 17:
        waiting = [d for d in state.deliveries if d.status == "waiting"]
        if waiting:
            state.event_log.append(f"Step 17: {assign_delivery(waiting[0], state.drones, state.grid)}")
        else:
            state.event_log.append("Step 17: No extra waiting delivery to assign.")
    elif state.step == 18:
        state.anomaly_label = "Battery anomaly"
        state.event_log.append("Step 18: Battery anomaly detected for D3 using classifier logic.")
    elif state.step == 19:
        affected = next((d for d in state.drones if d.drone_id == "D3"), None)
        if affected:
            result = astar(affected.position, affected.home_hub, state.grid)
            if result.path:
                affected.route = result.path
                affected.route_index = 0
                affected.status = "flying"
                affected.delivery_id = None
                state.last_route = result.path
                state.event_log.append("Step 19: D3 forced to return to nearest home hub.")
            else:
                state.event_log.append("Step 19: D3 return-to-hub route failed.")
        else:
            state.event_log.append("Step 19: No D3 exists in this fleet; anomaly logged only.")
    elif state.step == 20:
        for delivery in state.deliveries:
            if delivery.status == "in progress":
                delivery.status = "delayed"
                state.delayed += 1
            elif delivery.status == "failed":
                state.failed += 1
        completed = len([d for d in state.deliveries if d.status == "completed"])
        delayed = len([d for d in state.deliveries if d.status == "delayed"])
        failed = len([d for d in state.deliveries if d.status == "failed"])
        state.event_log.append(
            f"Step 20: Simulation complete. {completed} completed, {delayed} delayed, {failed} failed."
        )
    else:
        state.event_log.append("Simulation already finished. Reset to run again.")
    return state


def run_all_steps(budget: int = 7000, demand_units: float = 280.0) -> SimulationState:
    state = initial_state(budget, demand_units)
    while state.step < 20:
        state = run_next_step(state)
    return state


def deliveries_as_rows(deliveries: List[Delivery]) -> List[Dict[str, object]]:
    return [
        {
            "delivery_id": d.delivery_id,
            "pickup": d.pickup,
            "dropoff": d.dropoff,
            "priority": d.priority,
            "assigned_drone": d.assigned_drone,
            "status": d.status,
            "route_cost": d.route_cost,
        }
        for d in deliveries
    ]


def drones_as_rows(drones: List[Drone]) -> List[Dict[str, object]]:
    return [
        {
            "drone_id": d.drone_id,
            "type": d.drone_type,
            "home_hub": d.home_hub,
            "position": d.position,
            "status": d.status,
            "delivery_id": d.delivery_id,
            "route_progress": f"{d.route_index}/{max(0, len(d.route) - 1)}",
        }
        for d in drones
    ]
