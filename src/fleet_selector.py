"""Fleet selection using brute force and a tiny genetic algorithm."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class DroneType:
    name: str
    cost: int
    payload_kg: float
    range_cells: int


LIGHT_DRONE = DroneType("Light Drone", cost=1000, payload_kg=2.0, range_cells=12)
HEAVY_DRONE = DroneType("Heavy Drone", cost=1800, payload_kg=5.0, range_cells=20)
DRONE_TYPES = [LIGHT_DRONE, HEAVY_DRONE]


@dataclass
class FleetOption:
    light_count: int
    heavy_count: int
    cost: int
    capacity_score: float
    coverage_percentage: float
    budget_used_percentage: float
    score: float

    def as_dict(self) -> Dict[str, float | int]:
        return {
            "light_count": self.light_count,
            "heavy_count": self.heavy_count,
            "cost": self.cost,
            "capacity_score": round(self.capacity_score, 2),
            "coverage_%": round(self.coverage_percentage, 2),
            "budget_used_%": round(self.budget_used_percentage, 2),
            "score": round(self.score, 3),
        }


def evaluate_fleet(light_count: int, heavy_count: int, budget: int, demand_units: float) -> FleetOption:
    cost = light_count * LIGHT_DRONE.cost + heavy_count * HEAVY_DRONE.cost
    if cost > budget:
        return FleetOption(light_count, heavy_count, cost, 0, 0, 100, -999)

    # A simple explainable capacity score: payload x range.
    capacity_score = (
        light_count * LIGHT_DRONE.payload_kg * LIGHT_DRONE.range_cells
        + heavy_count * HEAVY_DRONE.payload_kg * HEAVY_DRONE.range_cells
    )
    coverage_percentage = min(100.0, (capacity_score / max(demand_units, 1.0)) * 100.0)
    budget_used_percentage = (cost / budget) * 100.0 if budget else 100.0
    score = (0.75 * coverage_percentage) - (0.25 * budget_used_percentage)
    return FleetOption(light_count, heavy_count, cost, capacity_score, coverage_percentage, budget_used_percentage, score)


def brute_force_fleet(budget: int, demand_units: float) -> Tuple[FleetOption, List[FleetOption]]:
    max_light = budget // LIGHT_DRONE.cost
    max_heavy = budget // HEAVY_DRONE.cost
    options: List[FleetOption] = []
    for light in range(max_light + 1):
        for heavy in range(max_heavy + 1):
            option = evaluate_fleet(light, heavy, budget, demand_units)
            if option.cost <= budget and (light + heavy) > 0:
                options.append(option)
    options.sort(key=lambda opt: opt.score, reverse=True)
    return options[0], options


def genetic_fleet(
    budget: int,
    demand_units: float,
    population_size: int = 18,
    generations: int = 25,
    mutation_rate: float = 0.25,
    seed: int = 7,
) -> Tuple[FleetOption, List[Dict[str, float]]]:
    """Small GA for AI coverage. Chromosome = [light_count, heavy_count]."""
    rng = random.Random(seed)
    max_light = max(1, budget // LIGHT_DRONE.cost)
    max_heavy = max(1, budget // HEAVY_DRONE.cost)

    def random_chromosome() -> List[int]:
        return [rng.randint(0, max_light), rng.randint(0, max_heavy)]

    def repair(chromosome: List[int]) -> List[int]:
        chromosome[0] = max(0, min(max_light, chromosome[0]))
        chromosome[1] = max(0, min(max_heavy, chromosome[1]))
        while chromosome[0] * LIGHT_DRONE.cost + chromosome[1] * HEAVY_DRONE.cost > budget:
            if chromosome[1] > 0 and rng.random() < 0.6:
                chromosome[1] -= 1
            elif chromosome[0] > 0:
                chromosome[0] -= 1
            else:
                break
        if chromosome[0] + chromosome[1] == 0:
            chromosome[0] = 1 if LIGHT_DRONE.cost <= budget else 0
        return chromosome

    population = [repair(random_chromosome()) for _ in range(population_size)]
    history: List[Dict[str, float]] = []

    for gen in range(generations):
        scored = [evaluate_fleet(ch[0], ch[1], budget, demand_units) for ch in population]
        scored.sort(key=lambda opt: opt.score, reverse=True)
        history.append({"generation": gen + 1, "best_score": scored[0].score, "best_cost": scored[0].cost})

        survivors = [[opt.light_count, opt.heavy_count] for opt in scored[: max(2, population_size // 3)]]
        next_population = survivors.copy()
        while len(next_population) < population_size:
            p1 = rng.choice(survivors)
            p2 = rng.choice(survivors)
            child = [p1[0], p2[1]] if rng.random() < 0.5 else [p2[0], p1[1]]
            if rng.random() < mutation_rate:
                idx = rng.choice([0, 1])
                child[idx] += rng.choice([-1, 1])
            next_population.append(repair(child))
        population = next_population

    final_options = [evaluate_fleet(ch[0], ch[1], budget, demand_units) for ch in population]
    final_options.sort(key=lambda opt: opt.score, reverse=True)
    return final_options[0], history
