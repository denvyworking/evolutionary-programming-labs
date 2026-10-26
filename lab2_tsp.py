"""Laboratory work 2: permutation genetic algorithm for a travelling salesman problem."""

from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path


CITY_COUNT = 20
DATA_SEED = 20260226
RUN_SEED_START = 20260201
POPULATION_SIZE = 60
GENERATIONS = 99
EVALUATIONS = POPULATION_SIZE * (GENERATIONS + 1)


@dataclass(frozen=True)
class City:
    index: int
    name: str
    x: float
    y: float


@dataclass
class RunResult:
    seed: int
    best_value: float
    best_route: list[int]
    history: list[float]


def make_cities() -> list[City]:
    rng = random.Random(DATA_SEED)
    return [City(i, f"C{i + 1:02d}", rng.uniform(0, 100), rng.uniform(0, 100))
            for i in range(CITY_COUNT)]


def distance_matrix(cities: list[City]) -> list[list[float]]:
    return [
        [math.hypot(a.x - b.x, a.y - b.y) for b in cities]
        for a in cities
    ]


def route_length(route: list[int], distances: list[list[float]]) -> float:
    return sum(distances[route[i]][route[(i + 1) % len(route)]]
               for i in range(len(route)))


def is_valid_route(route: list[int]) -> bool:
    return len(route) == CITY_COUNT and set(route) == set(range(CITY_COUNT))


def tournament(population: list[list[int]], scores: list[float],
               rng: random.Random, size: int = 3) -> list[int]:
    candidates = [rng.randrange(len(population)) for _ in range(size)]
    return population[min(candidates, key=lambda i: scores[i])].copy()


def ox_crossover(first: list[int], second: list[int],
                 rng: random.Random) -> tuple[list[int], list[int]]:
    left, right = sorted(rng.sample(range(len(first)), 2))

    def make_child(parent_a: list[int], parent_b: list[int]) -> list[int]:
        child = [-1] * len(parent_a)
        child[left:right] = parent_a[left:right]
        remaining = [gene for gene in parent_b if gene not in child]
        position = right
        for gene in remaining:
            if position == len(child):
                position = 0
            child[position] = gene
            position += 1
        return child

    return make_child(first, second), make_child(second, first)


def pmx_crossover(first: list[int], second: list[int],
                  rng: random.Random) -> tuple[list[int], list[int]]:
    left, right = sorted(rng.sample(range(len(first)), 2))

    def make_child(parent_a: list[int], parent_b: list[int]) -> list[int]:
        child = [-1] * len(parent_a)
        child[left:right] = parent_a[left:right]
        for index in range(left, right):
            if parent_b[index] in child:
                continue
            target = index
            while left <= target < right:
                target = parent_b.index(parent_a[target])
            child[target] = parent_b[index]
        for index, gene in enumerate(parent_b):
            if child[index] == -1:
                child[index] = gene
        return child

    return make_child(first, second), make_child(second, first)


def mutate_swap(route: list[int], rng: random.Random) -> None:
    first, second = rng.sample(range(len(route)), 2)
    route[first], route[second] = route[second], route[first]


def mutate_inversion(route: list[int], rng: random.Random) -> None:
    first, second = sorted(rng.sample(range(len(route)), 2))
    route[first:second] = reversed(route[first:second])


def run_ga(seed: int, distances: list[list[float]], operator: str) -> RunResult:
    rng = random.Random(seed)
    population = [rng.sample(range(CITY_COUNT), CITY_COUNT)
                   for _ in range(POPULATION_SIZE)]
    scores = [route_length(route, distances) for route in population]
    best_index = min(range(len(population)), key=scores.__getitem__)
    best_route = population[best_index].copy()
    best_value = scores[best_index]
    history = [best_value]
    crossover = ox_crossover if operator == "OX" else pmx_crossover

    for _ in range(GENERATIONS):
        next_population = [best_route.copy()]
        while len(next_population) < POPULATION_SIZE:
            first = tournament(population, scores, rng)
            second = tournament(population, scores, rng)
            children = crossover(first, second, rng)
            for child in children:
                if rng.random() < 0.2:
                    mutate_swap(child, rng)
                if rng.random() < 0.2:
                    mutate_inversion(child, rng)
                next_population.append(child)
                if len(next_population) == POPULATION_SIZE:
                    break
        population = next_population
        scores = [route_length(route, distances) for route in population]
        current = min(range(len(population)), key=scores.__getitem__)
        if scores[current] < best_value:
            best_value = scores[current]
            best_route = population[current].copy()
        history.append(best_value)
    if not is_valid_route(best_route):
        raise RuntimeError("GA produced an invalid permutation")
    return RunResult(seed, best_value, best_route, history)


def nearest_neighbor(distances: list[list[float]]) -> RunResult:
    route = [0]
    remaining = set(range(1, CITY_COUNT))
    while remaining:
        route.append(min(remaining, key=lambda city: distances[route[-1]][city]))
        remaining.remove(route[-1])
    value = route_length(route, distances)
    return RunResult(0, value, route, [value])


def random_search(seed: int, distances: list[list[float]]) -> RunResult:
    rng = random.Random(seed)
    best_value = float("inf")
    best_route: list[int] = []
    history = []
    for _ in range(EVALUATIONS):
        route = rng.sample(range(CITY_COUNT), CITY_COUNT)
        value = route_length(route, distances)
        if value < best_value:
            best_value, best_route = value, route
        history.append(best_value)
    return RunResult(seed, best_value, best_route, history)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_convergence_svg(path: Path, series: dict[str, list[float]]) -> None:
    width, height, left, top, right, bottom = 1000, 560, 75, 35, 25, 65
    values = [value for line in series.values() for value in line]
    low, high = min(values), max(values)
    plot_width, plot_height = width - left - right, height - top - bottom

    def point(index: int, value: float, count: int) -> tuple[float, float]:
        x = left + index * plot_width / max(1, count - 1)
        y = top + (high - value) * plot_height / max(1e-12, high - low)
        return x, y

    colors = ["#1f77b4", "#d62728", "#2ca02c"]
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="black"/>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="black"/>',
        f'<text x="{width/2}" y="{height-18}" text-anchor="middle">Generation / evaluation</text>',
        f'<text transform="translate(18,{height/2}) rotate(-90)" text-anchor="middle">Route length</text>',
    ]
    for line_index, (name, values) in enumerate(series.items()):
        coordinates = " ".join(
            f"{x:.1f},{y:.1f}" for index, value in enumerate(values)
            for x, y in [point(index, value, len(values))]
        )
        color = colors[line_index % len(colors)]
        lines += [
            f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="2"/>',
            f'<text x="{width-right-180}" y="{top + 20*line_index}" fill="{color}">{name}</text>',
        ]
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_statistics_svg(path: Path, values: dict[str, dict[str, float]]) -> None:
    width, height, left, top, right, bottom = 1000, 600, 90, 45, 35, 80
    plot_width, plot_height = width - left - right, height - top - bottom
    maximum = max(statistic for method in values.values() for statistic in method.values())
    colors = {"min": "#2ca02c", "mean": "#1f77b4", "median": "#ff7f0e", "max": "#d62728"}
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="black"/>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="black"/>',
        f'<text x="{width/2}" y="{height-22}" text-anchor="middle">Method</text>',
        f'<text transform="translate(20,{height/2}) rotate(-90)" text-anchor="middle">Best route length</text>',
    ]
    methods = list(values)
    group_width = plot_width / len(methods)
    bar_width = group_width / (len(colors) + 1)
    for method_index, method in enumerate(methods):
        start = left + method_index * group_width
        for statistic_index, (statistic, color) in enumerate(colors.items()):
            value = values[method][statistic]
            bar_height = value / maximum * plot_height if maximum else 0
            x = start + (statistic_index + 0.5) * bar_width
            y = height - bottom - bar_height
            lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width*.8:.1f}" '
                         f'height="{bar_height:.1f}" fill="{color}"/>')
        lines.append(f'<text x="{start + group_width/2:.1f}" y="{height-bottom+20}" '
                     'text-anchor="middle">' + method + '</text>')
    for index, (name, color) in enumerate(colors.items()):
        lines.append(f'<rect x="{width-right-210+index*52}" y="15" width="12" height="12" fill="{color}"/>')
        lines.append(f'<text x="{width-right-195+index*52}" y="26">{name}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_route_svg(path: Path, cities: list[City], route: list[int]) -> None:
    width, height, margin = 900, 650, 55
    xs = [city.x for city in cities]
    ys = [city.y for city in cities]
    scale = min((width - 2 * margin) / (max(xs) - min(xs)),
                (height - 2 * margin) / (max(ys) - min(ys)))

    def point(city: City) -> tuple[float, float]:
        return margin + (city.x - min(xs)) * scale, height - margin - (city.y - min(ys)) * scale

    points = [point(cities[index]) for index in route + [route[0]]]
    polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             f'<polyline points="{polyline}" fill="none" stroke="#1f77b4" stroke-width="2"/>']
    for city in cities:
        x, y = point(city)
        lines += [f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="#d62728"/>',
                  f'<text x="{x+7:.1f}" y="{y-7:.1f}">{city.name}</text>']
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_flowchart_svg(path: Path) -> None:
    labels = ["Generate permutation population", "Evaluate closed routes",
              "Tournament selection", "OX or PMX crossover",
              "Swap/inversion mutation", "Keep elite and repeat", "Save best route"]
    lines = ['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="780">',
             '<rect width="100%" height="100%" fill="white"/>']
    for index, label in enumerate(labels):
        y = 30 + index * 100
        lines += [f'<rect x="300" y="{y}" width="400" height="55" rx="12" fill="#eaf2f8" stroke="#21618c"/>',
                  f'<text x="500" y="{y+34}" text-anchor="middle">{label}</text>']
        if index:
            lines.append(f'<line x1="500" y1="{y-45}" x2="500" y2="{y}" stroke="#333" marker-end="url(#arrow)"/>')
    lines.insert(1, '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#333"/></marker></defs>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_experiments(output_dir: Path, runs: int = 20) -> None:
    cities = make_cities()
    distances = distance_matrix(cities)
    results: dict[str, list[RunResult]] = {"OX": [], "PMX": [], "Random search": []}
    for offset in range(runs):
        seed = RUN_SEED_START + offset
        results["OX"].append(run_ga(seed, distances, "OX"))
        results["PMX"].append(run_ga(seed, distances, "PMX"))
        results["Random search"].append(random_search(seed, distances))
    heuristic = nearest_neighbor(distances)

    rows = []
    for method, method_results in results.items():
        for result in method_results:
            rows.append({"method": method, "seed": result.seed,
                         "best_length": f"{result.best_value:.10f}",
                         "route": " ".join(str(city + 1) for city in result.best_route),
                         "evaluations": EVALUATIONS})
    rows.append({"method": "Nearest neighbor", "seed": heuristic.seed,
                 "best_length": f"{heuristic.best_value:.10f}",
                 "route": " ".join(str(city + 1) for city in heuristic.best_route),
                 "evaluations": CITY_COUNT})
    write_csv(output_dir / "results.csv", rows)
    write_csv(output_dir / "cities.csv",
              [{"city": city.name, "x": f"{city.x:.6f}", "y": f"{city.y:.6f}",
                "data_seed": DATA_SEED} for city in cities])

    trajectory = {}
    for method in ("OX", "PMX", "Random search"):
        histories = [result.history for result in results[method]]
        if method == "Random search":
            indices = [round(i * (EVALUATIONS - 1) / GENERATIONS)
                       for i in range(GENERATIONS + 1)]
            histories = [[history[index] for index in indices]
                         for history in histories]
        length = min(len(history) for history in histories)
        trajectory[method] = [statistics.mean(history[i] for history in histories)
                              for i in range(length)]
    write_csv(output_dir / "trajectory.csv",
              [{"step": i, **{method: f"{trajectory[method][i]:.10f}"
                               for method in trajectory}}
               for i in range(len(trajectory["OX"]))])
    write_convergence_svg(output_dir / "convergence.svg", trajectory)

    stats = {}
    for method, method_results in results.items():
        values = [result.best_value for result in method_results]
        stats[method] = {"min": min(values), "mean": statistics.mean(values),
                         "median": statistics.median(values), "max": max(values)}
    heuristic_stats = heuristic.best_value
    stats["Nearest neighbor"] = {"min": heuristic_stats, "mean": heuristic_stats,
                                 "median": heuristic_stats, "max": heuristic_stats}
    write_statistics_svg(output_dir / "statistics.svg", stats)
    best = min(results["OX"] + results["PMX"], key=lambda result: result.best_value)
    write_route_svg(output_dir / "best_route.svg", cities, best.best_route)
    write_flowchart_svg(output_dir / "flowchart.svg")

    print(f"cities={CITY_COUNT} evaluations_per_run={EVALUATIONS} runs={runs}")
    for method, values in stats.items():
        print(f"{method}: mean={values['mean']:.6f} best={values['min']:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-experiments", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("lab2_results"))
    args = parser.parse_args()
    if args.run_experiments:
        run_experiments(args.output_dir)
    else:
        cities = make_cities()
        result = run_ga(RUN_SEED_START, distance_matrix(cities), "OX")
        print(f"seed={result.seed} best={result.best_value:.12g}")
        print("route=", " ".join(str(city + 1) for city in result.best_route))


if __name__ == "__main__":
    main()
