import random
import math
import statistics
import os
import matplotlib.pyplot as plt

# ----------------------------
# Problem settings
# ----------------------------

POP_SIZE = 60
GENERATIONS = 80
ELITE_SIZE = 10
MUTATION_STD = 0.05
TOURNAMENT_SIZE = 2
RUNS = 20

LOWER_BOUND = 0.0
UPPER_BOUND = 1.0

RESULTS_DIR = "results"


# ----------------------------
# Basic utilities
# ----------------------------

def clip(value, lower=LOWER_BOUND, upper=UPPER_BOUND):
    return max(lower, min(upper, value))


def random_individual():
    return [random.uniform(0, 1), random.uniform(0, 1)]


def evaluate(ind):
    x, y = ind
    f1 = x**2 + y**2
    f2 = (x - 1)**2 + y**2
    return (f1, f2)


def scalar_score(ind):
    f1, f2 = evaluate(ind)
    return f1 + f2


def crossover(p1, p2):
    # Simple arithmetic crossover
    alpha = random.random()
    child_x = alpha * p1[0] + (1 - alpha) * p2[0]
    child_y = alpha * p1[1] + (1 - alpha) * p2[1]
    return [child_x, child_y]


def mutate(ind, std=MUTATION_STD):
    x, y = ind
    x += random.gauss(0, std)
    y += random.gauss(0, std)
    return [clip(x), clip(y)]


# ----------------------------
# Scalar GA
# ----------------------------

def tournament_select_scalar(population):
    contestants = random.sample(population, TOURNAMENT_SIZE)
    contestants.sort(key=scalar_score)
    return contestants[0]


def run_scalar_ga(seed=None):
    if seed is not None:
        random.seed(seed)

    population = [random_individual() for _ in range(POP_SIZE)]

    for _ in range(GENERATIONS):
        population.sort(key=scalar_score)
        new_population = population[:ELITE_SIZE]

        while len(new_population) < POP_SIZE:
            p1 = tournament_select_scalar(population)
            p2 = tournament_select_scalar(population)
            child = crossover(p1, p2)
            child = mutate(child)
            new_population.append(child)

        population = new_population

    population.sort(key=scalar_score)
    return population


# ----------------------------
# Pareto GA
# ----------------------------

def dominates(a, b):
    f1a, f2a = evaluate(a)
    f1b, f2b = evaluate(b)

    no_worse = (f1a <= f1b) and (f2a <= f2b)
    strictly_better = (f1a < f1b) or (f2a < f2b)
    return no_worse and strictly_better


def fast_nondominated_sort(population):
    domination_sets = {}
    dominated_count = {}
    fronts = [[]]

    for p in population:
        domination_sets[id(p)] = []
        dominated_count[id(p)] = 0

        for q in population:
            if p is q:
                continue
            if dominates(p, q):
                domination_sets[id(p)].append(q)
            elif dominates(q, p):
                dominated_count[id(p)] += 1

        if dominated_count[id(p)] == 0:
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in domination_sets[id(p)]:
                dominated_count[id(q)] -= 1
                if dominated_count[id(q)] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)

    fronts.pop()  # remove empty last front
    return fronts


def crowding_distance(front):
    if not front:
        return {}

    distances = {id(ind): 0.0 for ind in front}
    objectives = [0, 1]  # f1 and f2

    values = {id(ind): evaluate(ind) for ind in front}

    for obj_index in objectives:
        sorted_front = sorted(front, key=lambda ind: values[id(ind)][obj_index])

        distances[id(sorted_front[0])] = float("inf")
        distances[id(sorted_front[-1])] = float("inf")

        min_val = values[id(sorted_front[0])][obj_index]
        max_val = values[id(sorted_front[-1])][obj_index]

        if max_val == min_val:
            continue

        for i in range(1, len(sorted_front) - 1):
            prev_val = values[id(sorted_front[i - 1])][obj_index]
            next_val = values[id(sorted_front[i + 1])][obj_index]
            distances[id(sorted_front[i])] += (next_val - prev_val) / (max_val - min_val)

    return distances


def assign_rank_and_crowding(population):
    fronts = fast_nondominated_sort(population)
    info = {}

    for rank, front in enumerate(fronts):
        distances = crowding_distance(front)
        for ind in front:
            info[id(ind)] = {
                "rank": rank,
                "crowding": distances[id(ind)]
            }

    return fronts, info


def better_pareto(ind1, ind2, info):
    a = info[id(ind1)]
    b = info[id(ind2)]

    if a["rank"] < b["rank"]:
        return ind1
    if b["rank"] < a["rank"]:
        return ind2

    if a["crowding"] > b["crowding"]:
        return ind1
    return ind2


def tournament_select_pareto(population, info):
    contestants = random.sample(population, TOURNAMENT_SIZE)
    winner = contestants[0]
    for c in contestants[1:]:
        winner = better_pareto(winner, c, info)
    return winner


def environmental_selection(population, size):
    fronts = fast_nondominated_sort(population)
    new_population = []

    for front in fronts:
        if len(new_population) + len(front) <= size:
            new_population.extend(front)
        else:
            distances = crowding_distance(front)
            sorted_front = sorted(front, key=lambda ind: distances[id(ind)], reverse=True)
            remaining = size - len(new_population)
            new_population.extend(sorted_front[:remaining])
            break

    return new_population


def run_pareto_ga(seed=None):
    if seed is not None:
        random.seed(seed)

    population = [random_individual() for _ in range(POP_SIZE)]

    for _ in range(GENERATIONS):
        _, info = assign_rank_and_crowding(population)

        offspring = []
        while len(offspring) < POP_SIZE:
            p1 = tournament_select_pareto(population, info)
            p2 = tournament_select_pareto(population, info)
            child = crossover(p1, p2)
            child = mutate(child)
            offspring.append(child)

        combined = population + offspring
        population = environmental_selection(combined, POP_SIZE)

    return population


# ----------------------------
# Analysis helpers
# ----------------------------

def get_nondominated_set(population):
    nondominated = []
    for p in population:
        dominated = False
        for q in population:
            if p is not q and dominates(q, p):
                dominated = True
                break
        if not dominated:
            nondominated.append(p)
    return nondominated


def x_diversity(population):
    xs = [ind[0] for ind in population]
    if len(xs) < 2:
        return 0.0
    return statistics.stdev(xs)


def summarize_population(population):
    objs = [evaluate(ind) for ind in population]
    xs = [ind[0] for ind in population]
    ys = [ind[1] for ind in population]
    return {
        "f1_values": [o[0] for o in objs],
        "f2_values": [o[1] for o in objs],
        "x_values": xs,
        "y_values": ys,
        "x_diversity": x_diversity(population),
        "nondominated_count": len(get_nondominated_set(population)),
    }


# ----------------------------
# Plotting
# ----------------------------

def ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def plot_objective_space(scalar_pop, pareto_pop, filename):
    scalar_objs = [evaluate(ind) for ind in scalar_pop]
    pareto_objs = [evaluate(ind) for ind in pareto_pop]

    plt.figure(figsize=(7, 5))
    plt.scatter(
        [o[0] for o in scalar_objs],
        [o[1] for o in scalar_objs],
        alpha=0.7,
        label="Scalar GA"
    )
    plt.scatter(
        [o[0] for o in pareto_objs],
        [o[1] for o in pareto_objs],
        alpha=0.7,
        label="Pareto GA"
    )
    plt.xlabel("f1")
    plt.ylabel("f2")
    plt.title("Final Population in Objective Space")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


def plot_decision_space(scalar_pop, pareto_pop, filename):
    plt.figure(figsize=(7, 5))
    plt.scatter(
        [ind[0] for ind in scalar_pop],
        [ind[1] for ind in scalar_pop],
        alpha=0.7,
        label="Scalar GA"
    )
    plt.scatter(
        [ind[0] for ind in pareto_pop],
        [ind[1] for ind in pareto_pop],
        alpha=0.7,
        label="Pareto GA"
    )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Final Population in Decision Space")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


# ----------------------------
# Main experiment
# ----------------------------

def main():
    ensure_results_dir()

    scalar_diversities = []
    pareto_diversities = []

    scalar_nd_counts = []
    pareto_nd_counts = []

    last_scalar_pop = None
    last_pareto_pop = None

    for run in range(RUNS):
        scalar_pop = run_scalar_ga(seed=run)
        pareto_pop = run_pareto_ga(seed=run)

        scalar_summary = summarize_population(scalar_pop)
        pareto_summary = summarize_population(pareto_pop)

        scalar_diversities.append(scalar_summary["x_diversity"])
        pareto_diversities.append(pareto_summary["x_diversity"])

        scalar_nd_counts.append(scalar_summary["nondominated_count"])
        pareto_nd_counts.append(pareto_summary["nondominated_count"])

        last_scalar_pop = scalar_pop
        last_pareto_pop = pareto_pop

    plot_objective_space(
        last_scalar_pop,
        last_pareto_pop,
        os.path.join(RESULTS_DIR, "objective_space.png")
    )

    plot_decision_space(
        last_scalar_pop,
        last_pareto_pop,
        os.path.join(RESULTS_DIR, "decision_space.png")
    )

    print("=== Summary over", RUNS, "runs ===")
    print()
    print("Scalar GA")
    print("Mean x-diversity:", round(statistics.mean(scalar_diversities), 4))
    print("Std x-diversity :", round(statistics.stdev(scalar_diversities), 4))
    print("Mean nondominated count:", round(statistics.mean(scalar_nd_counts), 2))
    print()
    print("Pareto GA")
    print("Mean x-diversity:", round(statistics.mean(pareto_diversities), 4))
    print("Std x-diversity :", round(statistics.stdev(pareto_diversities), 4))
    print("Mean nondominated count:", round(statistics.mean(pareto_nd_counts), 2))
    print()
    print("Plots saved in:", RESULTS_DIR)


if __name__ == "__main__":
    main()
