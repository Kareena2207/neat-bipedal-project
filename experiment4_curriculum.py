"""
Experiment 4 — Curriculum Learning
====================================
Instead of training directly on hardcore terrain, we progressively
increase difficulty as the robot improves:

  Stage 1 (fitness < 50):   Flat terrain, short episodes (800 steps)
  Stage 2 (fitness < 150):  Flat terrain, full episodes (1600 steps)
  Stage 3 (fitness < 220):  Hardcore terrain, full episodes
  Stage 4 (fitness >= 220): Hardcore terrain, extended episodes (2000 steps)

HYPOTHESIS:
  Curriculum learning will reach higher final fitness on hardcore
  than training directly on hardcore from scratch (Experiment 3B),
  because early stages give the robot a foundation before facing
  the full challenge.

COMPARISON:
  At the end, we compare this run's best fitness against
  Experiment 3B's best fitness on hardcore terrain.
"""

import os
import pickle
import numpy as np
import neat
import gymnasium as gym
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR    = os.path.expanduser("~/Desktop/neat_bipedal_project")
CONFIG_PATH = os.path.join(BASE_DIR, "config-neat.txt")

CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints_curriculum")
RESULTS_DIR    = os.path.join(BASE_DIR, "results_curriculum")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR,    exist_ok=True)

# ── Curriculum stage selector ──────────────────────────────────────────────
def get_stage(best_fitness):
    """
    Returns (stage_number, env_name, max_steps) based on current best.
    Called once per generation to determine training conditions.
    """
    if best_fitness < 50:
        return 1, "BipedalWalker-v3", 800
    elif best_fitness < 150:
        return 2, "BipedalWalker-v3", 1600
    elif best_fitness < 220:
        return 3, "BipedalWalkerHardcore-v3", 1600
    else:
        return 4, "BipedalWalkerHardcore-v3", 2000

# ── Global state for curriculum ────────────────────────────────────────────
# We need to share current_best_fitness between the training loop
# and the fitness function. We use a mutable container for this.
curriculum_state = {
    "best_fitness": float("-inf"),
    "current_stage": 1,
    "stage_history": [],   # (generation, stage) tuples for plotting
    "generation": 0
}

# ── Fitness function ───────────────────────────────────────────────────────
def evaluate_curriculum(genomes, config):
    """
    Evaluates all genomes using the current curriculum stage.
    The stage is determined by the best fitness from the previous generation.
    """
    stage, env_name, max_steps = get_stage(
        curriculum_state["best_fitness"]
    )
    curriculum_state["current_stage"] = stage
    curriculum_state["stage_history"].append(
        (curriculum_state["generation"], stage)
    )
    curriculum_state["generation"] += 1

    # Print stage info every 10 generations
    if curriculum_state["generation"] % 10 == 0:
        print(f"  [Curriculum] Stage {stage}: {env_name} "
              f"| max_steps={max_steps} "
              f"| best_so_far={curriculum_state['best_fitness']:.2f}")

    gen_best = float("-inf")

    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        env = gym.make(env_name, render_mode=None)
        obs, _ = env.reset()

        total_reward = 0.0
        for _ in range(max_steps):
            actions = np.clip(net.activate(obs), -1.0, 1.0)
            obs, reward, terminated, truncated, _ = env.step(actions)
            total_reward += reward
            if terminated or truncated:
                break

        env.close()
        genome.fitness = total_reward

        if total_reward > gen_best:
            gen_best = total_reward

    # Update shared best fitness for next generation's stage selection
    if gen_best > curriculum_state["best_fitness"]:
        curriculum_state["best_fitness"] = gen_best

# ── Training loop ──────────────────────────────────────────────────────────
def train_curriculum(generations=1000):
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        CONFIG_PATH
    )

    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    population.add_reporter(
        neat.Checkpointer(
            generation_interval=10,
            filename_prefix=os.path.join(
                CHECKPOINT_DIR, "curriculum-checkpoint-"
            )
        )
    )

    print("\n" + "="*60)
    print("Experiment 4 — Curriculum Learning")
    print(f"Stages: Flat(800) → Flat(1600) → Hardcore → Hardcore(2000)")
    print(f"Generations: {generations}")
    print("="*60 + "\n")

    winner = population.run(evaluate_curriculum, generations)

    # Save winner
    winner_path = os.path.join(RESULTS_DIR, "winner_curriculum.pkl")
    with open(winner_path, "wb") as f:
        pickle.dump(winner, f)

    print(f"\n{'='*60}")
    print(f"CURRICULUM TRAINING COMPLETE")
    print(f"  Best fitness:  {winner.fitness:.4f}")
    print(f"  Nodes:         {len(winner.nodes)}")
    print(f"  Connections:   {len(winner.connections)}")
    print(f"  Final stage:   {curriculum_state['current_stage']}")
    print(f"{'='*60}")

    # ── Plot 1: Fitness history ────────────────────────────────────────────
    gens         = range(len(stats.most_fit_genomes))
    best_fitness = [g.fitness for g in stats.most_fit_genomes]
    avg_fitness  = stats.get_fitness_mean()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Shade background by stage
    stage_colors = {1: "#E3F2FD", 2: "#E8F5E9", 3: "#FFF3E0", 4: "#FCE4EC"}
    stage_labels = {
        1: "Stage 1: Flat/800",
        2: "Stage 2: Flat/1600",
        3: "Stage 3: Hardcore",
        4: "Stage 4: Hardcore/2000"
    }

    if curriculum_state["stage_history"]:
        prev_gen   = 0
        prev_stage = curriculum_state["stage_history"][0][1]
        for gen, stage in curriculum_state["stage_history"][1:]:
            if stage != prev_stage:
                ax1.axvspan(prev_gen, gen,
                           alpha=0.3,
                           color=stage_colors[prev_stage],
                           label=stage_labels[prev_stage])
                ax2.axvspan(prev_gen, gen,
                           alpha=0.3,
                           color=stage_colors[prev_stage])
                prev_gen   = gen
                prev_stage = stage
        # Final segment
        ax1.axvspan(prev_gen, len(gens),
                   alpha=0.3,
                   color=stage_colors[prev_stage],
                   label=stage_labels[prev_stage])
        ax2.axvspan(prev_gen, len(gens),
                   alpha=0.3,
                   color=stage_colors[prev_stage])

    ax1.plot(gens, best_fitness, 'b-', linewidth=2, label='Best fitness')
    ax1.plot(gens, avg_fitness,  'r--', linewidth=1.5,
             label='Avg fitness', alpha=0.7)
    ax1.axhline(y=0,   color='black', linewidth=0.8,
                linestyle='--', alpha=0.4)
    ax1.set_ylabel('Fitness', fontsize=12)
    ax1.set_title('Experiment 4: Curriculum Learning Progress', fontsize=14)
    ax1.legend(fontsize=10, loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Stage number over time
    if curriculum_state["stage_history"]:
        stage_gens   = [s[0] for s in curriculum_state["stage_history"]]
        stage_values = [s[1] for s in curriculum_state["stage_history"]]
        ax2.step(stage_gens, stage_values, 'purple',
                 linewidth=2, where='post')
        ax2.set_ylabel('Curriculum Stage', fontsize=12)
        ax2.set_xlabel('Generation', fontsize=12)
        ax2.set_yticks([1, 2, 3, 4])
        ax2.set_yticklabels([
            'Stage 1\nFlat/800',
            'Stage 2\nFlat/1600',
            'Stage 3\nHardcore',
            'Stage 4\nHardcore+',
        ])
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(RESULTS_DIR, "fitness_curriculum.png"), dpi=150
    )
    plt.close()
    print(f"✓ Curriculum plot saved")

    return winner, stats


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, default=1000)
    args = parser.parse_args()
    train_curriculum(args.generations)
