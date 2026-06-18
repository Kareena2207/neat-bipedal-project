"""
Experiment 2 — Multi-Objective Fitness Functions
=================================================
We train THREE separate NEAT populations on BipedalWalker-v3.
Each population uses a different fitness function:

  Version A — Speed:      pure forward distance (baseline, already done)
  Version B — Efficiency: penalize large motor torques
  Version C — Stability:  penalize body tilt

Each version will evolve a DIFFERENT walking strategy even though
the robot and environment are identical. This proves that the
fitness function directly shapes the evolved behavior.
"""

import os
import pickle
import argparse
import numpy as np
import neat
import gymnasium as gym
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR    = os.path.expanduser("~/Desktop/neat_bipedal_project")
CONFIG_PATH = os.path.join(BASE_DIR, "config-neat.txt")

# ── Fitness Functions ──────────────────────────────────────────────────────

def evaluate_speed(genomes, config):
    """
    Version A — Pure speed.
    Identical to baseline. Rewards forward distance only.
    Already done — included here for reference/re-running.
    """
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        env = gym.make("BipedalWalker-v3", render_mode=None)
        obs, _ = env.reset()
        total_reward = 0.0
        for _ in range(1600):
            actions = np.clip(net.activate(obs), -1.0, 1.0)
            obs, reward, terminated, truncated, _ = env.step(actions)
            total_reward += reward
            if terminated or truncated:
                break
        env.close()
        genome.fitness = total_reward


def evaluate_efficiency(genomes, config):
    """
    Version B — Energy efficiency.
    Penalizes large motor torques on top of the base reward.
    obs[2] = horizontal velocity — we use this to confirm forward motion.

    The energy penalty discourages the robot from using full motor power
    at every step. This should evolve a smoother, gliding gait that
    uses minimal motor effort to maintain forward motion.

    Penalty scale 0.5: enough to discourage waste without overwhelming
    the forward-motion reward signal.
    """
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        env = gym.make("BipedalWalker-v3", render_mode=None)
        obs, _ = env.reset()
        total_reward   = 0.0
        energy_penalty = 0.0
        for _ in range(1600):
            actions = np.clip(net.activate(obs), -1.0, 1.0)
            obs, reward, terminated, truncated, _ = env.step(actions)
            total_reward   += reward
            # Penalize the sum of absolute motor torques each step
            energy_penalty += sum(abs(a) for a in actions) * 0.001
            if terminated or truncated:
                break
        env.close()
        genome.fitness = total_reward - energy_penalty


def evaluate_stability(genomes, config):
    """
    Version C — Stability.
    Penalizes body tilt (hull angle) at every step.
    obs[0] = hull angle (0 = perfectly upright, positive = tilting forward)
    obs[1] = hull angular velocity (how fast it is tilting)

    This should evolve a very upright, careful walker that prioritizes
    keeping the body level over moving fast. Expect a slower but more
    visually natural gait.

    Penalty scale 2.0: hull angle is small (radians) so we scale up
    to make the penalty meaningful relative to the reward signal.
    """
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        env = gym.make("BipedalWalker-v3", render_mode=None)
        obs, _ = env.reset()
        total_reward    = 0.0
        tilt_penalty    = 0.0
        for _ in range(1600):
            actions = np.clip(net.activate(obs), -1.0, 1.0)
            obs, reward, terminated, truncated, _ = env.step(actions)
            total_reward += reward
            # Penalize both tilt angle and angular velocity
            tilt_penalty += (abs(obs[0]) * 0.01 + abs(obs[1]) * 0.005)
            if terminated or truncated:
                break
        env.close()
        genome.fitness = total_reward - tilt_penalty


# ── Training function ──────────────────────────────────────────────────────

def run_experiment(version, eval_fn, generations=500):
    """
    Run one version of the multi-objective experiment.
    version: 'efficiency' or 'stability'
    """
    checkpoint_dir = os.path.join(BASE_DIR, f"checkpoints_{version}")
    results_dir    = os.path.join(BASE_DIR, f"results_{version}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir,    exist_ok=True)

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
                checkpoint_dir, f"{version}-checkpoint-"
            )
        )
    )

    print(f"\n{'='*60}")
    print(f"Experiment 2 — Version: {version.upper()}")
    print(f"Generations: {generations}")
    print(f"{'='*60}\n")

    winner = population.run(eval_fn, generations)

    # Save winner
    winner_path = os.path.join(results_dir, f"winner_{version}.pkl")
    with open(winner_path, "wb") as f:
        pickle.dump(winner, f)

    print(f"\n{'='*60}")
    print(f"COMPLETE: {version.upper()}")
    print(f"  Best fitness: {winner.fitness:.4f}")
    print(f"  Nodes:        {len(winner.nodes)}")
    print(f"  Connections:  {len(winner.connections)}")
    print(f"{'='*60}\n")

    # Save fitness plot
    generations_range = range(len(stats.most_fit_genomes))
    best_fitness      = [g.fitness for g in stats.most_fit_genomes]
    avg_fitness       = stats.get_fitness_mean()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(generations_range, best_fitness, linewidth=2,
            label=f'Best ({version})')
    ax.plot(generations_range, avg_fitness, '--', linewidth=1.5,
            label=f'Average ({version})', alpha=0.7)
    ax.set_xlabel('Generation', fontsize=13)
    ax.set_ylabel('Fitness', fontsize=13)
    ax.set_title(f'Experiment 2: {version.capitalize()} Fitness', fontsize=15)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"fitness_{version}.png"), dpi=150)
    plt.close()
    print(f"✓ Plot saved to results_{version}/fitness_{version}.png")

    return winner, stats


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--version",
        choices=["efficiency", "stability"],
        required=True,
        help="Which fitness function to run"
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=500
    )
    args = parser.parse_args()

    if args.version == "efficiency":
        run_experiment("efficiency", evaluate_efficiency, args.generations)
    elif args.version == "stability":
        run_experiment("stability", evaluate_stability, args.generations)
