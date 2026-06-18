"""
Experiment 3 — Hardcore Environment Transfer Test
==================================================
Part A: Take the stability winner (best walker so far)
        and run it on BipedalWalkerHardcore-v3 WITHOUT retraining.
        Shows whether evolved policies generalize to new challenges.

Part B: Train a fresh NEAT population directly on hardcore.
        Shows what structure NEAT evolves when the task is harder.

BipedalWalkerHardcore-v3 adds:
  - Stumps (small obstacles)
  - Stairs
  - Pitfalls (gaps in the ground)
  The robot must navigate all of these with the same 24 sensors / 4 motors.
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

# ── Part A: Transfer test ──────────────────────────────────────────────────
def transfer_test(num_episodes=20):
    """
    Load the stability winner and run it on hardcore WITHOUT retraining.
    We expect it to fail badly — it was never trained on obstacles.
    """
    print("\n" + "="*60)
    print("PART A — Transfer Test")
    print("Stability winner vs BipedalWalkerHardcore-v3")
    print("="*60)

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        CONFIG_PATH
    )

    winner_path = os.path.join(
        BASE_DIR, "results_stability", "winner_stability.pkl"
    )
    with open(winner_path, "rb") as f:
        genome = pickle.load(f)

    print(f"\nLoaded stability winner:")
    print(f"  Fitness (flat terrain): {genome.fitness:.4f}")
    print(f"  Nodes:                  {len(genome.nodes)}")
    print(f"  Connections:            {len(genome.connections)}")
    print(f"\nRunning {num_episodes} episodes on HARDCORE terrain...")

    net = neat.nn.FeedForwardNetwork.create(genome, config)

    rewards = []
    steps_list = []
    failures = 0

    for episode in range(num_episodes):
        # Only difference: BipedalWalkerHardcore-v3
        env = gym.make("BipedalWalkerHardcore-v3", render_mode=None)
        obs, _ = env.reset()
        total_reward = 0.0
        steps = 0

        for _ in range(1600):
            actions = np.clip(net.activate(obs), -1.0, 1.0)
            obs, reward, terminated, truncated, _ = env.step(actions)
            total_reward += reward
            steps += 1
            if terminated or truncated:
                break

        env.close()
        rewards.append(total_reward)
        steps_list.append(steps)
        if total_reward < 0:
            failures += 1

        print(f"  Episode {episode+1:2d}: "
              f"steps={steps:4d} | reward={total_reward:8.2f}")

    print(f"\n{'='*60}")
    print(f"TRANSFER TEST RESULTS")
    print(f"  Avg reward on hardcore:  {np.mean(rewards):.2f} "
          f"± {np.std(rewards):.2f}")
    print(f"  Avg reward on flat:      272.92 ± 6.89  (from Exp 2)")
    print(f"  Performance drop:        "
          f"{272.92 - np.mean(rewards):.2f} points")
    print(f"  Failure rate:            "
          f"{failures}/{num_episodes} "
          f"({failures/num_episodes*100:.0f}%)")
    print(f"  Avg steps survived:      {np.mean(steps_list):.0f}")
    print(f"{'='*60}")

    return np.mean(rewards)


# ── Part B: Train from scratch on hardcore ─────────────────────────────────
def train_hardcore(generations=1000):
    """
    Train a fresh NEAT population directly on hardcore terrain.
    This will take much longer than flat terrain training.
    We expect NEAT to evolve more complex networks to handle obstacles.
    """
    print("\n" + "="*60)
    print("PART B — Train from scratch on Hardcore")
    print("BipedalWalkerHardcore-v3 | Fresh population")
    print("="*60 + "\n")

    checkpoint_dir = os.path.join(BASE_DIR, "checkpoints_hardcore")
    results_dir    = os.path.join(BASE_DIR, "results_hardcore")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir,    exist_ok=True)

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        CONFIG_PATH
    )

    def evaluate_hardcore(genomes, config):
        for genome_id, genome in genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            env = gym.make("BipedalWalkerHardcore-v3", render_mode=None)
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

    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    population.add_reporter(
        neat.Checkpointer(
            generation_interval=10,
            filename_prefix=os.path.join(
                checkpoint_dir, "hardcore-checkpoint-"
            )
        )
    )

    winner = population.run(evaluate_hardcore, generations)

    with open(os.path.join(results_dir, "winner_hardcore.pkl"), "wb") as f:
        pickle.dump(winner, f)

    print(f"\n{'='*60}")
    print(f"HARDCORE TRAINING COMPLETE")
    print(f"  Best fitness:  {winner.fitness:.4f}")
    print(f"  Nodes:         {len(winner.nodes)}")
    print(f"  Connections:   {len(winner.connections)}")
    print(f"{'='*60}")

    # Plot fitness
    gens         = range(len(stats.most_fit_genomes))
    best_fitness = [g.fitness for g in stats.most_fit_genomes]
    avg_fitness  = stats.get_fitness_mean()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(gens, best_fitness, 'b-',  linewidth=2, label='Best (Hardcore)')
    ax.plot(gens, avg_fitness,  'r--', linewidth=1.5,
            label='Average (Hardcore)', alpha=0.7)
    ax.axhline(y=272.92, color='green', linestyle='--', linewidth=1.5,
               label='Stability winner on flat (272.92)')
    ax.set_xlabel('Generation', fontsize=13)
    ax.set_ylabel('Fitness', fontsize=13)
    ax.set_title('Experiment 3: Training on Hardcore Terrain', fontsize=15)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(results_dir, "fitness_hardcore.png"), dpi=150
    )
    plt.close()
    print(f"✓ Plot saved")

    return winner, stats


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--part",
        choices=["transfer", "train"],
        required=True,
        help="transfer = test stability winner on hardcore | "
             "train = fresh NEAT on hardcore"
    )
    parser.add_argument("--generations", type=int, default=1000)
    parser.add_argument("--episodes",    type=int, default=20)
    args = parser.parse_args()

    if args.part == "transfer":
        transfer_test(args.episodes)
    elif args.part == "train":
        train_hardcore(args.generations)
