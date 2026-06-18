"""
Ablation Study — NEAT with topology evolution DISABLED
=======================================================
This script trains NEAT with node_add_prob = 0 and conn_add_prob = 0,
meaning the network structure is FIXED from generation 0 forever.
Only weights and biases can change — no new nodes, no new connections.

PURPOSE:
  Compare this run's fitness curve against your baseline NEAT run.
  If topology evolution matters, this run should plateau at a lower
  fitness than your baseline 256.48 score.

This directly proves NEAT's core contribution to the field.
"""

import os
import pickle
import numpy as np
import neat
import gymnasium as gym
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.expanduser("~/Desktop/neat_bipedal_project")
CONFIG_PATH    = os.path.join(BASE_DIR, "config-neat-ablation.txt")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints_ablation")
RESULTS_DIR    = os.path.join(BASE_DIR, "results_ablation")
WINNER_PATH    = os.path.join(RESULTS_DIR, "winner_ablation.pkl")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR,    exist_ok=True)

# ── Fitness function (identical to baseline) ───────────────────────────────
def evaluate_genomes(genomes, config):
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

# ── Load ablation config ───────────────────────────────────────────────────
config = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    CONFIG_PATH
)

# ── Confirm topology mutation is disabled ─────────────────────────────────
print("\n" + "="*60)
print("ABLATION STUDY — Fixed Topology NEAT")
print("="*60)
print(f"  node_add_prob: {config.genome_config.node_add_prob}")
print(f"  conn_add_prob: {config.genome_config.conn_add_prob}")
print(f"  Population:    {config.pop_size}")
print("  Topology evolution: DISABLED")
print("  Only weights and biases will evolve")
print("="*60 + "\n")

assert config.genome_config.node_add_prob == 0.0, "node_add_prob must be 0!"
assert config.genome_config.conn_add_prob == 0.0, "conn_add_prob must be 0!"

# ── Create population ──────────────────────────────────────────────────────
population = neat.Population(config)

population.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
population.add_reporter(stats)
population.add_reporter(
    neat.Checkpointer(
        generation_interval=10,
        filename_prefix=os.path.join(CHECKPOINT_DIR, "ablation-checkpoint-")
    )
)

# ── Run 500 generations ────────────────────────────────────────────────────
# 500 is enough to see where fixed-topology plateaus.
# Your baseline hit 256 — we expect this to plateau much lower.
NUM_GENERATIONS = 500
winner = population.run(evaluate_genomes, NUM_GENERATIONS)

# ── Save winner ────────────────────────────────────────────────────────────
with open(WINNER_PATH, "wb") as f:
    pickle.dump(winner, f)

print(f"\n{'='*60}")
print(f"ABLATION STUDY COMPLETE")
print(f"  Best fitness:  {winner.fitness:.4f}")
print(f"  Nodes:         {len(winner.nodes)}")
print(f"  Connections:   {len(winner.connections)}")
print(f"{'='*60}\n")

# ── Plot fitness history ───────────────────────────────────────────────────
generations  = range(len(stats.most_fit_genomes))
best_fitness = [g.fitness for g in stats.most_fit_genomes]
avg_fitness  = stats.get_fitness_mean()
std_fitness  = stats.get_fitness_stdev()

fig, ax = plt.subplots(figsize=(12, 6))

avg_arr = np.array(avg_fitness)
std_arr = np.array(std_fitness)

ax.plot(generations, best_fitness, 'r-',  linewidth=2,
        label='Best Fitness (Fixed Topology)')
ax.plot(generations, avg_arr,      'r--', linewidth=1.5,
        label='Average Fitness (Fixed Topology)', alpha=0.7)
ax.fill_between(generations,
                avg_arr - std_arr,
                avg_arr + std_arr,
                alpha=0.15, color='red')

# Draw a horizontal reference line showing baseline NEAT's best
ax.axhline(y=256.48, color='blue', linestyle='--', linewidth=2,
           label='Baseline NEAT best (256.48)')

ax.set_xlabel('Generation', fontsize=13)
ax.set_ylabel('Fitness (Total Reward)', fontsize=13)
ax.set_title('Ablation Study: Fixed Topology vs Full NEAT', fontsize=15)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

save_path = os.path.join(RESULTS_DIR, "ablation_fitness.png")
plt.tight_layout()
plt.savefig(save_path, dpi=150)
plt.close()
print(f"✓ Plot saved to {save_path}")
