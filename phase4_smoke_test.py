import neat
import numpy as np
import gymnasium as gym
import os

# ── Create output directories ─────────────────────────────────────────────
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("results",     exist_ok=True)

# ── Fitness function (same logic as neat_bipedal.py) ─────────────────────
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

# ── Load config ───────────────────────────────────────────────────────────
config = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    "config-neat.txt"
)

# ── Create population ─────────────────────────────────────────────────────
population = neat.Population(config)

# ── Add reporters ─────────────────────────────────────────────────────────
population.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
population.add_reporter(stats)
population.add_reporter(
    neat.Checkpointer(
        generation_interval=2,
        filename_prefix="checkpoints/neat-checkpoint-"
    )
)

# ── Run 5 generations only (smoke test) ───────────────────────────────────
print("\n" + "="*60)
print("SMOKE TEST — 5 generations")
print("="*60 + "\n")

winner = population.run(evaluate_genomes, 5)

print("\n" + "="*60)
print("SMOKE TEST PASSED")
print(f"Best genome fitness: {winner.fitness:.4f}")
print(f"Best genome nodes:   {len(winner.nodes)}")
print(f"Best genome conns:   {len(winner.connections)}")
print("Check that checkpoints/ folder now has files in it.")
print("="*60)
