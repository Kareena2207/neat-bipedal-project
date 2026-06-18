import pickle
import numpy as np
import neat
import gymnasium as gym
import argparse
import os

BASE_DIR = os.path.expanduser("~/Desktop/neat_bipedal_project")

parser = argparse.ArgumentParser()
parser.add_argument("--version", required=True,
    help="baseline, ablation, efficiency, or stability")
args = parser.parse_args()

# Pick correct config and winner path
if args.version == "ablation":
    config_path = os.path.join(BASE_DIR, "config-neat-ablation.txt")
    winner_path = os.path.join(BASE_DIR,
                    "results_ablation", "winner_ablation.pkl")
else:
    config_path = os.path.join(BASE_DIR, "config-neat.txt")
    winner_path = os.path.join(BASE_DIR,
                    f"results_{args.version}",
                    f"winner_{args.version}.pkl") \
                  if args.version != "baseline" else \
                  os.path.join(BASE_DIR, "results", "winner_genome.pkl")

config = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    config_path
)

with open(winner_path, "rb") as f:
    genome = pickle.load(f)

print(f"\n{args.version.upper()} winner:")
print(f"  Fitness:     {genome.fitness:.4f}")
print(f"  Nodes:       {len(genome.nodes)}")
print(f"  Connections: {len(genome.connections)}")

net = neat.nn.FeedForwardNetwork.create(genome, config)
env = gym.make("BipedalWalker-v3", render_mode="human")
obs, _ = env.reset()

total_reward = 0.0
step = 0
print(f"\nWatching {args.version} walker... (Ctrl+C to stop)\n")

try:
    while True:
        actions = np.clip(net.activate(obs), -1.0, 1.0)
        obs, reward, terminated, truncated, _ = env.step(actions)
        total_reward += reward
        step += 1
        if terminated or truncated:
            print(f"Episode ended | Steps: {step} | "
                  f"Reward: {total_reward:.2f}")
            obs, _ = env.reset()
            total_reward = 0.0
            step = 0
except KeyboardInterrupt:
    print("\nStopped.")
finally:
    env.close()
