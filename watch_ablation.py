import pickle
import numpy as np
import neat
import gymnasium as gym
import os

CONFIG_PATH = os.path.expanduser(
    "~/Desktop/neat_bipedal_project/config-neat-ablation.txt"
)
WINNER_PATH = os.path.expanduser(
    "~/Desktop/neat_bipedal_project/results_ablation/winner_ablation.pkl"
)

config = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    CONFIG_PATH
)

with open(WINNER_PATH, "rb") as f:
    genome = pickle.load(f)

print(f"Ablation winner stats:")
print(f"  Fitness:     {genome.fitness:.4f}")
print(f"  Nodes:       {len(genome.nodes)}")
print(f"  Connections: {len(genome.connections)}")

net = neat.nn.FeedForwardNetwork.create(genome, config)
env = gym.make("BipedalWalker-v3", render_mode="human")
obs, _ = env.reset()

total_reward = 0.0
step = 0
print("\nWatching ablation winner... (Ctrl+C to stop)")

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
