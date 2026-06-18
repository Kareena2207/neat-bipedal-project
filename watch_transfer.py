import pickle, numpy as np, neat, gymnasium as gym, os

BASE_DIR = os.path.expanduser("~/Desktop/neat_bipedal_project")

config = neat.Config(
    neat.DefaultGenome, neat.DefaultReproduction,
    neat.DefaultSpeciesSet, neat.DefaultStagnation,
    os.path.join(BASE_DIR, "config-neat.txt")
)

winner_path = os.path.join(
    BASE_DIR, "results_stability", "winner_stability.pkl"
)
with open(winner_path, "rb") as f:
    genome = pickle.load(f)

print(f"Stability winner on HARDCORE terrain")
print(f"Fitness on flat: {genome.fitness:.4f}")
print(f"Nodes: {len(genome.nodes)} | Connections: {len(genome.connections)}")

net = neat.nn.FeedForwardNetwork.create(genome, config)
env = gym.make("BipedalWalkerHardcore-v3", render_mode="human")
obs, _ = env.reset()

total_reward = 0.0
step = 0
print("\nWatching transfer test... (Ctrl+C to stop)")

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
