import pickle, numpy as np, neat, gymnasium as gym, os

BASE_DIR = os.path.expanduser("~/Desktop/neat_bipedal_project")

config = neat.Config(
    neat.DefaultGenome, neat.DefaultReproduction,
    neat.DefaultSpeciesSet, neat.DefaultStagnation,
    os.path.join(BASE_DIR, "config-neat.txt")
)

# First extract best from checkpoints
ckpt_dir = os.path.join(BASE_DIR, "checkpoints_hardcore")
files = sorted(
    [f for f in os.listdir(ckpt_dir) if f.startswith("hardcore-checkpoint-")],
    key=lambda x: int(x.split("-")[-1])
)
best, best_g = float("-inf"), None
print("Scanning hardcore checkpoints...")
for f in files:
    pop = neat.Checkpointer.restore_checkpoint(os.path.join(ckpt_dir, f))
    for gid, g in pop.population.items():
        if g.fitness and g.fitness > best:
            best, best_g = g.fitness, g
print(f"Best hardcore genome: {best:.4f} | Nodes: {len(best_g.nodes)}")

net = neat.nn.FeedForwardNetwork.create(best_g, config)
env = gym.make("BipedalWalkerHardcore-v3", render_mode="human")
obs, _ = env.reset()

total_reward = 0.0
step = 0
print("Watching hardcore walker... (Ctrl+C to stop)")
try:
    while True:
        actions = np.clip(net.activate(obs), -1.0, 1.0)
        obs, reward, terminated, truncated, _ = env.step(actions)
        total_reward += reward
        step += 1
        if terminated or truncated:
            print(f"Episode ended | Steps: {step} | Reward: {total_reward:.2f}")
            obs, _ = env.reset()
            total_reward = 0.0
            step = 0
except KeyboardInterrupt:
    print("\nStopped.")
finally:
    env.close()
