import pickle, numpy as np, neat, gymnasium as gym, os

BASE_DIR = os.path.expanduser("~/Desktop/neat_bipedal_project")

config = neat.Config(
    neat.DefaultGenome, neat.DefaultReproduction,
    neat.DefaultSpeciesSet, neat.DefaultStagnation,
    os.path.join(BASE_DIR, "config-neat.txt")
)

ckpt_dir = os.path.join(BASE_DIR, "checkpoints_curriculum")
files = sorted(
    [f for f in os.listdir(ckpt_dir)
     if f.startswith("curriculum-checkpoint-")],
    key=lambda x: int(x.split("-")[-1])
)
best, best_g = float("-inf"), None
print("Scanning curriculum checkpoints...")
for f in files:
    pop = neat.Checkpointer.restore_checkpoint(os.path.join(ckpt_dir, f))
    for gid, g in pop.population.items():
        if g.fitness and g.fitness > best:
            best, best_g = g.fitness, g
print(f"Best curriculum genome: {best:.4f} | Nodes: {len(best_g.nodes)}")

net = neat.nn.FeedForwardNetwork.create(best_g, config)

# Ask user which environment to watch
print("\nWhich environment?")
print("  1 = Hardcore (what it was trained on)")
print("  2 = Flat terrain (to see if walking skill survived)")
choice = input("Enter 1 or 2: ").strip()
env_name = "BipedalWalkerHardcore-v3" if choice == "1" else "BipedalWalker-v3"
print(f"\nWatching on {env_name}... (Ctrl+C to stop)")

env = gym.make(env_name, render_mode="human")
obs, _ = env.reset()

total_reward = 0.0
step = 0
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
