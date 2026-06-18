"""
================================================================================
NEAT (NeuroEvolution of Augmenting Topologies) — BipedalWalker-v3
================================================================================

WHAT IS NEAT?
-------------
NEAT is an algorithm that evolves artificial neural networks using a genetic
algorithm. Unlike normal neural network training (backpropagation), NEAT:

  1. Starts with MINIMAL networks (few/no hidden nodes)
  2. Gradually GROWS complexity via mutation (adding nodes, adding connections)
  3. Evolves network WEIGHTS at the same time as TOPOLOGY (structure)
  4. Uses SPECIATION to protect new structural innovations from dying out too
     early — giving them time to prove their worth before competing with
     older, more optimized structures.

WHY BIPEDALWALKER?
------------------
BipedalWalker-v3 (from OpenAI Gymnasium) places a 2-legged robot in a
2D physics world. The robot has:
  - 24 sensor inputs (lidar, joint angles, velocities, etc.)
  - 4 motor outputs (hip/knee torques for each leg)
  - A fitness signal: +reward for moving forward, -penalty for falling

This makes it a perfect NEAT demo because:
  - The solution requires temporal coordination (legs must alternate)
  - Simple topologies fail, so we can WATCH complexity grow over generations
  - The rendered result is visually dramatic and intuitive

FILE STRUCTURE:
  neat_bipedal.py     ← This file (main logic)
  config-neat.txt     ← NEAT hyperparameter configuration
  checkpoints/        ← Saved population states every N generations
  results/            ← Plots and saved winner genome

HOW TO RUN:
  python neat_bipedal.py --mode train       # Train from scratch
  python neat_bipedal.py --mode watch       # Watch saved winner
  python neat_bipedal.py --mode visualize   # Generate all plots
  python neat_bipedal.py --mode checkpoint  # Resume from checkpoint

DEPENDENCIES:
  pip install neat-python gymnasium[box2d] pygame matplotlib networkx
================================================================================
"""

import os
import pickle
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving plots
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import neat
import gymnasium as gym


# ================================================================================
# SECTION 1: CONSTANTS & PATHS
# ================================================================================
# Centralizing all file paths here makes it easy to change them in one place.

CONFIG_PATH      = os.path.join(os.path.dirname(__file__), "config-neat.txt")
CHECKPOINT_DIR   = os.path.join(os.path.dirname(__file__), "checkpoints")
RESULTS_DIR      = os.path.join(os.path.dirname(__file__), "results")
WINNER_PATH      = os.path.join(RESULTS_DIR, "winner_genome.pkl")

# Create output directories if they don't exist yet
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ================================================================================
# SECTION 2: ENVIRONMENT CONSTANTS
# ================================================================================
# BipedalWalker-v3 has exactly 24 inputs and 4 outputs.
# These MUST match num_inputs and num_outputs in config-neat.txt.
#
# The 24 inputs are:
#   [0]     Hull angle (how tilted the body is)
#   [1]     Hull angular velocity
#   [2]     Horizontal velocity (vx)
#   [3]     Vertical velocity (vy)
#   [4]     Hip joint 1 angle
#   [5]     Hip joint 1 speed
#   [6]     Knee joint 1 angle
#   [7]     Knee joint 1 speed
#   [8]     Leg 1 ground contact (binary: 0 or 1)
#   [9]     Hip joint 2 angle
#   [10]    Hip joint 2 speed
#   [11]    Knee joint 2 angle
#   [12]    Knee joint 2 speed
#   [13]    Leg 2 ground contact (binary: 0 or 1)
#   [14-23] 10 lidar rangefinder readings (terrain ahead)
#
# The 4 outputs map to motor torques:
#   [0]  Hip 1 torque  (range: -1 to +1)
#   [1]  Knee 1 torque
#   [2]  Hip 2 torque
#   [3]  Knee 2 torque

NUM_INPUTS  = 24
NUM_OUTPUTS = 4

# How many physics steps per evaluation.
# 1600 steps ≈ enough time for a good walker to cross significant terrain.
MAX_STEPS_PER_EVAL = 1600

# If the robot's reward drops below this threshold in a single step,
# it has likely fallen — we terminate early to save computation time.
EARLY_TERMINATION_REWARD = -100.0

# How many times we run each genome to average out environmental randomness.
# BipedalWalker has some terrain randomness, so 1 run can be noisy.
# Set to 1 for speed during development, 3 for more reliable fitness.
NUM_EVAL_RUNS = 1


# ================================================================================
# SECTION 3: FITNESS EVALUATION
# ================================================================================
# This is THE most important function in the whole project.
# NEAT doesn't know anything about BipedalWalker — it just asks:
# "How good is this network?" — and we answer with a single number (fitness).
#
# NEAT will try to MAXIMIZE this number over generations.
# The higher the fitness, the more likely this genome survives and reproduces.

def evaluate_genome(genome, config):
    """
    Evaluate a single genome (neural network) in BipedalWalker.

    HOW IT WORKS:
      1. Build a neural network from the genome's node/connection genes
      2. Run the walker in the physics simulation for MAX_STEPS_PER_EVAL steps
      3. At each step: feed observations → network → get actions → step env
      4. Accumulate total reward as fitness

    INPUTS:
      genome  — a neat.DefaultGenome object containing node genes,
                connection genes, and their weights/biases
      config  — the neat.Config object (needed to build the network)

    OUTPUT:
      float — total accumulated reward (fitness score)
    """

    # neat.nn.FeedForwardNetwork.create() reads the genome's genes and
    # assembles them into an actual callable neural network.
    # This network has .activate(inputs) → list of output values
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    # We may run multiple episodes and average — reduces noise from
    # random terrain generation
    total_fitness = 0.0

    for run in range(NUM_EVAL_RUNS):
        # Create the environment WITHOUT rendering (faster for training)
        env = gym.make("BipedalWalker-v3", render_mode=None)

        # Reset returns: observation (24 floats), info (dict we ignore)
        observation, _ = env.reset()

        episode_reward = 0.0

        for step in range(MAX_STEPS_PER_EVAL):
            # ── FORWARD PASS ──────────────────────────────────────────────
            # Feed the 24 sensor values into the network.
            # net.activate() propagates them through all layers
            # and returns 4 output values (one per motor).
            raw_actions = net.activate(observation)

            # The network outputs are in [-1, 1] range because we use
            # tanh activation. BipedalWalker expects actions in [-1, 1],
            # so we can pass them directly. We use np.clip just to be safe.
            actions = np.clip(raw_actions, -1.0, 1.0)

            # ── ENVIRONMENT STEP ──────────────────────────────────────────
            # Step the physics simulation one frame forward.
            # Returns:
            #   observation — new sensor readings (24 floats)
            #   reward      — reward this step (typically small positive
            #                 for forward motion, -100 for falling)
            #   terminated  — True if episode ended naturally (e.g. fell)
            #   truncated   — True if we hit the time limit
            #   info        — extra info dict (we ignore)
            observation, reward, terminated, truncated, info = env.step(actions)

            episode_reward += reward

            # Early termination: if the robot fell (-100 penalty),
            # stop wasting computation time on this genome
            if terminated or truncated:
                break

        env.close()
        total_fitness += episode_reward

    # Return the AVERAGE fitness across all evaluation runs
    return total_fitness / NUM_EVAL_RUNS


def evaluate_genomes(genomes, config):
    """
    Called by NEAT once per generation to evaluate ALL genomes.

    NEAT passes a list of (genome_id, genome) tuples.
    We MUST set genome.fitness for every genome — NEAT uses this
    to decide which genomes survive, reproduce, and which die.

    This function is also where we could implement parallel evaluation
    (using multiprocessing) to speed up training significantly.
    """
    total = len(genomes)
    for i, (genome_id, genome) in enumerate(genomes):
        genome.fitness = evaluate_genome(genome, config)
        # Print progress so you can see training is working
        if (i + 1) % 10 == 0 or (i + 1) == total:
            print(f"  Evaluated {i+1}/{total} genomes | "
                  f"Latest fitness: {genome.fitness:.2f}")


# ================================================================================
# SECTION 4: TRAINING LOOP
# ================================================================================
# This is the main NEAT loop. It:
#   1. Loads the config file
#   2. Creates the initial population (all minimal networks)
#   3. Runs generation after generation, calling evaluate_genomes each time
#   4. Reports statistics and saves checkpoints
#   5. Returns the best genome found

def train(num_generations=200, checkpoint_interval=10):
    """
    Run the NEAT evolutionary loop.

    WHAT HAPPENS EACH GENERATION:
      1. evaluate_genomes() scores every genome in the population
      2. NEAT groups genomes into species by structural similarity
         (using the compatibility distance formula)
      3. Within each species, the top performers survive (elitism)
      4. New genomes are created by:
           a. Mutation only (small structural or weight changes)
           b. Crossover between two parents from the same species
      5. Stagnant species (no improvement for N generations) are culled
      6. Repeat

    INPUTS:
      num_generations     — how many generations to run
      checkpoint_interval — save population state every N generations
                            (allows resuming from checkpoint if crashed)
    """

    # ── LOAD CONFIG ───────────────────────────────────────────────────────────
    # neat.Config reads config-neat.txt and sets up all NEAT parameters.
    # The four classes passed here tell NEAT which implementations to use
    # for genome, reproduction, speciation, and stagnation logic.
    config = neat.Config(
        neat.DefaultGenome,        # How genomes are structured & mutated
        neat.DefaultReproduction,  # How new genomes are created
        neat.DefaultSpeciesSet,    # How species are maintained
        neat.DefaultStagnation,    # When to remove stagnant species
        CONFIG_PATH
    )

    # ── CREATE POPULATION ─────────────────────────────────────────────────────
    # The population starts with pop_size genomes, all with:
    #   - 24 input nodes (one per sensor)
    #   - 4 output nodes (one per motor)
    #   - 0 hidden nodes (minimal structure — NEAT grows them as needed)
    #   - Connections between inputs and outputs with random weights
    population = neat.Population(config)

    # ── REPORTERS ─────────────────────────────────────────────────────────────
    # Reporters are objects that NEAT calls at key moments to report progress.

    # StdOutReporter prints generation stats to the console.
    # True = show species-level details
    population.add_reporter(neat.StdOutReporter(True))

    # StatisticsReporter collects data we'll use for plotting later.
    # It records: best genome each generation, mean fitness, stdev fitness
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    # Checkpointer saves the ENTIRE population to disk every N generations.
    # This is critical — training can take hours, and if it crashes,
    # you can resume from the last checkpoint instead of starting over.
    checkpointer = neat.Checkpointer(
        generation_interval=checkpoint_interval,
        filename_prefix=os.path.join(CHECKPOINT_DIR, "neat-checkpoint-")
    )
    population.add_reporter(checkpointer)

    # ── RUN EVOLUTION ─────────────────────────────────────────────────────────
    # population.run() is the main loop. It:
    #   - Calls evaluate_genomes(genomes, config) each generation
    #   - Calls all reporters at appropriate moments
    #   - Stops early if any genome exceeds fitness_threshold (from config)
    #   - Returns the single best genome found across ALL generations
    print("\n" + "="*60)
    print("Starting NEAT training on BipedalWalker-v3")
    print(f"Population size: {config.pop_size}")
    print(f"Max generations: {num_generations}")
    print("="*60 + "\n")

    winner = population.run(evaluate_genomes, num_generations)

    # ── SAVE WINNER ───────────────────────────────────────────────────────────
    # Pickle saves the winner genome object to disk so we can load it
    # later to watch it run, analyze it, or resume training from it.
    with open(WINNER_PATH, "wb") as f:
        pickle.dump(winner, f)
    print(f"\n✓ Winner genome saved to {WINNER_PATH}")

    # ── GENERATE PLOTS ────────────────────────────────────────────────────────
    plot_fitness_history(stats)
    plot_speciation(stats)
    plot_network_topology(winner, config)

    return winner, stats


# ================================================================================
# SECTION 5: VISUALIZATION — FITNESS HISTORY
# ================================================================================
# This plot answers: "Did NEAT actually improve over time?"
# We show both the BEST fitness and AVERAGE fitness per generation.
# A widening gap between them shows increasing population diversity.

def plot_fitness_history(stats):
    """
    Plot best and average fitness over all generations.

    READING THE PLOT:
      - Blue line (Best): the single best genome each generation
      - Red dashed (Average): mean fitness across all genomes
      - The gap between them represents population diversity
      - Upward trend confirms evolution is working
      - Plateaus suggest the algorithm is exploring structural innovations
    """
    generations  = range(len(stats.most_fit_genomes))
    best_fitness = [g.fitness for g in stats.most_fit_genomes]
    avg_fitness  = stats.get_fitness_mean()
    std_fitness  = stats.get_fitness_stdev()

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot best fitness
    ax.plot(generations, best_fitness, 'b-', linewidth=2,
            label='Best Fitness', zorder=3)

    # Plot average fitness with shaded standard deviation band
    avg_arr = np.array(avg_fitness)
    std_arr = np.array(std_fitness)
    ax.plot(generations, avg_arr, 'r--', linewidth=1.5,
            label='Average Fitness', zorder=3)
    ax.fill_between(generations,
                    avg_arr - std_arr,
                    avg_arr + std_arr,
                    alpha=0.2, color='red', label='±1 Std Dev')

    # Mark the generation where the best genome was found
    best_gen = best_fitness.index(max(best_fitness))
    ax.axvline(x=best_gen, color='green', linestyle=':', linewidth=1.5,
               label=f'Best found (gen {best_gen})')
    ax.scatter([best_gen], [max(best_fitness)], color='green',
               zorder=5, s=100)

    ax.set_xlabel('Generation', fontsize=13)
    ax.set_ylabel('Fitness (Total Reward)', fontsize=13)
    ax.set_title('NEAT Fitness Progress — BipedalWalker-v3', fontsize=15)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    save_path = os.path.join(RESULTS_DIR, "fitness_history.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✓ Fitness history plot saved to {save_path}")


# ================================================================================
# SECTION 6: VISUALIZATION — SPECIATION
# ================================================================================
# This is one of NEAT's most unique features. Species protect structural
# innovations. This plot shows how many species existed each generation
# and how the population was distributed among them.
#
# WHY THIS MATTERS:
#   - Too few species → premature convergence (everyone looks the same)
#   - Too many species → fragmentation (species too small to improve)
#   - Healthy NEAT runs maintain a moderate, dynamic number of species

def plot_speciation(stats):
    """
    Plot a stacked area chart showing species population over generations.

    Each colored band = one species.
    The height of each band = how many genomes belong to that species.
    Species appearing and disappearing shows the dynamic nature of NEAT.
    """
    # stats.get_species_sizes() returns a list of dicts:
    # [{species_id: count, ...}, {species_id: count, ...}, ...]
    # One dict per generation
    species_data = stats.get_species_sizes()
    if not species_data:
        print("No speciation data available yet.")
        return

    # Get all unique species IDs that ever appeared
    all_species = sorted(set(sid for gen in species_data for sid in gen))
    generations = range(len(species_data))

    # Build a 2D array: rows=generations, cols=species
    # Fill with 0 for generations where a species didn't exist
    size_matrix = np.zeros((len(species_data), len(all_species)))
    for gen_idx, gen_dict in enumerate(species_data):
        for sp_idx, sp_id in enumerate(all_species):
            size_matrix[gen_idx, sp_idx] = gen_dict.get(sp_id, 0)

    # Color palette — one color per species
    colors = plt.cm.tab20(np.linspace(0, 1, len(all_species)))

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.stackplot(generations,
                 size_matrix.T,   # Transpose: each row = one species
                 labels=[f"Species {s}" for s in all_species],
                 colors=colors, alpha=0.8)

    ax.set_xlabel('Generation', fontsize=13)
    ax.set_ylabel('Number of Genomes', fontsize=13)
    ax.set_title('Species Population Over Time (Speciation History)',
                 fontsize=15)

    # Only show legend if there aren't too many species (gets cluttered)
    if len(all_species) <= 15:
        ax.legend(loc='upper right', fontsize=8, ncol=2)

    ax.grid(True, alpha=0.3, axis='y')

    save_path = os.path.join(RESULTS_DIR, "speciation_history.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✓ Speciation history plot saved to {save_path}")


# ================================================================================
# SECTION 7: VISUALIZATION — NETWORK TOPOLOGY
# ================================================================================
# This shows the ACTUAL neural network that NEAT evolved.
# Start with 0 hidden nodes — NEAT adds them as needed.
# By the end you'll see a real, evolved network structure.
#
# WHY THIS IS IMPRESSIVE:
#   - No human designed this network
#   - No backpropagation was used
#   - NEAT discovered the topology by evolution

def plot_network_topology(genome, config, title="Winner Network Topology"):
    """
    Draw the evolved neural network as a directed graph.

    Layout:
      - Input nodes (sensors) on the LEFT
      - Output nodes (motors) on the RIGHT
      - Hidden nodes in the MIDDLE
      - Edge color = weight (red=negative, green=positive)
      - Edge thickness = |weight| magnitude
    """
    G = nx.DiGraph()

    # ── IDENTIFY NODE TYPES ───────────────────────────────────────────────────
    # neat.Config stores input/output key ranges
    input_keys  = config.genome_config.input_keys   # e.g., [-1, -2, ..., -24]
    output_keys = config.genome_config.output_keys  # e.g., [0, 1, 2, 3]
    hidden_keys = [k for k in genome.nodes.keys()
                   if k not in input_keys and k not in output_keys]

    all_keys = list(input_keys) + list(output_keys) + hidden_keys

    # ── BUILD GRAPH ───────────────────────────────────────────────────────────
    for key in all_keys:
        G.add_node(key)

    edge_weights = []
    for conn_key, conn in genome.connections.items():
        if conn.enabled:  # Only draw active connections (not disabled ones)
            G.add_edge(conn_key[0], conn_key[1])
            edge_weights.append(conn.weight)

    # ── LAYOUT: manual layered positioning ───────────────────────────────────
    pos = {}
    # Input nodes: evenly spaced vertically on the left
    n_inputs = len(input_keys)
    for i, k in enumerate(input_keys):
        pos[k] = (-2, i - n_inputs / 2)

    # Output nodes: evenly spaced vertically on the right
    n_outputs = len(output_keys)
    for i, k in enumerate(output_keys):
        pos[k] = (2, i - n_outputs / 2)

    # Hidden nodes: spread in the middle using spring layout if any exist
    if hidden_keys:
        sub_G = G.subgraph(hidden_keys)
        spring_pos = nx.spring_layout(sub_G, seed=42)
        for k, p in spring_pos.items():
            pos[k] = (p[0] * 0.8, p[1] * 4)  # Scale to fit between layers

    # ── DRAW ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(16, 10))

    # Node colors by type
    node_colors = []
    for k in G.nodes():
        if k in input_keys:
            node_colors.append('#4CAF50')   # Green = input
        elif k in output_keys:
            node_colors.append('#2196F3')   # Blue = output
        else:
            node_colors.append('#FF9800')   # Orange = hidden

    # Edge colors by weight sign & magnitude
    if edge_weights:
        edge_colors = ['#D32F2F' if w < 0 else '#388E3C'
                       for w in edge_weights]
        edge_widths = [min(abs(w) * 0.5 + 0.3, 3.0) for w in edge_weights]
    else:
        edge_colors = []
        edge_widths = []

    nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                           node_size=300, ax=ax, alpha=0.9)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=6, font_color='white')

    if G.edges():
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors,
                               width=edge_widths, ax=ax,
                               arrows=True, arrowsize=10,
                               connectionstyle='arc3,rad=0.1')

    # Legend
    legend_elements = [
        mpatches.Patch(color='#4CAF50', label=f'Input nodes ({n_inputs})'),
        mpatches.Patch(color='#2196F3', label=f'Output nodes ({n_outputs})'),
        mpatches.Patch(color='#FF9800', label=f'Hidden nodes ({len(hidden_keys)})'),
        mpatches.Patch(color='#D32F2F', label='Negative weight'),
        mpatches.Patch(color='#388E3C', label='Positive weight'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

    ax.set_title(f'{title}\n'
                 f'Nodes: {len(G.nodes())} | '
                 f'Connections: {len(G.edges())} | '
                 f'Hidden: {len(hidden_keys)}',
                 fontsize=13)
    ax.axis('off')

    save_path = os.path.join(RESULTS_DIR, "network_topology.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Network topology plot saved to {save_path}")


# ================================================================================
# SECTION 8: COMPLEXITY GROWTH PLOT
# ================================================================================
# One of NEAT's core claims is that it starts simple and grows complexity
# only when needed. This plot directly demonstrates that claim.

def plot_complexity_growth(stats):
    """
    Plot how the average number of nodes and connections grows over generations.

    WHAT TO EXPECT:
      - Early generations: few nodes, few connections (minimal networks)
      - Later generations: more nodes/connections as evolution adds complexity
      - If complexity stops growing but fitness plateaus: algorithm is stuck
    """
    generations      = range(len(stats.most_fit_genomes))
    avg_nodes        = [np.mean([len(g.nodes)       for g in gen_genomes])
                        for gen_genomes in stats.generation_statistics]
    avg_connections  = [np.mean([len(g.connections)  for g in gen_genomes])
                        for gen_genomes in stats.generation_statistics]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1.plot(generations, avg_nodes, 'purple', linewidth=2)
    ax1.set_ylabel('Avg # Hidden Nodes', fontsize=12)
    ax1.set_title('Network Complexity Growth Over Generations', fontsize=14)
    ax1.grid(True, alpha=0.3)

    ax2.plot(generations, avg_connections, 'darkorange', linewidth=2)
    ax2.set_ylabel('Avg # Connections', fontsize=12)
    ax2.set_xlabel('Generation', fontsize=12)
    ax2.grid(True, alpha=0.3)

    save_path = os.path.join(RESULTS_DIR, "complexity_growth.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✓ Complexity growth plot saved to {save_path}")


# ================================================================================
# SECTION 9: WATCH THE WINNER
# ================================================================================
# After training, load the best genome and render it visually.
# This is the "wow factor" of your project demo.

def watch_winner(genome=None, config=None):
    """
    Load and visually render the best evolved walker.

    If genome/config not passed, loads from saved files.
    Runs the walker in a pygame window with real-time rendering.
    """
    # Load from disk if not passed directly
    if config is None:
        config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            CONFIG_PATH
        )

    if genome is None:
        if not os.path.exists(WINNER_PATH):
            print(f"No winner found at {WINNER_PATH}. Run training first.")
            return
        with open(WINNER_PATH, "rb") as f:
            genome = pickle.load(f)

    print(f"\nWinner genome stats:")
    print(f"  Fitness:     {genome.fitness:.2f}")
    print(f"  Nodes:       {len(genome.nodes)}")
    print(f"  Connections: {len(genome.connections)}")

    net = neat.nn.FeedForwardNetwork.create(genome, config)

    # render_mode="human" opens a pygame window
    env = gym.make("BipedalWalker-v3", render_mode="human")
    observation, _ = env.reset()

    total_reward = 0.0
    step = 0

    print("\nWatching winner walk... (close the window to stop)")
    try:
        while True:
            raw_actions = net.activate(observation)
            actions = np.clip(raw_actions, -1.0, 1.0)
            observation, reward, terminated, truncated, _ = env.step(actions)
            total_reward += reward
            step += 1

            if terminated or truncated:
                print(f"Episode ended at step {step} | "
                      f"Total reward: {total_reward:.2f}")
                # Reset for another run
                observation, _ = env.reset()
                total_reward = 0.0
                step = 0

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        env.close()


# ================================================================================
# SECTION 10: RESUME FROM CHECKPOINT
# ================================================================================
# If training crashes or you want to continue from a saved state,
# this function finds the latest checkpoint and resumes from there.

def resume_from_checkpoint(num_generations=200):
    """
    Find the most recent checkpoint and resume training from it.

    Checkpoints are saved as:
      checkpoints/neat-checkpoint-{generation}

    This function finds the one with the highest generation number
    and resumes evolution from that population state.
    """
    # Find all checkpoint files
    checkpoint_files = [
        f for f in os.listdir(CHECKPOINT_DIR)
        if f.startswith("neat-checkpoint-")
    ]

    if not checkpoint_files:
        print("No checkpoints found. Starting from scratch.")
        return train(num_generations)

    # Sort by generation number (the number at the end of the filename)
    checkpoint_files.sort(key=lambda x: int(x.split("-")[-1]))
    latest = os.path.join(CHECKPOINT_DIR, checkpoint_files[-1])
    print(f"Resuming from checkpoint: {latest}")

    # neat.Checkpointer.restore_checkpoint() loads the full population
    population = neat.Checkpointer.restore_checkpoint(latest)

    # Re-add reporters (they don't get saved in checkpoints)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    population.add_reporter(
        neat.Checkpointer(
            generation_interval=10,
            filename_prefix=os.path.join(CHECKPOINT_DIR, "neat-checkpoint-")
        )
    )

    winner = population.run(evaluate_genomes, num_generations)

    with open(WINNER_PATH, "wb") as f:
        pickle.dump(winner, f)

    plot_fitness_history(stats)
    plot_speciation(stats)
    plot_network_topology(winner, population.config)

    return winner, stats


# ================================================================================
# SECTION 11: MAIN ENTRY POINT
# ================================================================================

def main():
    parser = argparse.ArgumentParser(
        description="NEAT BipedalWalker — Bio-inspired AI Project"
    )
    parser.add_argument(
        "--mode",
        choices=["train", "watch", "visualize", "checkpoint"],
        default="train",
        help=(
            "train      = Run evolution from scratch\n"
            "watch      = Watch saved winner walk\n"
            "visualize  = Generate all plots from saved stats\n"
            "checkpoint = Resume training from latest checkpoint"
        )
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=200,
        help="Number of generations to run (default: 200)"
    )
    args = parser.parse_args()

    if args.mode == "train":
        winner, stats = train(num_generations=args.generations)
        watch_winner(winner)

    elif args.mode == "watch":
        watch_winner()

    elif args.mode == "checkpoint":
        winner, stats = resume_from_checkpoint(args.generations)
        watch_winner(winner)

    elif args.mode == "visualize":
        # Load saved winner and generate all plots
        config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            CONFIG_PATH
        )
        if os.path.exists(WINNER_PATH):
            with open(WINNER_PATH, "rb") as f:
                winner = pickle.load(f)
            plot_network_topology(winner, config)
            print("Load stats object separately if available.")
        else:
            print("No winner genome found. Run training first.")


if __name__ == "__main__":
    main()
