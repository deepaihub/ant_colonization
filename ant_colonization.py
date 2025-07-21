import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- Simulation Parameters ---
GRID_SIZE = 100
NUM_ANTS = 100
STEPS = 2000

# Pheromone parameters
EVAPORATION_RATE = 0.995  # Slower evaporation for better path reinforcement
PHEROMONE_DEPOSIT_AMOUNT = 5.0  # Stronger pheromone deposits
MIN_PHEROMONE = 0.1  # Minimum pheromone level for exploration

# Ant behavior parameters
PHEROMONE_INFLUENCE = 3.0  # Stronger pheromone following
DIRECTION_MOMENTUM = 0.3   # Less random wandering
EXPLORATION_RATE = 0.1     # Small chance of random exploration

# --- Ant States ---
FORAGING = 0  # Searching for food
RETURNING = 1 # Returning to the nest with food

class AntSimulation:
    """
    Improved ant colony optimization simulation with proper pheromone logic.
    """

    def __init__(self, grid_size, num_ants):
        """Initializes the simulation world."""
        self.grid_size = grid_size
        self.num_ants = num_ants

        # Initialize nest and food locations
        self.nest_pos = np.array([20, grid_size // 2])
        self.food_pos = np.array([grid_size - 20, grid_size // 2])
        
        # Single pheromone grid for food trails
        self.food_pheromone = np.zeros((grid_size, grid_size))
        
        # Initialize pheromone grid with minimum values for exploration
        # self.food_pheromone.fill(MIN_PHEROMONE)

        # Place ants at the nest initially
        self.ant_positions = np.array([self.nest_pos.copy() for _ in range(num_ants)], dtype=float)
        self.ant_states = np.full(num_ants, FORAGING)
        
        # Give each ant a random initial direction
        self.ant_directions = np.random.uniform(-np.pi, np.pi, num_ants)
        
        # Track ant paths for path quality assessment
        self.ant_paths = [[] for _ in range(num_ants)]
        self.successful_trips = 0

    def update(self):
        """
        Main update loop for the simulation.
        Moves ants, updates pheromones, and handles state changes.
        """
        # --- Move Ants ---
        for i in range(self.num_ants):
            if self.ant_states[i] == FORAGING:
                self.move_ant(i, self.food_pos, RETURNING)
            else:  # RETURNING
                self.move_ant_returning(i, self.nest_pos, FORAGING)
        
        # --- Deposit Pheromones ---
        for i in range(self.num_ants):
            x, y = self.ant_positions[i].astype(int)
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                if self.ant_states[i] == RETURNING:
                    # Only returning ants with food drop strong pheromones
                    # This creates the "food is this way" trail
                    self.food_pheromone[y, x] += PHEROMONE_DEPOSIT_AMOUNT
                else:
                    # Foraging ants drop very weak pheromones (exploration)
                    self.food_pheromone[y, x] += PHEROMONE_DEPOSIT_AMOUNT * 0.1

        # --- Evaporate Pheromones ---
        self.food_pheromone *= EVAPORATION_RATE
        # Maintain minimum pheromone level for exploration
        # self.food_pheromone = np.maximum(self.food_pheromone, MIN_PHEROMONE)

    def move_ant(self, i, target_pos, next_state):
        """
        Moves foraging ants - they follow pheromone trails to find food.
        """
        pos = self.ant_positions[i]
        
        # Check if ant has reached food
        if np.linalg.norm(pos - target_pos) < 3:
            self.ant_states[i] = next_state
            self.ant_directions[i] += np.pi + np.random.uniform(-0.5, 0.5)  # Turn around with some randomness
            self.successful_trips += 1
            return

        # Get direction choices based on pheromone sensing
        new_direction = self.choose_direction(i, self.food_pheromone)
        self.ant_directions[i] = new_direction
        
        # Move ant
        self.move_ant_step(i)

    def move_ant_returning(self, i, target_pos, next_state):
        """
        Moves returning ants - they head directly back to nest.
        """
        pos = self.ant_positions[i]
        
        # Check if ant has reached nest
        if np.linalg.norm(pos - target_pos) < 3:
            self.ant_states[i] = next_state
            self.ant_directions[i] = np.random.uniform(-np.pi, np.pi)  # Random direction to explore
            return

        # Head directly toward nest with some randomness
        direction_to_nest = np.arctan2(target_pos[1] - pos[1], target_pos[0] - pos[0])
        self.ant_directions[i] = direction_to_nest + np.random.uniform(-0.2, 0.2)
        
        # Move ant
        self.move_ant_step(i)

    def choose_direction(self, i, pheromone_grid):
        """
        Choose movement direction based on pheromone trails and exploration.
        """
        pos = self.ant_positions[i]
        current_direction = self.ant_directions[i]
        
        # Random exploration with small probability
        if np.random.random() < EXPLORATION_RATE:
            return np.random.uniform(-np.pi, np.pi)
        
        # Define sensing directions (left, forward, right)
        sense_angles = np.array([
            current_direction - 0.4,  # Left
            current_direction,        # Forward
            current_direction + 0.4   # Right
        ])
        
        sense_dist = 8.0
        pheromone_strengths = np.zeros(3)
        
        # Sample pheromone strength in each direction
        for j, angle in enumerate(sense_angles):
            sense_pos = pos + sense_dist * np.array([np.cos(angle), np.sin(angle)])
            px, py = sense_pos.astype(int)
            
            if 0 <= px < self.grid_size and 0 <= py < self.grid_size:
                pheromone_strengths[j] = pheromone_grid[py, px]
            else:
                pheromone_strengths[j] = MIN_PHEROMONE  # Wall avoidance
        
        # Add momentum to forward direction
        pheromone_strengths[1] += DIRECTION_MOMENTUM
        
        # Choose direction probabilistically based on pheromone strength
        weights = np.power(pheromone_strengths, PHEROMONE_INFLUENCE)
        
        if np.sum(weights) > 0:
            probabilities = weights / np.sum(weights)
            chosen_idx = np.random.choice(3, p=probabilities)
        else:
            chosen_idx = 1  # Go straight if no pheromones
        
        # Add small random deviation
        chosen_angle = sense_angles[chosen_idx] + np.random.uniform(-0.1, 0.1)
        return chosen_angle

    def move_ant_step(self, i):
        """
        Execute one movement step for an ant.
        """
        pos = self.ant_positions[i]
        direction = self.ant_directions[i]
        
        # Calculate new position
        step_size = 1.0
        new_pos = pos + step_size * np.array([np.cos(direction), np.sin(direction)])
        
        # Boundary collision handling
        if not (0 <= new_pos[0] < self.grid_size and 0 <= new_pos[1] < self.grid_size):
            # Bounce off walls
            self.ant_directions[i] += np.pi + np.random.uniform(-0.5, 0.5)
            return
        
        self.ant_positions[i] = new_pos


# --- Visualization Setup ---
sim = AntSimulation(GRID_SIZE, NUM_ANTS)

fig, ax = plt.subplots(figsize=(12, 10))
ax.set_facecolor('black')
fig.set_facecolor('black')
ax.set_xticks([])
ax.set_yticks([])
ax.set_title("Ant Colony Optimization", color='white', fontsize=16)

# Pheromone visualization
pheromone_display = np.zeros((GRID_SIZE, GRID_SIZE, 3))
im = ax.imshow(pheromone_display, origin='lower', alpha=0.8)

# Nest and Food visualization
nest_plot, = ax.plot([], [], 'o', color='cyan', markersize=20, label='Nest')
food_plot, = ax.plot([], [], 'o', color='lime', markersize=20, label='Food')

# Ants visualization
ant_foraging_plot, = ax.plot([], [], '.', color='white', markersize=4, alpha=0.8)
ant_returning_plot, = ax.plot([], [], '.', color='gold', markersize=6, alpha=0.9)

# Add legend
ax.legend(loc='upper right', facecolor='black', edgecolor='white', labelcolor='white')

# Statistics text
stats_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, color='white', 
                    verticalalignment='top', fontsize=10, family='monospace')

def animate(frame):
    """Animation function called for each frame."""
    sim.update()
    
    # Update pheromone display
    # Normalize for better visualization
    pheromone_max = sim.food_pheromone.max()
    if pheromone_max > MIN_PHEROMONE:
        pheromone_norm = (sim.food_pheromone - MIN_PHEROMONE) / (pheromone_max - MIN_PHEROMONE)
    else:
        pheromone_norm = sim.food_pheromone * 0
    
    # Clear display
    pheromone_display[:,:,:] = 0
    
    # Show pheromone trails in bright green
    pheromone_display[:,:,1] = np.clip(pheromone_norm * 2, 0, 1)  # Green channel
    
    im.set_data(pheromone_display)
    
    # Update nest and food positions
    nest_plot.set_data([sim.nest_pos[0]], [sim.nest_pos[1]])
    food_plot.set_data([sim.food_pos[0]], [sim.food_pos[1]])
    
    # Update ant positions
    foraging_ants = sim.ant_positions[sim.ant_states == FORAGING]
    returning_ants = sim.ant_positions[sim.ant_states == RETURNING]
    
    if len(foraging_ants) > 0:
        ant_foraging_plot.set_data(foraging_ants[:, 0], foraging_ants[:, 1])
    else:
        ant_foraging_plot.set_data([], [])

    if len(returning_ants) > 0:
        ant_returning_plot.set_data(returning_ants[:, 0], returning_ants[:, 1])
    else:
        ant_returning_plot.set_data([], [])
    
    # Update statistics
    foraging_count = np.sum(sim.ant_states == FORAGING)
    returning_count = np.sum(sim.ant_states == RETURNING)
    stats_text.set_text(f'Frame: {frame:4d}\nForaging: {foraging_count:2d}\nReturning: {returning_count:2d}\nTrips: {sim.successful_trips:4d}')

    return im, nest_plot, food_plot, ant_foraging_plot, ant_returning_plot, stats_text

# --- Run Simulation ---
ani = animation.FuncAnimation(fig, animate, frames=STEPS, interval=50, blit=True)
ani.save('ant_colonization.gif', writer='pillow', fps=30)
plt.tight_layout()
plt.show()