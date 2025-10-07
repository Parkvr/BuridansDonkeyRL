import gymnasium as gym
from gymnasium import spaces
import numpy as np

class BuridansDonkeyEnv(gym.Env):
    """
    A custom Gym environment for the Buridan's Donkey problem as described in
    "Dynamic Preferences in Multi-Criteria Reinforcement Learning".
    """
    metadata = {'render_modes': ['human']}

    def __init__(self):
        super(BuridansDonkeyEnv, self).__init__()

        # Define the 3x3 grid world
        self.grid_size = 3
        
        # State Representation: A tuple (s, f, t) where s is the donkey's square,
        # f is food in the two piles, and t is time since last ate.
        # s: donkey's position (0-8)
        # f: tuple for food piles (0: no food, 1: food) at diagonally opposite squares (pos 0 and 8)
        # t: time since last ate (0-9)
        self.observation_space = spaces.MultiDiscrete([
            self.grid_size * self.grid_size,  # Donkey's position
            2,                                # Food pile 1 status
            2,                                # Food pile 2 status
            10                                # Time since last ate
        ])

        # Action Space: move up, down, left, right, and stay
        self.action_space = spaces.Discrete(5)
        self._action_to_direction = {
            0: np.array([-1, 0]),  # Up
            1: np.array([1, 0]),   # Down
            2: np.array([0, -1]),  # Left
            3: np.array([0, 1]),   # Right
            4: np.array([0, 0]),   # Stay
        }

        # Environment Parameters from the paper
        self.food_positions = [0, 8] # Top-left and bottom-right corners
        self.p_stolen = 0.9
        self.n_appear = 10
        self.max_hunger_time = 9

        # Penalities for reward vector (hunger, stolen_food, walking)
        self.hunger_penalty = -1.0
        self.stolen_penalty = -0.5
        self.walking_penalty = -1.0
        
        self.time_step = 0

    def _get_obs(self):
        return (self.donkey_pos, self.food_piles[0], self.food_piles[1], self.time_since_last_ate)

    def _pos_to_coord(self, pos):
        return np.array([pos // self.grid_size, pos % self.grid_size])

    def _coord_to_pos(self, coord):
        return coord[0] * self.grid_size + coord[1]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # The animal is originally in the center square
        self.donkey_pos = self.grid_size * self.grid_size // 2
        
        # Food is present at the two diagonally opposite squares
        self.food_piles = [1, 1]
        
        self.time_since_last_ate = 0
        self.time_step = 0

        return self._get_obs(), {}

    def step(self, action):
        # Initialize reward vector for this step
        reward_vector = np.zeros(3) # (hunger, stolen_food, walking)
        
        prev_pos = self.donkey_pos
        direction = self._action_to_direction[action]

        # --- 1. Update Donkey's Position & Walking Penalty ---
        if action != 4: # If not 'stay'
            current_coord = self._pos_to_coord(self.donkey_pos)
            new_coord = np.clip(current_coord + direction, 0, self.grid_size - 1)
            self.donkey_pos = self._coord_to_pos(new_coord)
            reward_vector[2] = self.walking_penalty # Add walking penalty

        # --- 2. Handle Eating and Hunger ---
        ate_food = False
        if action == 4: # Stay
            for i, pos in enumerate(self.food_positions):
                if self.donkey_pos == pos and self.food_piles[i] == 1:
                    # If the donkey chooses to stay at a square with food, then it eats the food.
                    self.food_piles[i] = 0
                    self.time_since_last_ate = 0
                    ate_food = True
                    break # Can only eat from one pile at a time
        
        if not ate_food:
            self.time_since_last_ate += 1
        
        if self.time_since_last_ate > self.max_hunger_time:
            self.time_since_last_ate = self.max_hunger_time
            # The donkey incurs a penalty of -1 per time step till it eats.
            reward_vector[0] = self.hunger_penalty

        # --- 3. Handle Food Theft ---
        # "If the donkey moves away from the neighboring square of a food pile,
        # there is a certain probability Pstolen with which the food is stolen."
        prev_coord = self._pos_to_coord(prev_pos)
        donkey_moved = (action != 4)
        
        for i, pos in enumerate(self.food_positions):
            if self.food_piles[i] == 1:
                food_coord = self._pos_to_coord(pos)
                # Check if previous position was adjacent (including diagonals)
                is_neighbor = np.max(np.abs(prev_coord - food_coord)) <= 1
                
                if is_neighbor and donkey_moved and self.np_random.random() < self.p_stolen:
                    self.food_piles[i] = 0
                    reward_vector[1] += self.stolen_penalty # Stolen penalty is -0.5 per plate

        # --- 4. Handle Food Regeneration ---
        self.time_step += 1
        if self.time_step % self.n_appear == 0:
            # Food is regenerated once every Nappear time-steps.
            for i in range(len(self.food_piles)):
                if self.food_piles[i] == 0:
                    self.food_piles[i] = 1

        terminated = False # This environment runs indefinitely
        truncated = False
        
        return self._get_obs(), reward_vector, terminated, truncated, {}