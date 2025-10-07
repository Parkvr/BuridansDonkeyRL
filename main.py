import numpy as np
import matplotlib.pyplot as plt
from buridans_donkey_env import BuridansDonkeyEnv
from agent import DMCRLAgent

def generate_random_weights():
    """Generates a random weight vector that sums to 1."""
    weights = np.random.rand(3)
    return weights / np.sum(weights)

if __name__ == "__main__":
    # --- Setup ---
    env = BuridansDonkeyEnv()
    agent = DMCRLAgent(env.action_space, env.observation_space)

    # --- Experiment Parameters ---
    # The paper uses 100,000 time-steps per weight. We'll use a smaller number for quick demonstration.
    TRAINING_STEPS_PER_WEIGHT = 20000
    EVALUATION_INTERVAL = 1000 # Evaluate every 1000 steps
    NUM_WEIGHT_CHANGES = 20

    all_rewards = []
    
    # --- Main Loop ---
    for i in range(NUM_WEIGHT_CHANGES):
        # 1. Obtain the current weight vector. [cite: 4, Table 1]
        current_weights = generate_random_weights()
        print(f"\n--- Weight Vector {i+1}/{NUM_WEIGHT_CHANGES} ---")
        print(f"Weights (Hunger, Stolen, Walk): {current_weights}")

        # 2. Agent selects an initial policy and prepares for learning.
        agent.new_weight_vector(current_weights)
        
        # 3. Learn the new policy through vector-based reinforcement learning. [cite: 4, Table 1]
        state, _ = env.reset()
        episode_rewards = []
        
        for step in range(TRAINING_STEPS_PER_WEIGHT):
            action = agent.choose_action(state, current_weights)
            next_state, reward_vector, _, _, _ = env.step(action)
            
            agent.learn(state, action, reward_vector, next_state, current_weights)
            state = next_state
            
            # For visualization, we calculate the weighted gain.
            weighted_reward = np.dot(reward_vector, current_weights)

            if (step + 1) % EVALUATION_INTERVAL == 0:
                # In the paper, evaluation is done with a greedy policy. For simplicity here,
                # we just plot the running average of the weighted rewards during ε-greedy learning.
                episode_rewards.append(weighted_reward)
        
        all_rewards.append(np.mean(episode_rewards))

        # 4. Agent decides whether to store the newly learned policy.
        agent.finish_learning_cycle(current_weights)
        print(f"Total policies stored: {len(agent.stored_policies)}")

    # --- Visualization ---
    plt.figure(figsize=(12, 6))
    plt.plot(all_rewards, marker='o', linestyle='-')
    plt.title("Average Weighted Reward After Each Weight Change")
    plt.xlabel("Weight Vector Index")
    plt.ylabel("Average Weighted Reward during Training")
    plt.grid(True)
    plt.show()