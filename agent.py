import numpy as np
from collections import defaultdict

class DMCRLAgent:
    """
    An agent that implements the Dynamic Multi-Criteria Reinforcement Learning
    (DMCRL) schema from the paper. It uses vectorized R-Learning.
    """
    def __init__(self, action_space, observation_space, alpha=0.1, beta=0.2, epsilon=0.1):
        self.action_space = action_space
        self.observation_space = observation_space
        self.alpha = alpha # Learning rate for average reward vector ρ
        self.beta = beta   # Learning rate for R-values
        self.epsilon = epsilon
        
        # DMCRL components
        # We store policies indirectly by storing their R-tables and ρ-vectors.
        self.stored_policies = [] # List of tuples: (R_table, rho_vector)
        self.delta = 0.01 # Threshold to add a new policy
        
        # Current policy components
        self.R = self._create_table()
        self.rho = np.zeros(3) # Average reward vector ρ

    def _create_table(self):
        # The state is a tuple, so we use a defaultdict
        return defaultdict(lambda: np.zeros((self.action_space.n, 3)))

    def choose_action(self, state, current_weights):
        """
        Chooses an action using an epsilon-greedy policy.
        The greedy action maximizes the weighted sum of the action-values.
        """
        if np.random.random() < self.epsilon:
            return self.action_space.sample()
        else:
            action_values = self.R[state]
            # The agent chooses an action that maximizes the value of w · R(s', a') [cite: 4, Eq. 6]
            weighted_q_values = np.dot(action_values, current_weights)
            return np.argmax(weighted_q_values)

    def learn(self, state, action, reward, next_state, current_weights):
        """
        Updates the agent's knowledge using vectorized R-Learning update rules.
        """
        # Select the best next action greedily based on current weights
        # a' = argmax_a' [w · R(s', a')] [cite: 4, Eq. 6]
        next_action_values = self.R[next_state]
        weighted_next_q = np.dot(next_action_values, current_weights)
        best_next_action = np.argmax(weighted_next_q)
        
        # Get the vector of R-values for the best next action
        best_next_R_vector = self.R[next_state][best_next_action]
        
        # Update R-table for the current state-action pair
        # R(s,a) ← R(s,a)(1 – β) + β(r_imm - ρ + R(s',a')) [cite: 4, Eq. 5, adapted for chosen a']
        current_R_vector = self.R[state][action]
        update_target_R = reward - self.rho + best_next_R_vector
        self.R[state][action] = (1 - self.beta) * current_R_vector + self.beta * update_target_R
        
        # Update the average reward vector ρ
        # ρ ← ρ(1 – α) + α(r_imm + R(s', a') – R(s, a)) [cite: 4, Eq. 7]
        update_target_rho = reward + best_next_R_vector - current_R_vector
        self.rho = (1 - self.alpha) * self.rho + self.alpha * update_target_rho
        
    def new_weight_vector(self, new_weights):
        """
        Handles a change in preferences (weights) according to the DMCRL algorithm schema. [cite: 4, Table 1]
        """
        initial_policy = None
        best_gain = -np.inf
        
        if not self.stored_policies:
            # If no policies are stored, start from scratch
            self.R = self._create_table()
            self.rho = np.zeros(3)
        else:
            # Step 2: Choose an appropriate policy for the new weight vector.
            # We need to pick the policy π_init that maximizes the inner product.
            # π_init = Argmax_{π∈Π} {w_new · ρ_π}
            for r_table, rho_vector in self.stored_policies:
                gain = np.dot(new_weights, rho_vector)
                if gain > best_gain:
                    best_gain = gain
                    initial_policy = (r_table, rho_vector)
            
            # Step 2a & 2b: Initialize value functions and average reward vector
            # We use deepcopy to avoid modifying the stored policy
            self.R = defaultdict(lambda: np.zeros((self.action_space.n, 3)), initial_policy[0])
            self.rho = np.copy(initial_policy[1])

        # Step 2c: The agent now learns with these initialized values.
        # This happens in the main training loop.
        
        # After learning, we decide whether to store the new policy.
        # This is handled by `finish_learning_cycle`.
        self.initial_gain_for_new_weights = best_gain

    def finish_learning_cycle(self, new_weights):
        """
        Called after a learning period to decide if the new policy should be stored.
        """
        # Step 3: If the weighted gain of the new policy improves by more
        # than δ, add π' to the set of stored policies.
        final_gain = np.dot(new_weights, self.rho)
        
        if final_gain > self.initial_gain_for_new_weights + self.delta:
            print(f"New policy stored! Gain improved from {self.initial_gain_for_new_weights:.3f} to {final_gain:.3f}")
            self.stored_policies.append((self.R.copy(), self.rho.copy()))