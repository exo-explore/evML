import numpy as np
import mdptoolbox

# Defining Parameters
p = 0.05        # Probability of triggering the challenge mechanism
q_d = 1         # Probability of true positives
q_h = 0         # Probability of false positives
R = 0.5         # Reward for completing the computation
C = 0.45        # Cost for completing the computation
C_1 = 0.45      # Cost for just decrypting the data
discount = 0.96 # Discount factor
K = 1000        # Cost of breaking the TEE (Trusted Execution Environment)
S = 100         # Cost of replacing the device 
W = 1           # Reward of knowing private data
U = 1           # Reward of altering the data

# States are enumerated as follows:
# Type A: 0
# Type B1: 1
# Type B2: 2
# Restart: 3

# Defining the Transition Model
# Dimensions: (number of actions, number of states, number of next states)
transition_model = np.zeros((3, 4, 4))

# Action 1 (a_A)
transition_model[0, :, :] = np.array([
    [1 - p*q_h, 0, 0, p*q_h],  # From state 0 to state 0
    [0, 1, 0, 0],  # From state 1 to state 1
    [0, 0, 1, 0],  # From state 2 to state 2
    [1 - p*q_h, 0, 0, p*q_h]   # From state 3 to state 0 (restart)
])

# Action 2 (a_B1)
transition_model[1, :, :] = np.array([
    [0, 1 - p*q_h, 0, p*q_h],  # From state 0 to state 1
    [0, 1 - p*q_h, 0, p*q_h],  # From state 1 to state 1
    [0, 1 - p*q_h, 0, p*q_h],  # From state 2 to state 1
    [0, 0, 0, 1]   # From state 3 to state 3 (restart)
])

# Action 3 (a_B2)
transition_model[2, :, :] = np.array([
    [0, 0, 1 - p*q_d, p*q_d],  # From state 0 to state 2 or 3
    [0, 0, 1 - p*q_d, p*q_d],  # From state 1 to state 2 or 3
    [0, 0, 1 - p*q_d, p*q_d],  # From state 2 to state 2 or 3
    [0, 0, 0, 1]           # From state 3 to state 3 (restart)
])

# Defining the Reward Model
# Dimensions: (number of actions, number of states, number of next states)
reward_model = np.zeros((3, 4, 4))

# Action 1 (a_A)
reward_model[0, :, :] = np.array([
    [R - C, 0, 0, -C],  # Reward
    [0, 0, 0, 0],      # No reward for transitioning from state 1
    [0, 0, 0, 0],      # No reward for transitioning from state 2
    [-S, 0, 0, -S]      # Cost for restarting from state 3
])

# Action 2 (a_B1)
reward_model[1, :, :] = np.array([
    [0, -K + R - C + W, 0, -K - C + W], 
    [0, R - C + W, 0, - C + W],      
    [0, R - C + W, 0, - C + W],      
    [0, 0, 0, 0]               
])

# Action 3 (a_B2)
reward_model[2, :, :] = np.array([
    [0, 0, -K + R - C_1 + W + U, -K - C_1 + W], 
    [0, 0, R - C_1 + W + U, -C_1 + W],          
    [0, 0, R - C_1 + W + U, -C_1 + W],          
    [0, 0, 0, 0]                                
])

initial_policy = np.zeros(4, dtype=int)  

pi = mdptoolbox.mdp.PolicyIteration(transition_model, reward_model, discount, policy0=initial_policy, max_iter=1000000) 
pi.run()

# Outputting the optimal policy and value function
print("The Policy:", pi.policy)             # Optimal action for each state
print("The value funciton is:", pi.V)       # Value function for each state
