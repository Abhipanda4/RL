import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

MOVE_UP = 0
MOVE_RIGHT = 1
MOVE_DOWN = 2
MOVE_LEFT = 3

def initialize_global_parameters(alpha=0.5, epsilon=0.1, gamma=1):
	global ALPHA
	ALPHA = alpha
	global EPSILON
	EPSILON = epsilon
	global GAMMA
	GAMMA = gamma


class GridWorld(object):
	def __init__(self, gridSize=(4, 12), start=(3, 0), goal=(3, 11), cliff=None):
		self.gridSize = gridSize
		self.start = start
		self.goal = goal
		self.cliff = cliff

	def get_next_state(self, curr_state, action):
		r, c = curr_state
		if action == MOVE_UP:
			new_state_r = max(0, r - 1)
			new_state_c = c
		elif action == MOVE_RIGHT:
			new_state_r = r
			new_state_c = min(self.gridSize[1] - 1, c + 1)
		elif action == MOVE_DOWN:
			new_state_r = min(self.gridSize[0] - 1, r + 1)
			new_state_c = c
		else:
			new_state_r = r
			new_state_c = max(0, c - 1)
		next_state = (new_state_r, new_state_c)
		# decide rewards
		if next_state in self.cliff:
			next_state = self.start
			R = -100
		elif next_state == self.goal:
			R = 0
		else:
			R = -1
		return R, next_state

class Agent(object):
	def __init__(self, env):
		self.nS = env.gridSize
		self.nA = 4
		self.Q = np.zeros((self.nS[0], self.nS[1], self.nA))
		self.curr_state = env.start

	def select_next_action(self):
		# using epsilon-greedy policy
		Q_vals = self.Q[self.curr_state[0], self.curr_state[1], :]
		if not np.random.binomial(1, EPSILON) == 1:
			# greedy action
			# selct action with maximum Q value
			max_actions = []
			for i in range(self.nA):
				if Q_vals[i] == max(Q_vals):
					max_actions.append(i)
			next_action = np.random.choice(max_actions)
		else:
			# choose randomly
			next_action = np.random.choice(range(self.nA))

		return next_action

	def reset(self, env):
		self.curr_state = env.start

def SARSA(agent, env):
	reward = 0
	while not agent.curr_state == env.goal:
		S = agent.curr_state
		A = agent.select_next_action()
		R, S_prime = env.get_next_state(agent.curr_state, A)
		# update Q vals
		# Q(S, A) = Q(S, A) + alpha(R + gamma * Q(S', A') - Q(S, A))
		# where A' is chosen epsilon-greedily.
		agent.curr_state = S_prime
		A_prime = agent.select_next_action()
		Q_S_A = agent.Q[S[0], S[1], A]
		Q_Spr_Apr = agent.Q[S_prime[0], S_prime[1], A_prime]
		agent.Q[S[0], S[1], A] = Q_S_A + ALPHA * \
									  (R + GAMMA * Q_Spr_Apr - Q_S_A)
		reward += R
	return reward

def Q_learning(agent, env):
	reward = 0
	while not agent.curr_state == env.goal:
		S = agent.curr_state
		A = agent.select_next_action()
		R, S_prime = env.get_next_state(agent.curr_state, A)
		# update Q vals
		# Q(S, A) = Q(S, A) + alpha(R + gamma * max_a Q(S', a) - Q(S, A))
		# where a is chosen so that Q(S', a) is maximum(greedy action)
		agent.curr_state = S_prime
		Q_S_A = agent.Q[S[0], S[1], A]
		Q_Spr_Apr = np.max(agent.Q[S_prime[0], S_prime[1], :])
		agent.Q[S[0], S[1], A] = Q_S_A + ALPHA * \
									  (R + GAMMA * Q_Spr_Apr - Q_S_A)
		reward += R
	return reward

def expected_SARSA(agent, env):
	reward = 0
	while not agent.curr_state == env.goal:
		S = agent.curr_state
		A = agent.select_next_action()
		R, S_prime = env.get_next_state(agent.curr_state, A)
		# update Q vals
		# Q(S, A) = Q(S, A) + alpha(R + gamma * E[Q(S', a)] - Q(S, A))
		# where E[Q(S', a)] = sum[(prob of taking action a) * Q(S', a)]
		agent.curr_state = S_prime
		Q_S_A = agent.Q[S[0], S[1], A]
		expected_Q = EPSILON / agent.nA * \
					 np.sum(agent.Q[S_prime[0], S_prime[1], :])
		expected_Q += (1 - EPSILON) * \
					  np.max(agent.Q[S_prime[0], S_prime[1], :])
		agent.Q[S[0], S[1], A] = Q_S_A + ALPHA * \
								 (R + GAMMA * expected_Q - Q_S_A)
		reward += R
	return reward

def n_step_SARSA(agent, env, num_steps):
	pass


num_episodes = 500
K = 10
def runner(algo_num, runs):
	initialize_global_parameters()
	cliff = [(3, i) for i in range(1, 11)]
	env = GridWorld(cliff=cliff)
	agent = Agent(env)
	cum_rewards = np.zeros(num_episodes)
	for run in tqdm(range(runs)):
		rewards = np.zeros(num_episodes)
		for e in range(num_episodes):
			if algo_num == 1:
				reward = SARSA(agent, env)
			elif algo_num == 2:
				reward = Q_learning(agent, env) 
			elif algo_num == 3:
				reward = expected_SARSA(agent, env)
			agent.reset(env)
			rewards[e] = reward
		cum_rewards += rewards
	# take average among K succesive episodes for smoothening the curve
	res = np.copy(cum_rewards)
	for i in range(K, num_episodes):
		res[i] = np.sum(cum_rewards[i-K:i+1]) / K
	return res / runs

def fig_6_5():
	sarsa_rewards = runner(1, 50)
	q_learning_rewards = runner(2, 50)
	plt.plot(sarsa_rewards, label="SARSA")
	plt.plot(q_learning_rewards, label="Q-Learning")
	plt.xlabel("Episodes")
	plt.ylabel("Rewards per episode")
	plt.legend()
	plt.show()

def fig_6_6():
	sarsa_rewards = runner(1, 50)
	q_learning_rewards = runner(2, 50)
	expected_SARSA_rewards = runner(3, 50)
	plt.plot(sarsa_rewards, label="SARSA")
	plt.plot(q_learning_rewards, label="Q-Learning")
	plt.plot(expected_SARSA_rewards, label="Expected_SARSA")
	plt.xlabel("Episodes")
	plt.ylabel("Rewards per episode")
	plt.legend()
	plt.show()

fig_6_5()