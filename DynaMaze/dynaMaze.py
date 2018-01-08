import os
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

MOVE_UP = 0
MOVE_RIGHT = 1
MOVE_DOWN = 2
MOVE_LEFT = 3

# helper function to adjust all global variables
def initialize_global_parameters(epsilon=0.1, alpha=0.5, gamma=0.95, kappa=0):
	global EPSILON
	EPSILON = epsilon
	global ALPHA
	ALPHA = alpha
	global GAMMA
	GAMMA = gamma
	global KAPPA
	KAPPA = kappa

class DynaMaze(object):
	def __init__(self, start, goal, obstacles, gridSize=[6,9]):
		self.gridSize = gridSize
		self.obstacles = obstacles
		self.start = start
		self.goal = goal

	def get_next_state(self, curr_state, action):
		'''
		Returns the next state of an agent given the current state
		and action taken
		'''
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
		# decide rewards
		R = 0
		# check for obstacles
		# return current state if next state turns out to be an obstacle
		if (new_state_r, new_state_c) in self.obstacles:
			(new_state_r, new_state_c) = curr_state
		# check goal state
		elif (new_state_r, new_state_c) == self.goal:
			R = 1
		return (new_state_r, new_state_c), R

	def change_obstacle(self, new_obstacles=None):
		self.obstacles = new_obstacles

	def render(self, agent, trajectory, reward, stay):
		# pretty print :)
		# do not call when debugging, takes time due to sleep call
		map = np.chararray(self.gridSize)
		for i in range(self.gridSize[0]):
			for j in range(self.gridSize[1]):
				map[i][j] = "."
		for cell in trajectory:
			map[cell[0]][cell[1]] = "#"
		map[self.start[0]][self.start[1]] = "S"
		for i in range(len(self.obstacles)):
			map[self.obstacles[i][0]][self.obstacles[i][1]] = "O"
		map[self.goal[0]][self.goal[1]] = "T"
		for i in range(self.gridSize[0]):
			for j in range(self.gridSize[1]):
				print(map[i][j].decode(), "|", end=' ')
			print()
		print()
		print("Reward Obtained:", reward)
		# do not clear the screen after last iteration
		if not stay:
			time.sleep(0.01)
			os.system('cls')


class Agent(object):
	def __init__(self, env):
		self.nS = env.gridSize
		self.nA = 4
		self.Q = np.zeros((self.nS[0], self.nS[1], self.nA))
		self.curr_state = env.start

	def select_next_action(self):
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

	def erase(self):
		self.Q = np.zeros((self.nS[0], self.nS[1], self.nA))


class SimpleModel(object):
	def __init__(self, rand=np.random):
		self.experience = dict()
		self.rand = rand

	def save(self, S, A, R, S_prime):
		if S not in self.experience.keys():
			self.experience[S] = dict()
		self.experience[S][A] = (R, S_prime)

	def prev_experience(self):
		s_index = self.rand.choice(len(self.experience.keys()))
		S = list(self.experience)[s_index]
		a_index = self.rand.choice(len(self.experience[S]))
		A = list(self.experience[S])[a_index]
		R, S_prime = self.experience[S][A]
		return S, A, R, S_prime


class TimeModel(object):
	def __init__(self, rand=np.random, agent=None):
		self.rand = rand
		self.experience = dict()
		self.nA = agent.nA
		self.time = 0

	def save(self, S, A, R, S_prime):
		"""
		stores the real experience of agent along with the timestamp;
		also initializes the timestamp for all unseen (S, A) pairs as 1
		"""
		self.time += 1
		if S not in self.experience.keys():
			self.experience[S] = dict()
			for a in range(self.nA):
				self.experience[S][a] = (0, S, 1)
		self.experience[S][A] = (R, S_prime, self.time)
		
	def prev_experience(self):
		"""
		returns (state, action, reward, next_state)
		where the action may or may not have been taken in the past
		"""
		s_index = self.rand.choice(len(self.experience.keys()))
		S = list(self.experience)[s_index]
		a_index = self.rand.choice(range(4))
		A = list(self.experience[S])[a_index]
		R, S_prime, time = self.experience[S][A]
		R += KAPPA * np.sqrt(self.time - time)
		return S, A, R, S_prime


class Dyna(object):
	def __init__(self, model):
		self.trajectory = []
		self.time = 0
		self.model = model

	def dynaQ(self, agent, env, planning_step):
		steps = 0
		while not agent.curr_state == env.goal:
			A = agent.select_next_action()
			S = agent.curr_state
			Q_S_A = agent.Q[S[0], S[1], A]
			S_prime, R = env.get_next_state(S, A)
			max_Q_Sprime_a = np.max(agent.Q[S_prime[0], S_prime[1], :])
			agent.Q[S[0], S[1], A] = Q_S_A + ALPHA * \
								     (R + GAMMA * max_Q_Sprime_a - Q_S_A)
			steps += 1
			self.trajectory.append(S)
			agent.curr_state = S_prime

			# save the experience in model
			self.model.save(S, A, R, S_prime)
			# planning on simulated experience
			for _ in range(planning_step):
				S, A, R, S_prime = self.model.prev_experience()
				Q_S_A = agent.Q[S[0], S[1], A]
				max_Q_Sprime_a = max(agent.Q[S_prime[0], S_prime[1], :])
				agent.Q[S[0], S[1], A] = Q_S_A + ALPHA \
									   * (R + GAMMA * max_Q_Sprime_a - Q_S_A)
		return steps


class Parameters(object):
	def __init__(self, planning_steps=5, runs=10,
				 max_time_steps=1000, obstacle_change_time=-1):
		self.planning_steps = planning_steps
		self.runs = runs
		self.max_time_steps = max_time_steps
		self.obstacle_change_time = obstacle_change_time


def fig_8_3():
	# global parameters
	alpha = 0.1
	epsilon = 0.1
	gamma = 0.95
	kappa = 0
	initialize_global_parameters(epsilon, alpha, gamma, kappa)
	# parameters for the environment
	obstacles = \
			[
				(1, 2),
				(2, 2),
				(3, 2),
				(4, 5),
				(0, 7),
				(1, 7),
				(2, 7)
			]
	start = (2, 0)
	goal = (0, 8)
	# episode parameters
	runs = 10
	planning_steps = [0, 5, 50]
	num_episodes = 50
	# a tensor to store number of steps per episode
	# in each of the planning step
	plotting_steps = []
	# fix random seed
	rand = np.random.RandomState(0)
	for p in range(len(planning_steps)):
		avg_steps_per_run = np.zeros(num_episodes)
		for run in tqdm(range(runs)):
			np.random.seed(run)
			env = DynaMaze(start, goal, obstacles)
			agent = Agent(env)
			model = SimpleModel(rand)
			steps_per_run = []
			for i in range(num_episodes):
				dynaQ = Dyna(model)
				# run an episode and find number of steps taken
				num_steps = dynaQ.dynaQ(agent, env, planning_steps[p])
				steps_per_run.append(num_steps)
				agent.reset(env)
				# env.render(agent, dynaQ.trajectory,\
				#			 agent.steps_per_episode[i], (i+1)==num_episodes)
			avg_steps_per_run += steps_per_run
		avg_steps_per_run /= runs
		plotting_steps.append(avg_steps_per_run)
	# plot the curves
	for i in range(len(planning_steps)):
		plt.plot(plotting_steps[i][1:], label= str(planning_steps[i]) \
										 + " planning steps")
	plt.xlabel('Episodes')
	plt.ylabel('Steps per Episode')
	plt.legend()
	plt.title("Dyna-Q algorithm on maze problem")
	plt.show()

def q_vs_q_plus(env, old_obstacles, new_obstacles, params):
	# for plotting
	cum_rewards_avg = np.zeros((2, params.max_time_steps))
	for run in tqdm(range(params.runs)):
		agent = Agent(env)
		# create 2 new models in each run
		dynaQ_model = SimpleModel()
		dynaQ_plus_model = TimeModel(agent=agent)
		# for easy iteration!!
		models = [dynaQ_model, dynaQ_plus_model]
		cum_rewards = np.zeros((2, params.max_time_steps))
		for index, model in enumerate(models):
			dyna = Dyna(model)
			total_steps = 0
			prev_steps = total_steps
			rewards = 0
			# set initial obstacles
			env.change_obstacle(old_obstacles)
			while (total_steps < params.max_time_steps):
				# run an episode
				total_steps += dyna.dynaQ(agent, env, params.planning_steps)
				# at the end of each episode:
				# update reards for all intermediate steps
				# increment reward by 1
				# final step receives an extra reward for reaching goal
				cum_rewards[index][prev_steps: total_steps] = rewards
				rewards += 1
				cum_rewards[index] \
						   [min(params.max_time_steps - 1, total_steps)]\
						   = rewards
				prev_steps = total_steps
				agent.reset(env)
				if total_steps > params.obstacle_change_time:
					# change the environment
					env.change_obstacle(new_obstacles)
			# start with fresh Q values for next model
			agent.erase()
		cum_rewards_avg += cum_rewards

	# take the average over all runs
	cum_rewards_avg = cum_rewards_avg / params.runs
	# plot figure
	plt.plot(cum_rewards_avg[0, :], label="Dyna-Q Algo")
	plt.plot(cum_rewards_avg[1, :], label="Dyna-Q+ Algo")
	plt.xlabel('Time steps')
	plt.ylabel('Cumulative Reward')
	plt.legend()
	plt.title("Dyna-Q vs Dyna-Q+ Algorithms")
	plt.show()

def fig_8_5():
	# global parameters
	alpha = 0.7
	epsilon = 0.1
	gamma = 0.95
	kappa = 1e-4
	initialize_global_parameters(epsilon, alpha, gamma, kappa)
	# decide parameters of environment
	start = (5, 3)
	goal = (0, 8)
	# initial obstacles
	obstacles_1 = [(3, i) for i in range(0, 8)]
	# new obstacles to be added later
	obstacles_2 = [(3, i) for i in range(1, 9)]
	env = DynaMaze(start, goal, obstacles_1)
	# decide parameters of algorithm
	planning_steps = 5
	runs = 20
	# decide timesteps
	max_time_steps = 3000
	obstacle_change_time = 1000
	params = Parameters(planning_steps, runs,
						max_time_steps, obstacle_change_time)
	q_vs_q_plus(env, obstacles_1, obstacles_2, params)

def fig_8_6():
	# global parameters
	alpha = 0.7
	epsilon = 0.1
	gamma = 0.95
	kappa = 1e-3
	initialize_global_parameters(epsilon, alpha, gamma, kappa)
	# decide parameters of environment
	start = (5, 3)
	goal = (0, 8)
	# initial obstacles
	obstacles_1 = [(3, i) for i in range(1, 9)]
	# new obstacles to be added later
	obstacles_2 = [(3, i) for i in range(1, 8)]
	env = DynaMaze(start, goal, obstacles_1)
	# decide parameters of algorithm
	planning_steps = 50
	runs = 5
	# decide timesteps
	max_time_steps = 6000
	obstacle_change_time = 3000
	params = Parameters(planning_steps, runs,
						max_time_steps, obstacle_change_time)
	q_vs_q_plus(env, obstacles_1, obstacles_2, params)

fig_8_3()
fig_8_5()
fig_8_6()
# TODO: Implement prioritized sweeping: fig_8_7()