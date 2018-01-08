import numpy as np
import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

def initialize_global_parameters(epsilon=0.8, alpha=0.5, gamma=0.9):
	global EPSILON
	EPSILON = epsilon
	global ALPHA
	ALPHA = alpha
	global GAMMA
	GAMMA = gamma

class WindyGridWorld(object):
	def __init__(self, size=[7,10], goal=[5,5], wind=[0]*10):
		self.size = size
		self.goal = goal
		self.wind = wind
		self.nA = 4

	def render(self, agent, trajectory, reward, stay):
		map = np.chararray(self.size)
		for i in range(self.size[0]):
			for j in range(self.size[1]):
				map[i][j] = "."
		for cell in trajectory:
			map[cell[0]][cell[1]] = "#"
		map[agent.global_start[0]][agent.global_start[1]] = "S"
		map[self.goal[0]][self.goal[1]] = "T"
		for i in range(self.size[0]):
			for j in range(self.size[1]):
				print(map[i][j].decode(), "|", end=' ')
			print()
		print()
		print("Reward Obtained:", reward)
		if not stay:
			time.sleep(0.01)
			os.system('cls')


class Agent(object):
	def __init__(self, env, start=[2,0]):
		self.global_start = start
		self.current_state = start
		self.max_R = env.size[0] - 1
		self.max_C = env.size[1] - 1
		self.action_values = np.zeros([env.nA, env.size[0], env.size[1]])

	def get_next_state(self, env, action):
		curr_row = self.current_state[0]
		curr_col = self.current_state[1]
		if action == 0:
			# moving up
			next_col = curr_col
			next_row = max(curr_row - env.wind[next_col] - 1, 0)
		elif action == 1:
			# moving right
			next_col = min(curr_col + 1, self.max_C)
			next_row = max(curr_row - env.wind[next_col], 0)
		elif action == 2:
			# moving down
			next_col = curr_col
			next_row = max(0, min(curr_row - env.wind[next_col] + 1, self.max_R))
		elif action == 3:
			# moving left
			next_col = max(curr_col - 1, 0)
			next_row = max(curr_row - env.wind[next_col], 0)

		return [next_row, next_col]


	def move_one_step(self, env):
		if not np.random.binomial(1, EPSILON) == 1:
			# follow greedy
			# action = np.argmax(self.action_values[:,self.current_state[0], self.current_state[1]])
			sorted_action_list = np.argsort(self.action_values[:, self.current_state[0], self.current_state[1]])[::-1]
			sorted_value_list = np.sort(self.action_values[:, self.current_state[0], self.current_state[1]])[::-1]
			equal_acts = 1
			for i in range(len(sorted_value_list) - 1):
				if sorted_value_list[i] == sorted_value_list[i + 1]:
					equal_acts += 1
				else:
					break

			action = int(sorted_action_list[int(np.random.uniform() * equal_acts)])
		else:
			action = int(np.random.uniform() * env.nA)

		
		return [action, self.get_next_state(env, action)]

	def reset(self):
		self.current_state = self.global_start


class TD0(object):
	def __init__(self):
		self.trajectory = []

	def generate_episode(self, agent, env, step_length=100):
		steps = 0
		while steps < step_length and not agent.current_state == env.goal:
			steps += 1
			self.trajectory.append(agent.current_state)
			old_state = agent.current_state
			[action_to_take, s_prime] = agent.move_one_step(env)
			# store Q(s,a)
			Q_s_a = agent.action_values[action_to_take, agent.current_state[0], agent.current_state[1]]
			# update agent's state to new state s'
			agent.current_state = s_prime
			# get a' -> action at s' by following current policy
			[a_prime, garbage] = agent.move_one_step(env)
			# store Q(s',a')
			Q_spr_apr = agent.action_values[a_prime, agent.current_state[0], agent.current_state[1]]
			R = -1
			if agent.current_state == env.goal:
				R = 0
			td_target = R + GAMMA * Q_spr_apr
			agent.action_values[action_to_take, old_state[0], old_state[1]] = Q_s_a + ALPHA * (td_target - Q_s_a)
		
		total_steps = len(self.trajectory)
		if agent.current_state == env.goal:
			# we've reached our destination
			self.trajectory.append(agent.current_state)

		return [self.trajectory, total_steps]


def run_episodes(num_iter, algo_num):
	env = WindyGridWorld(goal=[3,7], wind=[0,0,0,1,1,1,2,2,1,0])
	agent = Agent(env, start=[3,0])
	steps_taken = []
	for i in range(num_iter):
		if algo_num == 1:
			control = TD0()
		[path, steps] = control.generate_episode(agent, env, step_length=2500)
		steps_taken.append(steps)
		# env.render(agent, path, steps, (i+1)==num_iter)
		agent.reset()
	return steps_taken

def test_run():
	num_iter = 1500
	decay_window = 200
	initialize_global_parameters(epsilon=0.1, alpha=0.5, gamma=0.9)
	steps_taken = run_episodes(num_iter, decay_window, algo_num=1)
	plt.plot(steps_taken, label="SARSA")
	plt.xlabel("Iterations")
	plt.ylabel("Number of steps to reach goal")
	plt.title("SARSA in Windy Grid World")
	plt.legend()
	plt.show()

def fig_6_4():
	initialize_global_parameters(epsilon=0.1, alpha=0.5, gamma=0.9)
	total_steps = 0
	prev_steps = total_steps
	episodes = 0
	max_steps_to_take = 8000
	episodes_per_step = np.zeros(max_steps_to_take)
	env = WindyGridWorld(goal=[3,7], wind=[0,0,0,1,1,1,2,2,1,0])
	agent = Agent(env, start=[3,0])
	while total_steps < max_steps_to_take:
		control = TD0()
		print("Generating episode: ", episodes + 1)
		# run 1 episode to find steps taken in that episode
		[path, steps] = control.generate_episode(agent, env, 500)
		print(steps)
		total_steps += steps
		episodes_per_step[prev_steps: total_steps] = episodes
		episodes += 1
		episodes_per_step[min(max_steps_to_take - 1, total_steps)] = episodes
		prev_steps = total_steps
		agent.reset()

	plt.plot(episodes_per_step, label="SARSA")
	plt.xlabel("Number of timesteps")
	plt.ylabel("Number of Episodes")
	plt.title("SARSA in Windy Grid World")
	plt.legend()
	plt.show()


if __name__ == "__main__":
	fig_6_4()