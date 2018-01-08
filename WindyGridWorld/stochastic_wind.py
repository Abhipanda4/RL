"""
Re-solve the windy gridworld task with King’s
moves, assuming that the effect of the wind, if there is any, is stochastic,
sometimes varying by 1 from the mean values given for each column. That is,
a third of the time you move exactly according to these values, as in the
previous exercise, but also a third of the time you move one cell above that,
and another third of the time you move one cell below that.

For example, if you are one cell to the right of the goal and you move left,
then one-third of the time you move one cell above the goal, one-third of the
time you move two cells above the goal, and one-third of the time you move to
the goal.
"""

import numpy as np
import matplotlib.pyplot as pyplot
from tqdm import tqdm

global ALPHA
global GAMMA

def initialize_global_parameters(alpha=0.5, gamma=0.9):
	ALPHA = alpha
	GAMMA = gamma


class StochasticWorld(object):
	def __init__(self, size=[7,10], start=[2,0], goal=[5,5], wind=[0]*10, nA=8):
		self.size = size
		self.start = start
		self.goal = goal
		self.wind = wind
		self.nA = 8

	def get_next_state(self, curr_state, action):
		[curr_row, curr_col] = curr_state
		if action == UP:
			next_col = curr_col
			next_row = max(curr_row - env.wind[next_col] - 1, 0)
		elif action == RIGHT:
			next_col = min(curr_col + 1, self.max_C)
			next_row = max(curr_row - env.wind[next_col], 0)
		elif action == DOWN:
			next_col = curr_col
			next_row = max(0, min(curr_row - env.wind[next_col] + 1, self.max_R))
		elif action == LEFT:
			next_col = max(curr_col - 1, 0)
			next_row = max(curr_row - env.wind[next_col], 0)

		return [next_row, next_col]