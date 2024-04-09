import gym
import numpy as np

from monte_carlo import mc_state_value_estimation as mc
from monte_carlo import eval_policy as ep
from monte_carlo import grid_print as gp
desc=["SFFF", "FFFF", "FFFF", "HFFG"]
#NO RENDER
env = gym.make('FrozenLake-v1', desc=["SFFF", "FFFF", "FFFF", "HFFG"], map_name="4x4", is_slippery=True)
#WITH RENDER
#env=gym.make('FrozenLake-v1', desc=desc, map_name="4x4", is_slippery=True,render_mode="human")
s_num = env.observation_space.n
n_episodes = 10000
gamma = 1
est_values_mc = mc(env, s_num, n_episodes, gamma)
initial_policy = (1 / 4) * np.ones((16, 4))
v_init = np.zeros(env.observation_space.n)
max_iter = 1000
tol = 10 ** (-6)
state = env.reset()
done = False
total_reward = 0
print("Total Cumulative Reward:", total_reward)
env.close()
v_iter_policy_eval = ep(env, v_init, initial_policy, 1, max_iter, tol)
gp(v_iter_policy_eval, 4,'iterPolEvalEst.png')
