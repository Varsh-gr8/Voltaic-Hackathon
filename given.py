import gym
import pybullet
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
import numpy as np
import time
env = HoverAviary(gui=True)
env.reset()
for i in range(100):
  action = np.array([0.2,0.2,0.205,0.205])
  action = np.reshape(action, (1,4))
  state, reward, done, info, _ = env.step(action)
print(state)
env.render()
time.sleep(0.05)
env.close()