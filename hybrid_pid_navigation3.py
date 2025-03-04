import gym
import pybullet as p
import numpy as np
import time
import csv
from gym_pybullet_drones.envs.HoverAviary import HoverAviary

env = HoverAviary(gui=True)
env.reset()


def generate_random_waypoints(num_waypoints, x_range, y_range, z_range):
    waypoints = []
    for _ in range(num_waypoints):
        x = np.random.uniform(*x_range)
        y = np.random.uniform(*y_range)
        z = np.random.uniform(*z_range)
        waypoints.append([x, y, z])
    return waypoints


num_waypoints = 5
x_range = (-2, 2)
y_range = (-2, 2)
z_range = (1, 3)
waypoints = generate_random_waypoints(num_waypoints, x_range, y_range, z_range)

with open('pid_values.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Waypoint', 'Step', 'Error_X', 'Error_Y', 'Error_Z',
                     'Action_X', 'Action_Y', 'Action_Z', 'Action_W'])


def navigate_to_waypoints(env, waypoints, max_steps=500):
    for waypoint in waypoints:
        target_x, target_y, target_z = waypoint
        step_counter = 0
        print(f"Moving to waypoint: {waypoint}")

        while step_counter < max_steps:
            step_counter += 1
            action = np.zeros((1, 4))
            obs, reward, done, info, _ = env.step(action)
            print(f"Observation received: {obs} (Shape: {obs.shape})")

            if obs.size >= 3:
                pos = np.array(obs[0, :3], dtype=float)
            else:
                print(f"Invalid observation size: {obs.size}, assigning default position.")
                pos = np.array([0, 0, 1.5])

            error = np.array([target_x - pos[0], target_y - pos[1], target_z - pos[2], 0])
            random_scale = np.random.uniform(0.1, 0.3)
            base_action = np.clip(error * random_scale + 0.5, 0.4, 1.6)

            noise = np.random.normal(0, 0.05, base_action.shape)
            action = base_action + noise
            action = np.reshape(action, (1, 4))

            print(f"🎮 Action Shape: {action.shape} | Action: {action}")

            with open('pid_values.csv', mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([waypoint, step_counter, error[0], error[1], error[2],
                                 action[0, 0], action[0, 1], action[0, 2], action[0, 3]])

            env.step(action)
            env.render()

            if np.linalg.norm(error[:3]) < 0.05:
                print(f"Waypoint {waypoint} reached!")
                break

            time.sleep(0.1)


navigate_to_waypoints(env, waypoints)
env.close()


