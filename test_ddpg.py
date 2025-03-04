import gym
import pybullet as p
import numpy as np
import torch
from gym_pybullet_drones.envs.HoverAviary import HoverAviary

# ✅ Load the trained Actor model
from train import Actor, PID  # Import your actor network and PID class

MODEL_PATH = "ddpg_pid_model.pth"  # Path to trained model

# ✅ Initialize Environment (Enable GUI for Visualization)
env = HoverAviary(gui=True)
waypoints = [[0, 0, 1.5], [1, 1, 2.0], [2, -1, 2.5], [-1, 2, 1.8], [0, 0, 1.5]]  # Fixed waypoints

# ✅ Load the trained DDPG Actor model
state_size = 9
action_size = 3
actor = Actor(state_size, action_size)
actor.load_state_dict(torch.load(MODEL_PATH))
actor.eval()  # Set model to evaluation mode

# ✅ Function to Test the Trained Model
def test_trained_ddpg(env, actor):
    for waypoint in waypoints:
        target_x, target_y, target_z = waypoint
        print(f"🚀 Moving to waypoint: {waypoint}")
        
        pid_x = PID(Kp=0.5, Ki=0.1, Kd=0.05)  # Initial PID values
        done = False
        step_counter = 0

        obs, _, _, _, _ = env.step(np.zeros((1, 4)))  # Get initial state
        pos = np.array(obs[0, :3], dtype=float)

        while not done and step_counter < 50:
            step_counter += 1
            error = np.array([target_x - pos[0], target_y - pos[1], target_z - pos[2]])
            state = np.concatenate((pos, error, [pid_x.Kp, pid_x.Ki, pid_x.Kd]))

            # ✅ Use trained actor model to get PID tuning adjustments
            action = actor(torch.FloatTensor(state).unsqueeze(0)).detach().numpy()[0]
            pid_x.Kp += action[0] * 0.1  # Apply learned changes
            pid_x.Ki += action[1] * 0.02
            pid_x.Kd += action[2] * 0.1

            # ✅ Compute new PID-controlled motor action
            action_x = pid_x.update(error[0], 0.1)
            action_y = pid_x.update(error[1], 0.1)
            action_z = pid_x.update(error[2], 0.1)
            motor_action = np.clip(np.array([[action_x, action_y, action_z, 1.2]]), 0.5, 1.7)

            # ✅ Apply action to the drone
            next_obs, _, done, _, _ = env.step(motor_action)
            next_pos = np.array(next_obs[0, :3], dtype=float)

            # ✅ Debugging output
            print(f"📌 Step {step_counter} | Position: {pos} | Error: {error} | Action: {action}")

            pos = next_pos  # Update position
            env.render()  # Render environment for visualization

            # ✅ Stop if the waypoint is reached
            if np.linalg.norm(error) < 0.2:
                print(f"✅ Waypoint {waypoint} reached!")
                break

    env.close()

# ✅ Run the test
test_trained_ddpg(env, actor)
