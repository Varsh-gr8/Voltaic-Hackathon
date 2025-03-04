'''import gym
import pybullet as p
import numpy as np
import time
import csv
from gym_pybullet_drones.envs.HoverAviary import HoverAviary

class PID:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.prev_error = 0.0  # Store as a single float instead of an array
        self.integral = 0.0    # Store as a single float instead of an array

    def update(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        output = (self.Kp * error) + (self.Ki * self.integral) + (self.Kd * derivative)
        return float(output)  # Ensure the return value is a single scalar float


# ✅ Function to Generate Random Waypoints
def generate_random_waypoints(num_waypoints, x_range, y_range, z_range):
    return [np.random.uniform([x_range[0], y_range[0], z_range[0]], 
                              [x_range[1], y_range[1], z_range[1]]) for _ in range(num_waypoints)]

# ✅ Function to Navigate to Waypoints Using PID
def navigate_to_waypoints(env, waypoints, max_steps):
    pid_x = PID(Kp=0.8, Ki=0.2, Kd=0.3)
    pid_y = PID(Kp=0.8, Ki=0.2, Kd=0.3)
    pid_z = PID(Kp=1.0, Ki=0.3, Kd=0.4)

    dt = 0.1
    csv_filename = "drone_pid_logs.csv"

    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Waypoint', 'Step', 'Error_X', 'Error_Y', 'Error_Z',
                         'Action_X', 'Action_Y', 'Action_Z', 'Action_W'])

    for waypoint in waypoints:
        target_x, target_y, target_z = waypoint
        step_counter = 0
        print(f"📌 Moving to waypoint: {waypoint}")

        while step_counter < max_steps:
            step_counter += 1
            action = np.zeros((1, 4))
            obs, reward, done, info, _ = env.step(action)

            if obs.size >= 3:
                pos = np.array(obs[0, :3], dtype=float)
            else:
                print(f"⚠️ Invalid observation size: {obs.size}, assigning default position.")
                pos = np.array([0, 0, 1.5])

            # ✅ Correct PID Inputs
            error = np.array([target_x - pos[0], target_y - pos[1], target_z - pos[2]])

            # ✅ Ensure PID outputs are scalar values
            action_x = float(pid_x.update(error[0], dt))
            action_y = float(pid_y.update(error[1], dt))
            action_z = float(pid_z.update(error[2], dt))

            # ✅ Apply action values properly
            action = np.clip([action_x, action_y, action_z, 0.8], 0.4, 1.6).reshape(1, 4)

            print(f"🔍 Step {step_counter} | Target: {waypoint} | Position: {pos} | Error: {error}")
            print(f"⚡ Action Sent: {action}")

            with open(csv_filename, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([waypoint, step_counter, error[0], error[1], error[2],
                                 action[0, 0], action[0, 1], action[0, 2], action[0, 3]])

            env.step(action)
            env.render()

            if np.linalg.norm(error[:3]) < 0.1:
                print(f"✅ Waypoint {waypoint} reached!")
                break

            time.sleep(dt)

# ✅ Main Function
def main():
    env = HoverAviary(gui=True)
    env.reset()

    num_waypoints = 5
    x_range = (-2, 2)
    y_range = (-2, 2)
    z_range = (1, 3)
    waypoints = generate_random_waypoints(num_waypoints, x_range, y_range, z_range)

    max_steps = 500
    navigate_to_waypoints(env, waypoints, max_steps)

    env.close()

# ✅ Run the Main Function if the Script is Executed
if __name__ == "__main__":
    main()




import gym
import pybullet as p
import numpy as np
import time
import csv
from gym_pybullet_drones.envs.HoverAviary import HoverAviary

# ✅ Improved PID Controller Class
class PID:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.prev_error = 0.0  # Store as a single float
        self.integral = 0.0    # Store as a single float

    def update(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        output = (self.Kp * error) + (self.Ki * self.integral) + (self.Kd * derivative)
        return float(output)  # Return a scalar float value

# ✅ Function to Generate Random Waypoints
def generate_random_waypoints(num_waypoints, x_range, y_range, z_range):
    return [np.random.uniform([x_range[0], y_range[0], z_range[0]], 
                              [x_range[1], y_range[1], z_range[1]]) for _ in range(num_waypoints)]

# ✅ Function to Navigate to Waypoints Using PID
def navigate_to_waypoints(env, waypoints, max_steps):
    # ✅ Increased PID Gains for Faster Response
    pid_x = PID(Kp=1.2, Ki=0.2, Kd=0.3)
    pid_y = PID(Kp=1.2, Ki=0.2, Kd=0.3)
    pid_z = PID(Kp=1.5, Ki=0.3, Kd=0.4)  # Stronger altitude control

    dt = 0.1  # Time step for PID update
    csv_filename = "drone_pid_logs.csv"

    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Waypoint', 'Step', 'Error_X', 'Error_Y', 'Error_Z',
                         'Action_X', 'Action_Y', 'Action_Z', 'Action_W'])

    for waypoint in waypoints:
        target_x, target_y, target_z = waypoint
        step_counter = 0
        print(f"📌 Moving to waypoint: {waypoint}")

        while step_counter < max_steps:
            step_counter += 1
            action = np.zeros((1, 4))
            if np.any(np.isnan(action)) or np.any(np.isinf(action)):
               print(f"🚨 Invalid action detected: {action}. Resetting to default.")
               action = np.array([[1.0, 1.0, 1.0, 1.0]])  # Set to a safe default

            obs, reward, done, info, _ = env.step(action)

            if obs.size >= 3:
                pos = np.array(obs[0, :3], dtype=float)
            else:
                print(f"⚠️ Invalid observation size: {obs.size}, assigning default position.")
                pos = np.array([0, 0, 1.5])

            # ✅ Calculate Error
            error = np.array([target_x - pos[0], target_y - pos[1], target_z - pos[2]])

            # ✅ Ensure PID outputs are scalar values
            action_x = pid_x.update(error[0], dt)
            action_y = pid_y.update(error[1], dt)
            action_z = pid_z.update(error[2], dt)

            # ✅ Increase Minimum Action Strength for Movement
            #action = np.clip([action_x, action_y, action_z, 1.0], 0.6, 1.6).reshape(1, 4)
            action = np.clip(np.array([[action_x, action_y, action_z, 1.0]]), 0.6, 1.6)

            # ✅ Debugging Output
            print(f"🔍 Step {step_counter} | Position: {pos} | Target: {waypoint} | Error: {error}")
            print(f"⚡ Action Sent: {action}")

            # ✅ Log Data to CSV
            with open(csv_filename, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([f"[{target_x:.2f}, {target_y:.2f}, {target_z:.2f}]", step_counter,
                   round(error[0], 3), round(error[1], 3), round(error[2], 3),
                   round(action[0, 0], 3), round(action[0, 1], 3), round(action[0, 2], 3), round(action[0, 3], 3)])


            # ✅ Apply Action and Render
            env.step(action)
            env.render()

            # ✅ Increased Tolerance for Reaching Waypoint
            if np.linalg.norm(error[:3]) < 0.3:  # Was 0.1, now 0.3 for smoother behavior
                print(f"✅ Waypoint {waypoint} reached!")
                break

            time.sleep(dt)

# ✅ Main Function to Run the Drone Navigation
def main():
    env = HoverAviary(gui=True)
    env.reset()

    num_waypoints = 5
    x_range = (-2, 2)
    y_range = (-2, 2)
    z_range = (1, 3)
    waypoints = generate_random_waypoints(num_waypoints, x_range, y_range, z_range)

    max_steps = 500  # Define max_steps
    navigate_to_waypoints(env, waypoints, max_steps)

    env.close()

# ✅ Run the Main Function if the Script is Executed
if __name__ == "__main__":
    main()
'''
import gym
import pybullet as p
import numpy as np
import time
import csv
from gym_pybullet_drones.envs.HoverAviary import HoverAviary

# ✅ Tuned PID Controller for Initial Training Data Collection
class PID:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.prev_error = 0.0
        self.integral = 0.0

    def update(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        output = (self.Kp * error) + (self.Ki * self.integral) + (self.Kd * derivative)
        return float(output)  # Return scalar value

# ✅ Generate Random Waypoints for Diverse Training Data
def generate_random_waypoints(num_waypoints, x_range, y_range, z_range):
    return [np.random.uniform([x_range[0], y_range[0], z_range[0]], 
                              [x_range[1], y_range[1], z_range[1]]) for _ in range(num_waypoints)]

# ✅ Function to Navigate to Waypoints Using PID
def navigate_to_waypoints(env, waypoints, max_steps):
    # ✅ PID Gains for Data Collection (RL Will Replace This Later)
    pid_x = PID(Kp=0.5, Ki=0.1, Kd=0.05)
    pid_y = PID(Kp=1.5, Ki=0.1, Kd=0.05)
    pid_z = PID(Kp=0.8, Ki=0.2, Kd=0.2)

    dt = 0.1  # ✅ Faster update for smoother data collection
    csv_filename = "drone_rl_training_data.csv"

    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Step', 'State_X', 'State_Y', 'State_Z',
                         'Error_X', 'Error_Y', 'Error_Z',
                         'Action_X', 'Action_Y', 'Action_Z',
                         'Reward', 'Next_State_X', 'Next_State_Y', 'Next_State_Z', 'Done'])

    for waypoint in waypoints:
        target_x, target_y, target_z = waypoint
        step_counter = 0
        done = False

        print(f"📌 Moving to waypoint: {waypoint}")

        while step_counter < max_steps:
            step_counter += 1

            # ✅ Get Current State
            obs, reward, done, info, _ = env.step(np.zeros((1, 4)))  # Dummy action for initial state
            pos = np.array(obs[0, :3], dtype=float)

            # ✅ Calculate Error (Distance to Target)
            error = np.array([target_x - pos[0], target_y - pos[1], target_z - pos[2]])

            # ✅ Improved Reward Function for RL
            if np.linalg.norm(error[:3]) < 0.3:
                reward = 10  # ✅ Large reward for reaching waypoint
            else:
                reward = -np.linalg.norm(error[:3]) - 0.05 * np.linalg.norm(error)  # ✅ Penalize large movements

            # ✅ Apply PID Control (For Now)
            action_x = pid_x.update(error[0], dt)
            action_y = pid_y.update(error[1], dt)
            action_z = pid_z.update(error[2], dt)

            # ✅ Adjusted Action Clipping (Allows More Exploration)
            action = np.clip(np.array([[action_x, action_y, action_z, 1.2]]), 0.5, 1.7)

            # ✅ Take a Step with the Updated Action
            next_obs, _, done, _, _ = env.step(action)
            next_pos = np.array(next_obs[0, :3], dtype=float)

            # ✅ Debugging Print for Actions and Error
            print(f"🔍 Step {step_counter} | Position: {pos} | Target: {waypoint} | Error: {error} | Action: {action} | Reward: {reward}")

            # ✅ Render the Environment for Visualization
            #env.render()

            # ✅ Log Data for RL Training
            with open(csv_filename, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([step_counter, pos[0], pos[1], pos[2],
                                 error[0], error[1], error[2],
                                 action[0, 0], action[0, 1], action[0, 2],
                                 reward,
                                 next_pos[0], next_pos[1], next_pos[2],
                                 int(done)])

            # ✅ Stop Condition: Increased Tolerance to 0.3 (Prevents Overcorrection)
            if np.linalg.norm(error[:3]) < 0.3:
                done = True
                print(f"✅ Waypoint {waypoint} reached!")
                break

            time.sleep(dt / 2)  # ✅ Reduced sleep time for smoother effect

# ✅ Main Function to Run the Drone Navigation
def main():
    env = HoverAviary(gui=True)
    env.reset()

    num_waypoints = 5
    x_range = (-2, 2)
    y_range = (-2, 2)
    z_range = (1, 3)
    waypoints = generate_random_waypoints(num_waypoints, x_range, y_range, z_range)

    max_steps = 500  # Define max_steps
    navigate_to_waypoints(env, waypoints, max_steps)

    env.close()

# ✅ Run the Main Function if the Script is Executed
if __name__ == "__main__":
    main()





