#left
import gym
import pybullet as p
import numpy as np
import time
import csv
from gym_pybullet_drones.envs.HoverAviary import HoverAviary

# ✅ PID Controller for Smooth Motion
class PID:
    def __init__(self, Kp, Ki, Kd):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.prev_error = 0
        self.integral = 0

    def update(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return output

# ✅ Initialize the environment
env = HoverAviary(gui=True)  
env.reset()

# ✅ Define waypoints
waypoints = [
    [0, 0, 1.5],  
    [1, 1, 2],    
    [2, -1, 2.5],  
    [-1, 2, 1.8],  
    [0, 0, 1.5]    
]

# ✅ Open CSV for Logging
csv_filename = "pid_navigation_data.csv"

with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["x", "y", "z", "vx", "vy", "vz", "wx", "wy", "wz", "target_x", "target_y", "target_z", "error_x", "error_y", "error_z", "yaw_error", "rpm_1", "rpm_2", "rpm_3", "rpm_4"])

    # ✅ Initialize PID Controllers for XYZ & Yaw
    pid_x = PID(0.8, 0.02, 0.1)
    pid_y = PID(0.8, 0.02, 0.1)
    pid_z = PID(1.2, 0.02, 0.2)
    pid_yaw = PID(1.0, 0.01, 0.1)

    # ✅ Navigation Function
    def navigate_to_waypoints(env, waypoints, max_steps=500):
        for waypoint in waypoints:
            target_x, target_y, target_z = waypoint  
            step_counter = 0
            print(f"🚀 Moving to waypoint: {waypoint}")

            while step_counter < max_steps:
                step_counter += 1

                # ✅ Get latest state
                action = np.zeros((1, 4))  
                obs, reward, done, info, _ = env.step(action)  

                # ✅ Debugging print for `obs`
                print(f"📊 Observation received: {obs} (Shape: {obs.shape})")

                # ✅ Extract Position, Velocity, and Angular Velocity
                if obs.size >= 13:
                    pos = np.array(obs[0, :3], dtype=float)  
                    vel = np.array(obs[0, 3:6], dtype=float)  
                    ang_vel = np.array(obs[0, 6:9], dtype=float)  
                else:
                    print(f"❌ Invalid observation size: {obs.size}, assigning default position.")
                    pos = np.array([0, 0, 1.5])  
                    vel = np.zeros(3)
                    ang_vel = np.zeros(3)

                # ✅ Compute Yaw (Heading) Error
                target_yaw = np.arctan2(target_y - pos[1], target_x - pos[0])
                drone_yaw = np.arctan2(obs[0, 4], obs[0, 3])  # Extract yaw from observation
                yaw_error = (target_yaw - drone_yaw + np.pi) % (2 * np.pi) - np.pi

                # ✅ Compute error vector
                dt = 0.05  # Step time
                error_x = pid_x.update(target_x - pos[0], dt)
                error_y = pid_y.update(target_y - pos[1], dt)
                error_z = pid_z.update(target_z - pos[2], dt)
                error_yaw = pid_yaw.update(yaw_error, dt)

                # ✅ Generate motor RPMs using PID values
                action = np.array([
                    error_x,
                    error_y,
                    error_z,
                    error_yaw
                ])
                action = np.clip(action, 0.4, 1.6).reshape((1, 4))

                # ✅ Log Data to CSV
                writer.writerow([*pos, *vel, *ang_vel, target_x, target_y, target_z, target_x - pos[0], target_y - pos[1], target_z - pos[2], yaw_error, *action[0]])
                file.flush()  # Ensures data is written immediately

                env.step(action)
                env.render()

                # ✅ Stop if waypoint is reached
                if np.linalg.norm([target_x - pos[0], target_y - pos[1], target_z - pos[2]]) < 0.1:
                    print(f"✅ Waypoint {waypoint} reached!")
                    break

                time.sleep(0.05)

            else:
                print(f"❌ Max steps reached without reaching waypoint {waypoint}. Skipping...")

        print("🏁 All waypoints completed. Stopping simulation.")

    # ✅ Execute navigation
    navigate_to_waypoints(env, waypoints)

# ✅ Close environment
env.close()
print(f"✅ Data saved to {csv_filename}")
