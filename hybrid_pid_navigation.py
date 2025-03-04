import gym
import pybullet as p
import numpy as np
import time
from gym_pybullet_drones.envs.HoverAviary import HoverAviary

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

# ✅ Navigation Function
def navigate_to_waypoints(env, waypoints, max_steps=500, tolerance=0.05, stable_duration=2.0):
    for waypoint in waypoints:
        target_x, target_y, target_z = waypoint  
        step_counter = 0
        stable_time = 0.0  # Time the drone remains within the tolerance distance
        start_time = time.time()
        print(f"🚀 Moving to waypoint: {waypoint}")

        while step_counter < max_steps:
            step_counter += 1

            # ✅ Get latest state
            action = np.zeros((1, 4))  
            obs, reward, done, info, _ = env.step(action)  

            # ✅ Debugging print for `obs`
            print(f"📊 Observation received: {obs} (Shape: {obs.shape})")

            # ✅ Ensure `pos` is always 1D
            if obs.size >= 3:
                pos = np.array(obs[0, :3], dtype=float)  
            else:
                print(f"❌ Invalid observation size: {obs.size}, assigning default position.")
                pos = np.array([0, 0, 1.5])  

            # ✅ Compute error vector (Now includes yaw = 0)
            error = np.array([target_x - pos[0], target_y - pos[1], target_z - pos[2], 0])  # ✅ Fix applied
            action = np.clip(error * 0.2 + 0.5, 0.4, 1.6)  

            # ✅ Ensure action is (1,4)
            action = np.reshape(action, (1, 4))
            print(f"🔹 Action Shape: {action.shape} | Action: {action}")

            env.step(action)
            env.render()

            # ✅ Stop if waypoint is reached or if drone remains stable for a specified duration
            if np.linalg.norm(error[:3]) < tolerance:
                stable_time += time.time() - start_time
                if stable_time >= stable_duration:
                    print(f"✅ Waypoint {waypoint} reached and stabilized for {stable_duration} seconds!")
                    return
            else:
                stable_time = 0.0  # Reset stable time if outside tolerance

            time.sleep(0.05)
            start_time = time.time()

# ✅ Execute navigation
navigate_to_waypoints(env, waypoints)

# ✅ Close environment
env.close()
