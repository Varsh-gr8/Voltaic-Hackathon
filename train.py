'''
import gym
import pybullet as p
import numpy as np
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from gym_pybullet_drones.envs.HoverAviary import HoverAviary

# ✅ Optimized DDPG Hyperparameters
GAMMA = 0.99  
LR = 0.001  
TAU = 0.005  
EPSILON = 1.0  
EPSILON_MIN = 0.01  
EPSILON_DECAY = 0.999  # Learns stable strategies faster
BATCH_SIZE = 16  
MEMORY_SIZE = 2000  
MAX_STEPS = 50  
NUM_EPISODES = 500  # Increased for better learning

# ✅ Improved PID Controller (RL Adjusts `Kp`, `Ki`, `Kd`)
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
        return float(output)

# ✅ Actor Network
class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, 32)  
        self.fc2 = nn.Linear(32, 32)  
        self.fc3 = nn.Linear(32, action_size)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.tanh(self.fc3(x))

# ✅ Critic Network
class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size + action_size, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, state, action):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)
        x = torch.cat([state, action], dim=1)  
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# ✅ DDPG Agent
class DDPGAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.actor = Actor(state_size, action_size)
        self.critic = Critic(state_size, action_size)
        self.target_actor = Actor(state_size, action_size)
        self.target_critic = Critic(state_size, action_size)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR)
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.gamma = GAMMA
        self.tau = TAU
        self.epsilon = EPSILON

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.uniform(-0.1, 0.1, size=(self.action_size,))
        state = torch.FloatTensor(state).unsqueeze(0)
        return self.actor(state).detach().numpy()[0]

    def train(self):
        if len(self.memory) < BATCH_SIZE:
            return

        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, next_state, done in batch:
            target_action = self.target_actor(torch.FloatTensor(next_state))
            target_q_value = self.target_critic(torch.FloatTensor(next_state), target_action)
            y = reward + (self.gamma * target_q_value * (1 - done))
            q_value = self.critic(torch.FloatTensor(state), torch.FloatTensor(action))
            critic_loss = F.mse_loss(q_value, y.detach())

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            actor_loss = -self.critic(torch.FloatTensor(state), self.actor(torch.FloatTensor(state))).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

# ✅ Train DDPG with Optimized PID Navigation
def train_ddpg():
    env = HoverAviary(gui=False)  # Disable GUI for faster training
    agent = DDPGAgent(state_size=9, action_size=3)
    waypoints = [[0, 0, 1.5], [1, 1, 2.0], [2, -1, 2.5], [-1, 2, 1.8], [0, 0, 1.5]]

    for episode in range(NUM_EPISODES):
        env.reset()
        target_x, target_y, target_z = waypoints[0]
        pid_x = PID(Kp=0.5, Ki=0.1, Kd=0.05)
        done = False
        step_counter = 0

        obs, _, _, _, _ = env.step(np.zeros((1, 4)))
        pos = np.array(obs[0, :3], dtype=float)

        while not done and step_counter < MAX_STEPS:
            step_counter += 1
            error = np.array([target_x - pos[0], target_y - pos[1], target_z - pos[2]])
            state = np.concatenate((pos, error, [pid_x.Kp, pid_x.Ki, pid_x.Kd]))

            action = agent.act(state)
            pid_x.Kp += action[0] * 0.2  # Increased adjustment impact
            pid_x.Ki += action[1] * 0.05
            pid_x.Kd += action[2] * 0.2

            action_x = pid_x.update(error[0], 0.1)
            action_y = pid_x.update(error[1], 0.1)
            action_z = pid_x.update(error[2], 0.1)
            motor_action = np.clip(np.array([[action_x, action_y, action_z, 1.2]]), 0.5, 1.7)

            next_obs, _, done, _, _ = env.step(motor_action)
            next_pos = np.array(next_obs[0, :3], dtype=float)
            next_state = np.concatenate((next_pos, error, [pid_x.Kp, pid_x.Ki, pid_x.Kd]))

            prev_error = np.linalg.norm(error)
            next_error = np.linalg.norm(next_pos - np.array([target_x, target_y, target_z]))

            reward = 5 - next_error if next_error < prev_error else -1 - next_error  # Smoother penalty

            agent.remember(state, action, reward, next_state, done)
            agent.train()
            pos = next_pos

        agent.epsilon = max(EPSILON_MIN, agent.epsilon * EPSILON_DECAY)
        print(f"🎯 Episode {episode+1}/{NUM_EPISODES} | Steps: {step_counter} | Final Reward: {reward}")

    env.close()
    torch.save(agent.actor.state_dict(), "ddpg_pid_model.pth")

# ✅ Run Training
if __name__ == "__main__":
    train_ddpg()
'''
import gym
import pybullet as p
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from gym_pybullet_drones.envs.HoverAviary import HoverAviary

# ✅ RNN-Based Actor Network (Outputs Mean and Variance for PID Adjustments)
class RNNActor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, lr):
        super(RNNActor, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc_mu = nn.Linear(hidden_size, output_size)  # Mean output
        self.fc_sigma = nn.Linear(hidden_size, output_size)  # Variance output
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x, hidden):
        output, hidden = self.rnn(x, hidden)  # ✅ No extra unsqueeze
        mu = T.tanh(self.fc_mu(output[:, -1, :]))  # Mean action
        sigma = T.abs(self.fc_sigma(output[:, -1, :])) + 1e-6  # Ensure variance is positive
        return mu, sigma, hidden  # ✅ Now returns mean and variance

    def init_hidden(self):
        return T.zeros(1, 1, self.hidden_size)

# ✅ RNN-Based Critic Network (Estimates State-Value Function)
class RNNCritic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, lr):
        super(RNNCritic, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x, hidden):
        output, hidden = self.rnn(x, hidden)  # ✅ No extra unsqueeze
        return self.fc(output[:, -1, :]), hidden

    def init_hidden(self):
        return T.zeros(1, 1, self.hidden_size)

# ✅ PID Controller (Adjusts Motor Actions Based on Error)
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
        return (self.Kp * error) + (self.Ki * self.integral) + (self.Kd * derivative)

# ✅ Drone RL Agent
class DroneAgent:
    def __init__(self, actor_lr, critic_lr, input_size, gamma=0.9):
        self.gamma = gamma
        self.actor = RNNActor(input_size, hidden_size=128, output_size=3, lr=actor_lr)  # Output = [Kp, Ki, Kd] changes
        self.critic = RNNCritic(input_size, hidden_size=128, output_size=1, lr=critic_lr)
        self.actor_hidden = self.actor.init_hidden()
        self.critic_hidden = self.critic.init_hidden()
        self.log_probs = None
        self.advantages = None

    def choose_action(self, obs_traj):
        x = T.tensor(obs_traj, dtype=T.float).unsqueeze(0)  # ✅ Correct 3D shape
        mu, sigma, self.actor_hidden = self.actor(x, self.actor_hidden)  # ✅ No extra unsqueeze

        # Sample an action using Normal distribution
        distribution = T.distributions.Normal(mu, sigma)
        sampled_action = distribution.sample().squeeze()  # ✅ Keep as a PyTorch tensor
        self.log_probs = distribution.log_prob(sampled_action).sum()  # ✅ No error now

        return sampled_action.cpu().detach().numpy()  # ✅ Convert to NumPy only at the end

    def learn(self):
        if self.log_probs is None or self.advantages is None:
            print("⚠️ Warning: log_probs or advantages is None, skipping update.")
            return  # ✅ Prevent NoneType error

        actor_loss = -self.log_probs * self.advantages
        critic_loss = (self.estimate - self.ground_truth) ** 2
        (actor_loss + critic_loss).backward()
        self.actor.optimizer.step()
        self.critic.optimizer.step()
        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()

# ✅ Training the RNN-Based DDPG Agent
def train_rnn_pid():
    env = HoverAviary(gui=False)
    agent = DroneAgent(actor_lr=0.0001, critic_lr=0.0001, input_size=9)
    waypoints = [[0, 0, 1.5], [1, 1, 2.0], [2, -1, 2.5], [-1, 2, 1.8], [0, 0, 1.5]]

    num_episodes = 500
    for i in range(num_episodes):
        obs_traj = []
        done = False
        env.reset()
        target_x, target_y, target_z = waypoints[i % len(waypoints)]  # Cycle through waypoints

        pid_x = PID(Kp=0.5, Ki=0.1, Kd=0.05)

        obs, _, _, _, _ = env.step(np.zeros((1, 4)))  # Get initial observation
        pos = np.array(obs[0, :3], dtype=float)
        vel = np.array(obs[0, 3:6], dtype=float)

        full_state = np.concatenate((pos, vel, [pid_x.Kp, pid_x.Ki, pid_x.Kd]))
        obs_traj.append(full_state.tolist())

        while not done:
            new_kp, new_ki, new_kd = agent.choose_action(obs_traj)  # ✅ Now returns 3 values

            pid_x.Kp += new_kp * 0.05
            pid_x.Ki += new_ki * 0.01
            pid_x.Kd += new_kd * 0.05

            action_x = pid_x.update(pos[0], 0.1)
            action_y = pid_x.update(pos[1], 0.1)
            action_z = pid_x.update(pos[2], 0.1)
            motor_action = np.clip(np.array([[action_x, action_y, action_z, 1.4]]), 0.8, 1.6)

            next_obs, _, done, _, _ = env.step(motor_action)

        print(f"🎯 Episode {i+1}/{num_episodes} Complete.")

if __name__ == "__main__":
    train_rnn_pid()







