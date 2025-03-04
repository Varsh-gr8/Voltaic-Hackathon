import torch
import numpy as np
from stable_baselines3 import PPO, DDPG, SAC
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
env = HoverAviary(gui=False)
env.reset()
TIMESTEPS = 200000  
MODEL_SAVE_PATH = "hoveraviary_model"
models = {
    "PPO": PPO(
        "MlpPolicy", env, verbose=1, device=device,
        n_steps=8192, batch_size=2048, learning_rate=0.0005, gamma=0.995,
        ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5
    ),
    "DDPG": DDPG(
        "MlpPolicy", env, verbose=1, device=device,
        learning_rate=0.001, batch_size=1024, gamma=0.99, tau=0.005
    ),
    "SAC": SAC(
        "MlpPolicy", env, verbose=1, device=device,
        learning_rate=0.0003, batch_size=512, gamma=0.98, tau=0.005, ent_coef="auto"
    ),
}
for algo, model in models.items():
    print(f"Training {algo} model...")
    model.learn(total_timesteps=TIMESTEPS)
    model.save(f"{MODEL_SAVE_PATH}_{algo}")
    print(f"{algo} model saved successfully!\n")
for algo in models.keys():
    loaded_model = models[algo].load(f"{MODEL_SAVE_PATH}_{algo}", env)
    print(f"{algo} model loaded successfully!")

print("All models trained, saved, and ready to use!")