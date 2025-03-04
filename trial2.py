import torch
import os
import numpy as np
from stable_baselines3 import PPO, DDPG, SAC
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")
env = HoverAviary(gui=False)
TIMESTEPS = 300000
MODEL_SAVE_PATH = "hoveraviary_models"
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
models = {
    "PPO": PPO(
        "MlpPolicy",
        env,
        verbose=1,
        device=device,
        n_steps=2048,
        batch_size=64,
        learning_rate=0.0003,
        gamma=0.99,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
    ),
    "DDPG": DDPG(
        "MlpPolicy",
        env,
        verbose=1,
        device=device,
        learning_rate=0.001,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
    ),
    "SAC": SAC(
        "MlpPolicy",
        env,
        verbose=1,
        device=device,
        learning_rate=0.0003,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        ent_coef="auto",
    ),
}
for algo, model in models.items():
    model_path = f"{MODEL_SAVE_PATH}/{algo}.zip"
    if os.path.exists(model_path):
        print(f"{algo} model already exists. Skipping training.")
        models[algo] = models[algo].load(model_path, env)
    else:
        print(f"Training {algo} model...")
        model.learn(total_timesteps=TIMESTEPS)
        model.save(model_path)
        print(f"{algo} model saved successfully!\n")

# Load trained models for inference
trained_models = {algo: models[algo].load(f"{MODEL_SAVE_PATH}/{algo}.zip", env) for algo in models.keys()}
print("All models trained, saved, and ready to use!")
