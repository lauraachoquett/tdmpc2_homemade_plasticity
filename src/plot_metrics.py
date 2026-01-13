import matplotlib.pyplot as plt
import os 
import csv 
import numpy as np

def plot_metrics(folder: str):
    log_path = os.path.join(folder, "train.log")

    env_step = []
    episode_reward = []
    grad_norm = []
    weight_distance = []
    weight_magnitude = []
    zgr = []
    fzar = []
    srank = []

    required_fields = [
        "env_step",
        "episode_reward",
        "grad_norm",
        "weight_distance",
        "weight_magnitude",
        "zgr",
        "fzar",
        "srank",
    ]

    skipped = 0

    with open(log_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # vérifier que toutes les valeurs sont présentes et non vides
            if any(row[field] == "" for field in required_fields):
                skipped += 1
                continue

            try:
                env_step.append(int(row["env_step"]))
                episode_reward.append(float(row["episode_reward"]))
                grad_norm.append(float(row["grad_norm"]))
                weight_distance.append(float(row["weight_distance"]))
                weight_magnitude.append(float(row["weight_magnitude"]))
                zgr.append(float(row["zgr"]))
                fzar.append(float(row["fzar"]))
                srank.append(float(row["srank"]))
            except ValueError:
                skipped += 1
                continue

    print(f"Skipped {skipped} incomplete lines")
    # Création figure 2x2
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.flatten()  # pour indexer facilement axs[0], axs[1], ...

    # 1️⃣ Weight magnitude & distance
    axs[0].plot(env_step, weight_magnitude, label='Weight Magnitude', color='tab:blue')
    axs[0].plot(env_step, weight_distance, label='Weight Distance', color='tab:orange')
    axs[0].set_title('Weights Metrics')
    axs[0].set_xlabel('Episode')
    axs[0].set_ylabel('Value')
    axs[0].legend()
    axs[0].grid(True)

    # 2️⃣ ZGR
    axs[1].plot(env_step, np.array(zgr)*100, color='tab:red')
    axs[1].set_title('Zero Gradient Ratio (ZGR)')
    axs[1].set_xlabel('Episode')
    axs[1].set_ylabel('ZGR (%)')
    axs[1].grid(True)

    # 3️⃣ FZAR
    axs[2].plot(env_step, np.array(fzar)*100, color='tab:green')
    axs[2].set_title('Feature Zero Activation Ratio (FZAR)%')
    axs[2].set_xlabel('Episode')
    axs[2].set_ylabel('FZAR (%)')
    axs[2].grid(True)

    # 4️⃣ Feature Rank
    axs[3].scatter(env_step, srank, color='tab:purple')
    axs[3].set_title('Feature Rank (srank)')
    axs[3].set_xlabel('Episode')
    axs[3].set_ylabel('srank')
    axs[3].grid(True)

    plt.tight_layout()
    path_save_fig = os.path.join(folder,'fig/')
    os.makedirs(path_save_fig,exist_ok=True)
    plt.savefig(path_save_fig +'plot_metrics.png',dpi=300,bbox_inches='tight')
    
if __name__ == '__main__' : 
    path_train = 'logs/cartpole-swingup/state/test_fast/1' 
    plot_metrics(path_train)