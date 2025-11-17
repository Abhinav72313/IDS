import glob
import torch
import os
import time
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from NetworkEnv import IDSEnvironment
from agent import DTQNAgent
from config import *

df = pd.read_csv("./data.csv", low_memory=False)

df["label"] = df["label"].astype(str).str.strip().str.lower()
all_labels = sorted(df["label"].unique())
attack_labels = [lbl for lbl in all_labels if lbl != "normal"]
label_map = {"normal": 0}
label_map.update({lbl: i + 1 for i, lbl in enumerate(attack_labels)})

print(label_map)

df["label_encoded"] = df["label"].map(lambda x: label_map.get(x, 0)).astype(np.int64)

df = df.replace(r"^\s*$", np.nan, regex=True).fillna(0)
feature_df = df.drop(columns=["label", "label_encoded"], errors="ignore").select_dtypes(
    include=[np.number]
)
labels = df["label_encoded"].values

x_train, x_test, y_train, y_test = train_test_split(
    feature_df, labels, test_size=0.2, random_state=32, stratify=labels
)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train.astype(np.float32))
x_test_scaled = scaler.transform(x_test.astype(np.float32))

n_features = x_train.shape[1]
feature_names = list(x_train.columns)

# Use scaled numpy arrays instead of DataFrames to avoid conversion issues
train_env = IDSEnvironment(
    x_train_scaled,
    y_train,
    n_features=n_features,
    sequence_length=SEQUENCE_LENGTH,
    max_steps=MAX_STEPS_PER_EPISODE,
)
test_env = IDSEnvironment(
    x_test_scaled,
    y_test,
    n_features=n_features,
    sequence_length=SEQUENCE_LENGTH,
    max_steps=len(x_test_scaled) - 1,
)



def save_checkpoint(agent, episode, episode_rewards, filename):
    """Save model checkpoint"""
    checkpoint = {
        "episode": episode,
        "policy_net_state_dict": agent.policy_net.state_dict(),
        "target_net_state_dict": agent.target_net.state_dict(),
        "optimizer_state_dict": agent.optimizer.state_dict(),
        "episode_rewards": episode_rewards,
        "steps_done": agent.steps_done,
        "memory": agent.memory,
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved: {filename}")


def load_latest_checkpoint(agent, checkpoint_dir):
    """Load the latest checkpoint"""
    checkpoint_files = glob.glob(
        os.path.join(checkpoint_dir, "checkpoint_episode_*.pth")
    )
    if not checkpoint_files:
        print("No checkpoints found. Starting from scratch.")
        return 0, []

    # Get the latest checkpoint
    latest_checkpoint = max(
        checkpoint_files, key=lambda x: int(x.split("_episode_")[1].split(".pth")[0])
    )

    print(f"Loading checkpoint: {latest_checkpoint}")
    checkpoint = torch.load(latest_checkpoint)

    # Load model states
    agent.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
    agent.target_net.load_state_dict(checkpoint["target_net_state_dict"])
    agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Load training state
    agent.steps_done = checkpoint["steps_done"]
    agent.memory = checkpoint["memory"]

    start_episode = checkpoint["episode"] + 1
    episode_rewards = checkpoint["episode_rewards"]

    print(f"Resumed from episode {start_episode}")
    return start_episode, episode_rewards




# Create checkpoints directory
checkpoint_dir = "./checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

# Initialize agent
agent = DTQNAgent(n_features=n_features, n_actions=train_env.action_space.n)

# Load latest checkpoint if available
start_episode, episode_rewards = load_latest_checkpoint(agent, checkpoint_dir)

for i_episode in range(start_episode, NUM_EPISODES):

    start_time = time.time()
    state = train_env.reset()
    episode_reward = 0

    # Track metrics for this episode
    episode_actions = []
    episode_true_labels = []
    episode_rewards_list = []
    correct_predictions = 0
    total_predictions = 0

    # Progress bar for steps within each episode
    step_pbar = tqdm(
        range(MAX_STEPS_PER_EPISODE),
        desc=f"DTQN Episode {i_episode+1} Steps",
        leave=False,
        unit="step",
    )

    for t in step_pbar:
        # Select and perform an action - no need for .values since state is already numpy array
        action = agent.act(state)
        next_state, reward, done, true_label = train_env.step(action.item())

        # Track metrics
        episode_actions.append(action.item())
        episode_true_labels.append(true_label)
        episode_rewards_list.append(reward)

        if action.item() == true_label:
            correct_predictions += 1
        total_predictions += 1

        episode_reward += reward

        # Store the transition in memory - no need for DataFrame conversion
        agent.memory.push(state, action, next_state, reward, done)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        agent.optimize_model()

        # Update step progress bar with current reward and accuracy
        current_accuracy = (
            correct_predictions / total_predictions if total_predictions > 0 else 0
        )
        step_pbar.set_postfix(
            {
                "Reward": f"{episode_reward:.2f}",
                "Acc": f"{current_accuracy:.3f}",
                "Step": t + 1,
            }
        )

        if done:
            step_pbar.close()
            break
    else:
        step_pbar.close()

    episode_rewards.append(episode_reward)
    elapsed_time = (time.time() - start_time) / 60.0

    # Calculate episode metrics
    episode_accuracy = (
        correct_predictions / total_predictions if total_predictions > 0 else 0
    )

    # Calculate multi-class confusion matrix elements
    n_classes = len(set(episode_true_labels + episode_actions))

    # Calculate per-class metrics for training episode
    class_metrics = {}
    for class_id in range(n_classes):
        tp = sum(
            1
            for pred, true in zip(episode_actions, episode_true_labels)
            if pred == class_id and true == class_id
        )
        tn = sum(
            1
            for pred, true in zip(episode_actions, episode_true_labels)
            if pred != class_id and true != class_id
        )
        fp = sum(
            1
            for pred, true in zip(episode_actions, episode_true_labels)
            if pred == class_id and true != class_id
        )
        fn = sum(
            1
            for pred, true in zip(episode_actions, episode_true_labels)
            if pred != class_id and true == class_id
        )

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        class_metrics[class_id] = {
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }

    # Overall metrics (macro-average)
    precision = (
        np.mean([class_metrics[i]["precision"] for i in class_metrics])
        if class_metrics
        else 0
    )
    recall = (
        np.mean([class_metrics[i]["recall"] for i in class_metrics])
        if class_metrics
        else 0
    )
    f1_score = (
        np.mean([class_metrics[i]["f1_score"] for i in class_metrics])
        if class_metrics
        else 0
    )

    # False Positive Rate for normal class (class 0)
    normal_fp = class_metrics.get(0, {}).get("fp", 0)
    normal_tn = class_metrics.get(0, {}).get("tn", 0)
    fpr = normal_fp / (normal_fp + normal_tn) if (normal_fp + normal_tn) > 0 else 0

    # Calculate epsilon value for exploration tracking
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(
        -1.0 * agent.steps_done / EPS_DECAY
    )

    # Calculate moving averages
    avg_reward_10 = (
        np.mean(episode_rewards[-10:])
        if len(episode_rewards) >= 10
        else np.mean(episode_rewards)
    )
    avg_reward_100 = (
        np.mean(episode_rewards[-100:])
        if len(episode_rewards) >= 100
        else np.mean(episode_rewards)
    )

    # Get current learning rate (constant)
    current_lr = LR

    # Perform quick validation with more thorough sampling
    val_metrics = agent.quick_validation(
        test_env, max_samples=2000
    )  # Increased sample size

    print("\n\n")
    # Print comprehensive episode metrics
    print(f"Episode {i_episode + 1} Complete:")
    print(
        f"Performance: Reward={episode_reward:.2f}, Steps={t+1}, Time={elapsed_time:.2f}s"
    )

    # Training Metrics
    print("\n\n")
    print(f"TRAINING METRICS:")
    print(
        f"Accuracy: {episode_accuracy:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1_score:.3f}"
    )
    print(f"False Positive Rate (Normal): {fpr:.3f}")

    # Show per-class performance for training
    reverse_label_map = {v: k for k, v in label_map.items()}
    print(f" Per-Class Training Results:")
    for class_id in class_metrics:
        class_name = reverse_label_map.get(class_id, f"Class_{class_id}")
        cm = class_metrics[class_id]
        print(
            f"         {class_name}: P={cm['precision']:.3f}, R={cm['recall']:.3f}, F1={cm['f1_score']:.3f}"
        )

    print("\n\n")
    # Validation Metrics
    print(f"VALIDATION METRICS:")
    print(
        f"Accuracy: {val_metrics['accuracy']:.3f} | Precision: {val_metrics['precision']:.3f} | Recall: {val_metrics['recall']:.3f} | F1: {val_metrics['f1_score']:.3f}"
    )
    print(
        f"False Positive Rate (Normal): {val_metrics['fpr']:.3f} | Avg Reward: {val_metrics['reward']:.2f}"
    )
    print(f"Samples Tested: {val_metrics['samples']}")

    print("\n\n")
    # Show per-class performance for validation
    print(f"Per-Class Validation Results:")
    for class_id in val_metrics["class_metrics"]:
        class_name = reverse_label_map.get(class_id, f"Class_{class_id}")
        cm = val_metrics["class_metrics"][class_id]
        print(
            f"         {class_name}: P={cm['precision']:.3f}, R={cm['recall']:.3f}, F1={cm['f1_score']:.3f}"
        )

    # Performance comparison and overfitting detection
    acc_diff = episode_accuracy - val_metrics["accuracy"]
    f1_diff = f1_score - val_metrics["f1_score"]
    print("\n\n")
    print(f"Reward Trends: Avg(10)={avg_reward_10:.2f}, Avg(100)={avg_reward_100:.2f}")
    print(
        f"Training: ε={eps_threshold:.3f}, LR={current_lr:.6f}, Memory={len(agent.memory)}"
    )
    print("   " + "=" * 80)

    # Update the target network
    if i_episode % TARGET_UPDATE == 0:
        print(f"Updating target network at episode {i_episode+1}")
        agent.update_target_net()

    # Save checkpoint every episode
    checkpoint_filename = os.path.join(
        checkpoint_dir, f"checkpoint_episode_{i_episode}.pth"
    )
    save_checkpoint(agent, i_episode, episode_rewards, checkpoint_filename)

print("\n--- DTQN Training Complete ---")

# Save final model
final_model_path = os.path.join(checkpoint_dir, "final_model.pth")
save_checkpoint(agent, NUM_EPISODES - 1, episode_rewards, final_model_path)