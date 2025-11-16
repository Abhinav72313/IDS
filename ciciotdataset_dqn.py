# 1. IMPORTS
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
from gymnasium import spaces
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from collections import deque, namedtuple
import random
import math
import time
import os
import glob
from tqdm import tqdm

# --- Configuration & Hyperparameters ---
# Environment settings
SEQUENCE_LENGTH = 10  # How many timesteps to feed the network (for flattening)
NUM_FEATURES_GUESS = 46 # Initial guess for CICIoT2023 features

# DQN Agent settings
BATCH_SIZE = 128          # How many samples to train on at once
GAMMA = 0.99              # Discount factor for future rewards
EPS_START = 0.9           # Starting value of epsilon (exploration)
EPS_END = 0.05            # Minimum value of epsilon
EPS_DECAY = 10000         # How fast epsilon decays
TARGET_UPDATE = 10        # How often to update the target network (in episodes)
REPLAY_MEMORY_SIZE = 50000 # Max size of the replay buffer
LR = 0.0001               # Learning rate for the optimizer

# Model settings
HIDDEN_DIM_1 = 512        # First hidden layer dimension
HIDDEN_DIM_2 = 256        # Second hidden layer dimension
HIDDEN_DIM_3 = 128        # Third hidden layer dimension

# Training settings
NUM_EPISODES = 500        # Total number of episodes to train
MAX_STEPS_PER_EPISODE = 2000 # Max steps before an episode is "done"

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the Transition tuple for the replay memory
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))


# 2. DATA PREPARATION
def load_csvs_from_folder(folder_path):
    """Helper function to load all CSVs from a folder into one DataFrame."""
    all_files = glob.glob(os.path.join(folder_path, "*.csv"))
    if not all_files:
        raise FileNotFoundError(f"No CSV files found in folder: {folder_path}")
        
    df_list = []
    print(f"Loading files from {folder_path}...")
    for f in all_files:
        try:
            df_list.append(pd.read_csv(f))
        except Exception as e:
            print(f"Error loading file {f}: {e}")
    
    if not df_list:
        raise ValueError(f"All CSV files in {folder_path} failed to load.")
        
    return pd.concat(df_list, ignore_index=True)

def load_and_preprocess_data(base_path='CICIOT23'):
    """
    Loads, preprocesses, and splits data from the CICIOT23 folder structure.
    
    Returns:
    x_train_scaled, x_test_scaled, y_train, y_test, n_features
    """
    print("Loading and preprocessing data from CICIOT23 folders...")
    
    # 1. Define paths
    train_path = os.path.join(base_path, 'train')
    val_path = os.path.join(base_path, 'validation')
    test_path = os.path.join(base_path, 'test')
    
    # 2. Load data
    df_train = load_csvs_from_folder(train_path)
    df_val = load_csvs_from_folder(val_path)
    df_test = load_csvs_from_folder(test_path)
    
    # Combine train and validation sets for the training environment
    df_train_full = pd.concat([df_train, df_val], ignore_index=True)
    
    print(f"Loaded {len(df_train_full)} training samples and {len(df_test)} test samples.")
    
    # 3. Handle any missing values (e.g., fill with 0)
    # Note: A more sophisticated strategy might be needed.
    df_train_full = df_train_full.fillna(0)
    df_test = df_test.fillna(0)
    
    # 4. Encode the 'label' column
    # 0 for 'BenignTraffic', 1 for all attacks
    y_train = (df_train_full['label'] != 'BenignTraffic').astype(int)
    y_test = (df_test['label'] != 'BenignTraffic').astype(int)
    
    # 5. Separate features (X)
    x_train_df = df_train_full.drop('label', axis=1)
    x_test_df = df_test.drop('label', axis=1)

    # Handle potential non-numeric columns that can't be scaled
    # (e.g., 'timestamp' or 'flow_id' if they exist)
    x_train_df = x_train_df.select_dtypes(include=[np.number])
    x_test_df = x_test_df.select_dtypes(include=[np.number])
    
    # Ensure columns are in the same order
    x_test_df = x_test_df[x_train_df.columns]
    
    # Get the *actual* number of features
    n_features = x_train_df.shape[1]
    
    # Check if we have the expected number of features
    if n_features != NUM_FEATURES_GUESS:
        print(f"Warning: Expected {NUM_FEATURES_GUESS} features, but found {n_features}.")
    
    print(f"Using {n_features} features.")
    
    # 6. Scale the features (X)
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train_df)
    x_test_scaled = scaler.transform(x_test_df) # Use transform, not fit_transform
    
    print("Data scaling complete.")
    
    # 7. Return numpy arrays and the number of features
    return x_train_scaled, x_test_scaled, y_train.values, y_test.values, n_features


# 3. CUSTOM GYM ENVIRONMENT
class IDSEnvironment(gym.Env):
    """
    Custom Gym environment for the IDS.
    It feeds the agent sequences of network flows from the dataset.
    """
    def __init__(self, features, labels, n_features, sequence_length=SEQUENCE_LENGTH, max_steps=MAX_STEPS_PER_EPISODE):
        super(IDSEnvironment, self).__init__()
        
        self.features = features
        self.labels = labels
        self.n_features = n_features # Store the actual number of features
        self.sequence_length = sequence_length
        self.max_steps = max_steps
        self.current_step = 0
        self.total_samples = len(features)
        
        # Action space: 0 (classify as Benign), 1 (classify as Attack)
        self.action_space = spaces.Discrete(2)
        
        # Observation space: A sequence of flows
        # Shape: (SEQUENCE_LENGTH, n_features)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.sequence_length, self.n_features), 
            dtype=np.float32
        )
        
    def _get_state(self):
        """
        Gets the current state (sequence of flows).
        Pads with zeros if at the beginning of the dataset.
        """
        end_idx = self.current_step + 1
        start_idx = max(0, end_idx - self.sequence_length)
        
        state = self.features[start_idx:end_idx]
        
        # Pad with zeros if we don't have enough history
        if len(state) < self.sequence_length:
            padding = np.zeros((self.sequence_length - len(state), self.n_features), dtype=np.float32)
            state = np.vstack((padding, state))
            
        return state.astype(np.float32)

    def reset(self):
        """
        Resets the environment to a new random starting point.
        """
        # Start at a random point in the dataset (leaving room for a full episode)
        if (self.total_samples - self.max_steps - 1) <= 0:
            # Handle case where dataset is smaller than max_steps
            self.start_point = 0
        else:
            self.start_point = random.randint(0, self.total_samples - self.max_steps - 1)
        self.current_step = self.start_point
        self.steps_taken_in_episode = 0
        return self._get_state()

    def step(self, action):
        """
        Takes an action in the environment.
        Calculates the reward based on the synopsis goals.
        """
        true_label = self.labels[self.current_step]
        
        # --- Define the Reward Function ---
        # This is critical for your project's success.
        # We heavily penalize False Positives and False Negatives.
        
        reward = 0
        if action == true_label:
            if action == 1:
                reward = 10  # True Positive (Correctly caught an attack)
            else:
                reward = 1   # True Negative (Correctly ignored benign traffic)
        else:
            if action == 1:
                reward = -20 # False Positive (Goal: <0.5% FPR)
            else:
                reward = -50 # False Negative (Worst case: missed an attack)
        
        # Move to the next state
        self.current_step += 1
        self.steps_taken_in_episode += 1
        
        next_state = self._get_state()
        
        # Check if episode is done
        done = (self.steps_taken_in_episode >= self.max_steps or 
                self.current_step >= self.total_samples - 1)
        
        return next_state, reward, done, {}


# 4. REPLAY MEMORY
class ReplayMemory(object):
    """A cyclic buffer to store transitions for experience replay."""
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition."""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """Sample a batch of transitions."""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# 5. DEEP Q-NETWORK (DQN) MODEL
class DQN(nn.Module):
    """The Deep Q-Network model."""
    def __init__(self, n_features, n_actions, sequence_length=SEQUENCE_LENGTH):
        super(DQN, self).__init__()
        self.n_actions = n_actions
        self.sequence_length = sequence_length
        
        # Calculate input dimension (flattened sequence)
        input_dim = n_features * sequence_length
        
        # Define the network layers
        self.fc1 = nn.Linear(input_dim, HIDDEN_DIM_1)
        self.fc2 = nn.Linear(HIDDEN_DIM_1, HIDDEN_DIM_2)
        self.fc3 = nn.Linear(HIDDEN_DIM_2, HIDDEN_DIM_3)
        self.fc4 = nn.Linear(HIDDEN_DIM_3, n_actions)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, state_sequence):
        # state_sequence shape: (batch_size, seq_len, n_features)
        
        # Flatten the sequence
        batch_size = state_sequence.size(0)
        x = state_sequence.reshape(batch_size, -1)  # (batch_size, seq_len * n_features)
        
        # Pass through fully connected layers with ReLU activation
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        
        # Output Q-values (no activation on final layer)
        q_values = self.fc4(x)
        
        return q_values

# 6. DQN AGENT
class DQNAgent:
    def __init__(self, n_features, n_actions):
        self.n_actions = n_actions
        self.steps_done = 0
        
        # Initialize policy and target networks
        self.policy_net = DQN(n_features, n_actions).to(device)
        self.target_net = DQN(n_features, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() # Target net is only for evaluation
        
        # Setup optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        
        # Setup replay memory
        self.memory = ReplayMemory(REPLAY_MEMORY_SIZE)

    def act(self, state):
        """Selects an action using an epsilon-greedy policy."""
        # Calculate epsilon
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                        math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        
        # Exploration
        if random.random() < eps_threshold:
             # Take a random action
             return torch.tensor([[random.randrange(self.n_actions)]], device=device, dtype=torch.long)
        # Exploitation
        else:
            with torch.no_grad():
                # Convert state to tensor: (1, seq_len, n_features)
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                
                # Get Q-values from the policy network
                q_values = self.policy_net(state_tensor)
                
                # Select action with the highest Q-value
                return q_values.max(1)[1].view(1, 1)

    def optimize_model(self):
        """Performs one step of optimization on the policy network."""
        if len(self.memory) < BATCH_SIZE:
            return  # Not enough samples in memory
            
        # Sample a batch from memory
        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        
        # Create tensors for states, actions, rewards
        # 'done' is a boolean, 'next_state' can be None
        
        # Non-final next states
        non_final_mask = torch.tensor(
            tuple(s is not None for s in batch.next_state), 
            device=device, dtype=torch.bool
        )
        
        # Create a list of non-None next_states before concatenation
        non_final_next_states_list = [torch.tensor(s, dtype=torch.float32).unsqueeze(0) for s in batch.next_state if s is not None]
        
        if not non_final_next_states_list:
             # Handle edge case where all next_states in batch are None
             # For DQN, we need (batch_size, seq_len, n_features) - get n_features from fc1 input size
             n_features = self.policy_net.fc1.in_features // SEQUENCE_LENGTH
             non_final_next_states = torch.empty(0, SEQUENCE_LENGTH, n_features).to(device)
        else:
            non_final_next_states = torch.cat(non_final_next_states_list).to(device)

        
        state_batch = torch.cat(
             [torch.tensor(s, dtype=torch.float32).unsqueeze(0) for s in batch.state]
        ).to(device)
        
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat([torch.tensor([r], dtype=torch.float32) for r in batch.reward]).to(device)
        done_batch = torch.cat([torch.tensor([d], dtype=torch.bool) for d in batch.done]).to(device)

        # 1. Get Q(s, a)
        # Get Q-values from policy_net, then select the Q-value for the action taken
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # 2. Get V(s') = max_a' Q_target(s', a')
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        if non_final_next_states.size(0) > 0:
            with torch.no_grad():
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        
        # 3. Calculate expected Q-values (Bellman equation)
        # Q_expected = r + gamma * V(s')
        # If terminal state, Q_expected = r
        next_state_values[done_batch.squeeze()] = 0.0 # Set V(s') to 0 for terminal states
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch.squeeze()

        # 4. Calculate Loss
        # Huber Loss (Smooth L1 Loss) is robust to outliers
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        
        # 5. Optimize
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1) # Gradient clipping
        self.optimizer.step()

    def update_target_net(self):
        """Updates the target network by copying weights from the policy network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())


# 7. EVALUATION FUNCTION
def evaluate_agent(agent, env):
    """
    Evaluates the trained agent and prints a classification report.
    """
    print("\n--- Evaluating Agent ---")
    agent.policy_net.eval() # Set model to evaluation mode
    
    y_true = []
    y_pred = []
    
    # We will evaluate on the *entire* test environment
    # Note: This is different from training. We don't stop after MAX_STEPS.
    
    state = env.reset()
    # Reset to the *very beginning* for a full evaluation
    env.current_step = 0 
    
    # Progress bar for evaluation
    eval_pbar = tqdm(range(env.total_samples - 1), desc="Evaluating Agent", unit="sample")
    
    for step in eval_pbar:
        with torch.no_grad():
            # Convert state to tensor
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            
            # Choose the BEST action (no exploration)
            action = agent.policy_net(state_tensor).max(1)[1].item()
            
            # Get true label for this step
            true_label = env.labels[env.current_step]
            
            y_pred.append(action)
            y_true.append(true_label)
            
            # Take step (we ignore reward/done here)
            next_state, _, done, _ = env.step(action)
            state = next_state
            
            # Update progress bar
            if step > 0 and step % 1000 == 0:
                accuracy = sum(p == t for p, t in zip(y_pred, y_true)) / len(y_pred)
                eval_pbar.set_postfix({"Accuracy": f"{accuracy:.3f}", "Sample": step+1})
            
            if done:
                # This 'done' is from the env's step counter,
                # but we want to eval all data
                env.current_step = step + 1 # Manual override
                if env.current_step >= env.total_samples - 1:
                    break # Really done
    
    eval_pbar.close()
    agent.policy_net.train() # Set model back to train mode
    
    # --- Print Evaluation Metrics ---
    print("\n--- Evaluation Results ---")
    
    # Ensure we have both classes represented, otherwise classification_report fails
    if len(set(y_true)) < 2 or len(set(y_pred)) < 2:
        print("Warning: Only one class was predicted or present in the test set.")
        print(f"True labels unique: {np.unique(y_true)}")
        print(f"Predicted labels unique: {np.unique(y_pred)}")
        # Still print report if possible, it might just be one-sided
        if len(set(y_true)) > 0 and len(set(y_pred)) > 0:
             report = classification_report(y_true, y_pred, target_names=['Benign (0)', 'Attack (1)'], zero_division=0)
             print(report)
        return

    report = classification_report(y_true, y_pred, target_names=['Benign (0)', 'Attack (1)'], zero_division=0)
    print(report)
    
    try:
        auc_score = roc_auc_score(y_true, y_pred)
        print(f"ROC-AUC Score: {auc_score:.4f}")
    except ValueError:
        print("ROC-AUC calculation failed (likely only one class present).")
        
    print("--------------------------")


# 8. MAIN TRAINING BLOCK
if __name__ == "__main__":
    
    try:
        # 1. Load Data
        # This function now returns the actual number of features
        x_train, x_test, y_train, y_test, n_features = load_and_preprocess_data()
        
        # 2. Initialize Environments
        train_env = IDSEnvironment(x_train, y_train, n_features=n_features, sequence_length=SEQUENCE_LENGTH, max_steps=MAX_STEPS_PER_EPISODE)
        # Set max_steps for test_env to its full length for a complete evaluation
        test_env = IDSEnvironment(x_test, y_test, n_features=n_features, sequence_length=SEQUENCE_LENGTH, max_steps=len(x_test) - 1)
        
        # 3. Initialize Agent
        agent = DQNAgent(n_features=n_features, n_actions=train_env.action_space.n)
        
        print("\n--- Starting Training ---")
        
        episode_rewards = []
        
        for i_episode in range(NUM_EPISODES):
            # Print episode value at the start of each episode
            print(f"\n🚀 Starting Episode {i_episode + 1}/{NUM_EPISODES}")
            
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
            step_pbar = tqdm(range(MAX_STEPS_PER_EPISODE), desc=f"Episode {i_episode+1} Steps", leave=False, unit="step")
            
            for t in step_pbar:
                # Select and perform an action
                action = agent.act(state)
                next_state, reward, done, _ = train_env.step(action.item())
                
                # Get the true label for metrics tracking
                true_label = train_env.labels[train_env.current_step - 1]  # -1 because step() increments current_step
                
                # Track metrics
                episode_actions.append(action.item())
                episode_true_labels.append(true_label)
                episode_rewards_list.append(reward)
                
                if action.item() == true_label:
                    correct_predictions += 1
                total_predictions += 1
                
                episode_reward += reward
                
                # Store the transition in memory
                # If done, next_state is None
                agent.memory.push(state, action, next_state, reward, done)
                
                # Move to the next state
                state = next_state
                
                # Perform one step of the optimization (on the policy network)
                agent.optimize_model()
                
                # Update step progress bar with current reward and accuracy
                current_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
                step_pbar.set_postfix({
                    "Reward": f"{episode_reward:.2f}", 
                    "Acc": f"{current_accuracy:.3f}",
                    "Step": t+1
                })
                
                if done:
                    step_pbar.close()
                    break
            else:
                step_pbar.close()
                    
            episode_rewards.append(episode_reward)
            elapsed_time = time.time() - start_time
            
            # Calculate episode metrics
            episode_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
            
            # Calculate confusion matrix elements
            tp = sum(1 for pred, true in zip(episode_actions, episode_true_labels) if pred == 1 and true == 1)
            tn = sum(1 for pred, true in zip(episode_actions, episode_true_labels) if pred == 0 and true == 0)
            fp = sum(1 for pred, true in zip(episode_actions, episode_true_labels) if pred == 1 and true == 0)
            fn = sum(1 for pred, true in zip(episode_actions, episode_true_labels) if pred == 0 and true == 1)
            
            # Calculate precision, recall, F1-score
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # False Positive Rate
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            # Calculate epsilon value for exploration tracking
            eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * agent.steps_done / EPS_DECAY)
            
            # Calculate moving averages
            avg_reward_10 = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards)
            avg_reward_100 = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
            
            # Print comprehensive episode metrics
            print(f"✅ Episode {i_episode + 1} Complete:")
            print(f"   📊 Performance: Reward={episode_reward:.2f}, Steps={t+1}, Time={elapsed_time:.2f}s")
            print(f"   🎯 Accuracy: {episode_accuracy:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1_score:.3f}")
            print(f"   📈 Confusion Matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
            print(f"   ⚠️  False Positive Rate: {fpr:.3f}")
            print(f"   📊 Reward Trends: Avg(10)={avg_reward_10:.2f}, Avg(100)={avg_reward_100:.2f}")
            print(f"   🔍 Exploration: ε={eps_threshold:.3f}, Memory Size={len(agent.memory)}")
            print("   " + "="*60)
            
            # Update the target network
            if i_episode % TARGET_UPDATE == 0:
                print(f"*** Updating target network at episode {i_episode+1} ***")
                agent.update_target_net()

        print("\n--- Training Complete ---")
        
        # 9. Final Evaluation
        evaluate_agent(agent, test_env)

    except FileNotFoundError as e:
        print(f"\nFATAL ERROR: {e}")
        print("Please ensure the 'CICIOT23' folder exists in the same directory as the script,")
        print("and that it contains 'train', 'validation', and 'test' subfolders with CSV files.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

