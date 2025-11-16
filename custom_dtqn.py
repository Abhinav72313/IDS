import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_split_final_csv(
    csv_path='final.csv',
    label_col='label',
    test_size=0.15,
    val_size=0.15,
    random_state=42
):
    """
    Load a single final.csv, preprocess, encode labels, and split into
    train / validation / test sets.

    Label mapping:
        'normal' -> 0
        each attack type -> 1, 2, 3, ...

    Returns:
        x_train_scaled, x_val_scaled, x_test_scaled,
        y_train, y_val, y_test,
        n_features, label_map, scaler, feature_names
    """

    print(f"Loading dataset: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"Loaded {len(df)} rows and {df.shape[1]} columns")

    # --- Clean & normalize labels ---
    df[label_col] = df[label_col].astype(str).str.strip().str.lower()

    # --- Build label map (normal=0, others>0) ---
    all_labels = sorted(df[label_col].unique())
    attack_labels = [lbl for lbl in all_labels if lbl != 'normal']
    label_map = {'normal': 0}
    label_map.update({lbl: i + 1 for i, lbl in enumerate(attack_labels)})

    print("Label mapping:", label_map)

    # --- Encode labels ---
    df['label_encoded'] = df[label_col].map(lambda x: label_map.get(x, 0)).astype(np.int64)

    # --- Select numeric features ---
    df = df.replace(r'^\s*$', np.nan, regex=True).fillna(0)
    feature_df = df.drop(columns=[label_col, 'label_encoded'], errors='ignore').select_dtypes(include=[np.number])
    labels = df['label_encoded'].values

    # --- Split train / val / test ---
    x_train, x_temp, y_train, y_temp = train_test_split(
        feature_df, labels, test_size=(test_size + val_size),
        random_state=random_state, stratify=labels
    )
    rel_val_size = val_size / (test_size + val_size)
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=rel_val_size,
        random_state=random_state, stratify=y_temp
    )

    print(f"Split sizes -> Train: {len(x_train)}, Val: {len(x_val)}, Test: {len(x_test)}")

    # --- Scale numeric features ---
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train.astype(np.float32))
    x_val_scaled   = scaler.transform(x_val.astype(np.float32))
    x_test_scaled  = scaler.transform(x_test.astype(np.float32))

    n_features = x_train.shape[1]
    feature_names = list(x_train.columns)

    print(f"Scaling complete. Using {n_features} numeric features.")

    return (
        x_train_scaled, x_val_scaled, x_test_scaled,
        y_train, y_val, y_test,
        n_features, label_map, scaler, feature_names
    )
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
SEQUENCE_LENGTH = 10  # How many timesteps to feed the Transformer
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

# Transformer Model settings
EMBED_DIM = 128           # Embedding dimension for the Transformer
NUM_HEADS = 8             # Number of heads in Multi-Head Attention
NUM_LAYERS = 3            # Number of Transformer Encoder layers
DROPOUT_RATE = 0.1        # Dropout rate for regularization

# Training settings
NUM_EPISODES = 500        # Total number of episodes to train
MAX_STEPS_PER_EPISODE = 2000 # Max steps before an episode is "done"

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the Transition tuple for the replay memory
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))


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
        
        # Action space: 0 (normal), 1 (arp poisoning), 2 (dictionary ssh), 3 (tcp ddos)
        # Number of actions equals number of unique labels
        n_classes = len(np.unique(labels))
        self.action_space = spaces.Discrete(n_classes)
        
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
        Calculates the reward based on multi-class classification performance.
        """
        true_label = self.labels[self.current_step]
        
        # --- Multi-class Reward Function ---
        # Reward based on correct vs incorrect classification
        # Higher reward for correctly classifying attacks vs normal traffic
        
        reward = 0
        if action == true_label:
            # Correct classification
            if true_label == 0:  # normal
                reward = 1   # Correct classification of normal traffic
            else:  # any attack type
                reward = 10  # Higher reward for correctly detecting attacks
        else:
            # Incorrect classification
            if true_label == 0 and action != 0:
                reward = -20 # False Positive (classifying normal as attack)
            elif true_label != 0 and action == 0:
                reward = -50 # False Negative (missing an attack - worst case)
            else:
                reward = -10 # Wrong attack type classification
        
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


# 5. DEEP TRANSFORMER Q-NETWORK (DTQN) MODEL
class PositionalEncoding(nn.Module):
    """Adds positional information to the input sequence."""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class DTQN(nn.Module):
    """The Deep Transformer Q-Network model."""
    def __init__(self, n_features, n_actions, d_model=EMBED_DIM, n_head=NUM_HEADS, n_layers=NUM_LAYERS):
        super(DTQN, self).__init__()
        self.n_actions = n_actions
        self.d_model = d_model
        self.sequence_length = SEQUENCE_LENGTH
        
        # 1. Input Embedding layer
        self.input_embed = nn.Linear(n_features, d_model)
        
        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=SEQUENCE_LENGTH)
        
        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_head, 
            dim_feedforward=d_model * 4,
            dropout=DROPOUT_RATE,
            batch_first=False  # Keep default format for compatibility
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # 4. Output layers
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(DROPOUT_RATE)
        self.q_head = nn.Linear(d_model, n_actions)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, state_sequence):
        # state_sequence shape: (batch_size, seq_len, n_features)
        
        # 1. Embed input features
        embedded = self.input_embed(state_sequence) # (batch, seq_len, d_model)
        
        # 2. Permute for TransformerEncoder (expects seq_len, batch, d_model)
        embedded = embedded.permute(1, 0, 2) # (seq_len, batch, d_model)
        
        # 3. Add positional encoding
        embedded_pos = self.pos_encoder(embedded) # (seq_len, batch, d_model)
        
        # 4. Create causal mask for autoregressive behavior
        src_mask = nn.Transformer.generate_square_subsequent_mask(SEQUENCE_LENGTH).to(device)
        
        # 5. Pass through Transformer Encoder
        transformer_output = self.transformer_encoder(embedded_pos, src_mask) # (seq_len, batch, d_model)
        
        # 6. Get the output of the last token (most recent timestep)
        last_token_output = transformer_output[-1, :, :] # (batch, d_model)
        
        # 7. Apply layer normalization and dropout
        normalized_output = self.layer_norm(last_token_output)
        dropped_output = self.dropout(normalized_output)
        
        # 8. Generate Q-values
        q_values = self.q_head(dropped_output) # (batch, n_actions)
        
        return q_values

# 6. DTQN AGENT
class DTQNAgent:
    def __init__(self, n_features, n_actions):
        self.n_actions = n_actions
        self.steps_done = 0
        
        # Initialize policy and target networks
        self.policy_net = DTQN(n_features, n_actions).to(device)
        self.target_net = DTQN(n_features, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() # Target net is only for evaluation
        
        # Setup optimizer with learning rate scheduling
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.95)
        
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
             non_final_next_states = torch.empty(0, SEQUENCE_LENGTH, self.policy_net.input_embed.in_features).to(device)
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
        
        # Gradient clipping for transformer stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        self.scheduler.step()

    def update_target_net(self):
        """Updates the target network by copying weights from the policy network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def quick_validation(self, val_env, max_samples=1000):
        """
        Performs a quick validation on a subset of validation data.
        Returns validation metrics without full evaluation.
        """
        self.policy_net.eval()
        
        val_actions = []
        val_true_labels = []
        val_rewards = []
        
        state = val_env.reset()
        val_env.current_step = 0
        
        samples_tested = 0
        
        with torch.no_grad():
            while samples_tested < max_samples and val_env.current_step < val_env.total_samples - 1:
                # Convert state to tensor
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                
                # Choose the BEST action (no exploration)
                action = self.policy_net(state_tensor).max(1)[1].item()
                
                # Get true label
                true_label = val_env.labels[val_env.current_step]
                
                # Take step
                next_state, reward, done, _ = val_env.step(action)
                
                val_actions.append(action)
                val_true_labels.append(true_label)
                val_rewards.append(reward)
                
                state = next_state
                samples_tested += 1
                
                if done:
                    val_env.current_step = samples_tested
                    if val_env.current_step >= val_env.total_samples - 1:
                        break
        
        self.policy_net.train()
        
        # Calculate validation metrics
        val_accuracy = sum(1 for p, t in zip(val_actions, val_true_labels) if p == t) / len(val_actions) if val_actions else 0
        val_reward = sum(val_rewards) / len(val_rewards) if val_rewards else 0
        
        # Multi-class confusion matrix calculations
        n_classes = len(set(val_true_labels + val_actions))
        
        # Calculate per-class metrics
        class_metrics = {}
        for class_id in range(n_classes):
            tp = sum(1 for p, t in zip(val_actions, val_true_labels) if p == class_id and t == class_id)
            tn = sum(1 for p, t in zip(val_actions, val_true_labels) if p != class_id and t != class_id)
            fp = sum(1 for p, t in zip(val_actions, val_true_labels) if p == class_id and t != class_id)
            fn = sum(1 for p, t in zip(val_actions, val_true_labels) if p != class_id and t == class_id)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            class_metrics[class_id] = {
                'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
                'precision': precision, 'recall': recall, 'f1_score': f1
            }
        
        # Overall metrics (macro-average)
        val_precision = np.mean([class_metrics[i]['precision'] for i in class_metrics]) if class_metrics else 0
        val_recall = np.mean([class_metrics[i]['recall'] for i in class_metrics]) if class_metrics else 0
        val_f1 = np.mean([class_metrics[i]['f1_score'] for i in class_metrics]) if class_metrics else 0
        
        # False Positive Rate for normal class (class 0)
        normal_fp = class_metrics.get(0, {}).get('fp', 0)
        normal_tn = class_metrics.get(0, {}).get('tn', 0)
        val_fpr = normal_fp / (normal_fp + normal_tn) if (normal_fp + normal_tn) > 0 else 0
        
        return {
            'accuracy': val_accuracy,
            'reward': val_reward,
            'precision': val_precision,
            'recall': val_recall,
            'f1_score': val_f1,
            'fpr': val_fpr,
            'class_metrics': class_metrics,
            'samples': samples_tested
        }


# 7. EVALUATION FUNCTION
def evaluate_agent(agent, env, label_map):
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
    eval_pbar = tqdm(range(env.total_samples - 1), desc="Evaluating DTQN Agent", unit="sample")
    
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
    
    # Create class names for the report
    reverse_label_map = {v: k for k, v in label_map.items()}
    class_names = [reverse_label_map.get(i, f"Class_{i}") for i in range(len(label_map))]
    
    # Ensure we have multiple classes represented
    unique_true = set(y_true)
    unique_pred = set(y_pred)
    
    if len(unique_true) < 2:
        print("Warning: Only one class was present in the test set.")
        print(f"True labels unique: {unique_true}")
        print(f"Predicted labels unique: {unique_pred}")
    
    if len(unique_pred) < 2:
        print("Warning: Model only predicted one class.")
        print(f"Predicted labels unique: {unique_pred}")
    
    # Print detailed classification report
    try:
        report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
        print(report)
    except Exception as e:
        print(f"Error generating classification report: {e}")
        # Fallback to basic metrics
        accuracy = sum(p == t for p, t in zip(y_pred, y_true)) / len(y_pred)
        print(f"Overall Accuracy: {accuracy:.4f}")
    
    # Calculate and display confusion matrix
    from sklearn.metrics import confusion_matrix
    try:
        cm = confusion_matrix(y_true, y_pred)
        print("\nConfusion Matrix:")
        print("Predicted ->")
        print("Actual |")
        print("       v")
        for i, class_name in enumerate(class_names):
            print(f"{class_name:>15}: {cm[i]}")
    except Exception as e:
        print(f"Error generating confusion matrix: {e}")
    
    # Multi-class ROC-AUC (if possible)
    try:
        from sklearn.metrics import roc_auc_score
        if len(unique_true) > 2:
            # For multi-class, we need to use the predicted probabilities, not just the class predictions
            # Since we don't have probabilities here, we'll skip this or use a simpler approach
            print("ROC-AUC calculation skipped (requires predicted probabilities for multi-class).")
        elif len(unique_true) == 2 and len(unique_pred) == 2:
            # Binary classification case
            auc_score = roc_auc_score(y_true, y_pred)
            print(f"ROC-AUC Score: {auc_score:.4f}")
        else:
            print("ROC-AUC calculation skipped (insufficient classes).")
    except Exception as e:
        print(f"ROC-AUC calculation failed: {e}")
        
    print("--------------------------")


# 8. MAIN TRAINING BLOCK
if __name__ == "__main__":
    
    try:
        # 1. Load Data
        # This function now returns training, validation, and test data separately
        x_train, x_val, x_test, y_train, y_val, y_test, n_features, label_map, scaler, feature_names = load_and_split_final_csv("custom/final.csv")

        
        # 2. Initialize Environments
        train_env = IDSEnvironment(x_train, y_train, n_features=n_features, sequence_length=SEQUENCE_LENGTH, max_steps=MAX_STEPS_PER_EPISODE)
        val_env = IDSEnvironment(x_val, y_val, n_features=n_features, sequence_length=SEQUENCE_LENGTH, max_steps=len(x_val) - 1)
        test_env = IDSEnvironment(x_test, y_test, n_features=n_features, sequence_length=SEQUENCE_LENGTH, max_steps=len(x_test) - 1)
        
        # 3. Initialize Agent
        agent = DTQNAgent(n_features=n_features, n_actions=train_env.action_space.n)
        
        print("\n--- Starting DTQN Training ---")
        print(f"🧠 Model Architecture: Transformer with {NUM_LAYERS} layers, {NUM_HEADS} heads, {EMBED_DIM} embedding dim")
        print(f"📊 Dataset: {len(x_train)} training, {len(x_val)} validation, {len(x_test)} test samples")
        print(f"🔢 Features: {n_features}, Sequence Length: {SEQUENCE_LENGTH}")
        print(f"🏷️  Classes: {len(label_map)} classes - {list(label_map.keys())}")
        print(f"🎯 Action Space: {train_env.action_space.n} actions (one per class)")
        
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
            step_pbar = tqdm(range(MAX_STEPS_PER_EPISODE), desc=f"DTQN Episode {i_episode+1} Steps", leave=False, unit="step")
            
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
            
            # Calculate multi-class confusion matrix elements
            n_classes = len(set(episode_true_labels + episode_actions))
            
            # Calculate per-class metrics for training episode
            class_metrics = {}
            for class_id in range(n_classes):
                tp = sum(1 for pred, true in zip(episode_actions, episode_true_labels) if pred == class_id and true == class_id)
                tn = sum(1 for pred, true in zip(episode_actions, episode_true_labels) if pred != class_id and true != class_id)
                fp = sum(1 for pred, true in zip(episode_actions, episode_true_labels) if pred == class_id and true != class_id)
                fn = sum(1 for pred, true in zip(episode_actions, episode_true_labels) if pred != class_id and true == class_id)
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                class_metrics[class_id] = {
                    'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
                    'precision': precision, 'recall': recall, 'f1_score': f1
                }
            
            # Overall metrics (macro-average)
            precision = np.mean([class_metrics[i]['precision'] for i in class_metrics]) if class_metrics else 0
            recall = np.mean([class_metrics[i]['recall'] for i in class_metrics]) if class_metrics else 0
            f1_score = np.mean([class_metrics[i]['f1_score'] for i in class_metrics]) if class_metrics else 0
            
            # False Positive Rate for normal class (class 0)
            normal_fp = class_metrics.get(0, {}).get('fp', 0)
            normal_tn = class_metrics.get(0, {}).get('tn', 0)
            fpr = normal_fp / (normal_fp + normal_tn) if (normal_fp + normal_tn) > 0 else 0
            
            # Calculate epsilon value for exploration tracking
            eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * agent.steps_done / EPS_DECAY)
            
            # Calculate moving averages
            avg_reward_10 = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards)
            avg_reward_100 = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
            
            # Get current learning rate
            current_lr = agent.scheduler.get_last_lr()[0] if hasattr(agent.scheduler, 'get_last_lr') else LR
            
            # Perform quick validation
            val_metrics = agent.quick_validation(val_env, max_samples=1000)
            
            # Print comprehensive episode metrics
            print(f"✅ Episode {i_episode + 1} Complete:")
            print(f"   📊 Performance: Reward={episode_reward:.2f}, Steps={t+1}, Time={elapsed_time:.2f}s")
            
            # Training Metrics
            print(f"   🏋️  TRAINING METRICS:")
            print(f"      🎯 Accuracy: {episode_accuracy:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1_score:.3f}")
            print(f"      ⚠️  False Positive Rate (Normal): {fpr:.3f}")
            
            # Show per-class performance for training
            reverse_label_map = {v: k for k, v in label_map.items()}
            print(f"      📊 Per-Class Training Results:")
            for class_id in class_metrics:
                class_name = reverse_label_map.get(class_id, f"Class_{class_id}")
                cm = class_metrics[class_id]
                print(f"         {class_name}: P={cm['precision']:.3f}, R={cm['recall']:.3f}, F1={cm['f1_score']:.3f}")
            
            # Validation Metrics
            print(f"   🔍 VALIDATION METRICS:")
            print(f"      🎯 Accuracy: {val_metrics['accuracy']:.3f} | Precision: {val_metrics['precision']:.3f} | Recall: {val_metrics['recall']:.3f} | F1: {val_metrics['f1_score']:.3f}")
            print(f"      ⚠️  False Positive Rate (Normal): {val_metrics['fpr']:.3f} | Avg Reward: {val_metrics['reward']:.2f}")
            print(f"      📝 Samples Tested: {val_metrics['samples']}")
            
            # Show per-class performance for validation
            print(f"      📊 Per-Class Validation Results:")
            for class_id in val_metrics['class_metrics']:
                class_name = reverse_label_map.get(class_id, f"Class_{class_id}")
                cm = val_metrics['class_metrics'][class_id]
                print(f"         {class_name}: P={cm['precision']:.3f}, R={cm['recall']:.3f}, F1={cm['f1_score']:.3f}")
            
            # Performance comparison and overfitting detection
            acc_diff = episode_accuracy - val_metrics['accuracy']
            f1_diff = f1_score - val_metrics['f1_score']
            
            print(f"   📊 PERFORMANCE COMPARISON:")
            print(f"      🔍 Train vs Val Accuracy: {episode_accuracy:.3f} vs {val_metrics['accuracy']:.3f} (Δ={acc_diff:+.3f})")
            print(f"      🔍 Train vs Val F1-Score: {f1_score:.3f} vs {val_metrics['f1_score']:.3f} (Δ={f1_diff:+.3f})")
            
            # Overfitting warning
            if acc_diff > 0.1 or f1_diff > 0.1:
                print(f"      ⚠️  WARNING: Potential overfitting detected! (Δ>0.1)")
            elif acc_diff < -0.05:
                print(f"      � INFO: Validation performance better than training (unusual but possible)")
            else:
                print(f"      ✅ Good generalization gap")
            
            print(f"   �📊 Reward Trends: Avg(10)={avg_reward_10:.2f}, Avg(100)={avg_reward_100:.2f}")
            print(f"   🔧 Training: ε={eps_threshold:.3f}, LR={current_lr:.6f}, Memory={len(agent.memory)}")
            print("   " + "="*80)
            
            # Update the target network
            if i_episode % TARGET_UPDATE == 0:
                print(f"🔄 Updating target network at episode {i_episode+1}")
                agent.update_target_net()

        print("\n--- DTQN Training Complete ---")
        
        # 9. Final Evaluation
        evaluate_agent(agent, test_env, label_map)

    except FileNotFoundError as e:
        print(f"\nFATAL ERROR: {e}")
        print("Please ensure the 'CICIOT23' folder exists in the same directory as the script,")
        print("and that it contains 'train', 'validation', and 'test' subfolders with CSV files.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
