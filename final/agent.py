import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
from collections import deque ,namedtuple
from config import *

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

       

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
        
        # Reset to a RANDOM position in validation set for more representative sampling
        state = val_env.reset()
        
        samples_tested = 0
        
        with torch.no_grad():
            while samples_tested < max_samples and val_env.current_step < val_env.total_samples - 1:
                # Convert state to tensor - handle DataFrame properly
                if hasattr(state, 'values'):
                    state_array = state.values
                else:
                    state_array = state
                state_tensor = torch.tensor(state_array, dtype=torch.float32).unsqueeze(0).to(device)
                
                # Choose the BEST action (no exploration)
                action = self.policy_net(state_tensor).max(1)[1].item()
                
                # Take step
                next_state, reward, done, true_label = val_env.step(action)
                
                val_actions.append(action)
                val_true_labels.append(true_label)
                val_rewards.append(reward)
                
                state = next_state
                samples_tested += 1
                
                if done:
                    # Reset to another random position if we hit episode end
                    state = val_env.reset()
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

