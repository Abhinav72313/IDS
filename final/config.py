import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# --- Configuration & Hyperparameters ---
# Environment settings
SEQUENCE_LENGTH = 100  # How many timesteps to feed the Transformer

# DQN Agent settings
BATCH_SIZE = 128          # How many samples to train on at once
GAMMA = 0.99              # Discount factor for future rewards
EPS_START = 0.9           # Starting value of epsilon (exploration)
EPS_END = 0.05            # Minimum value of epsilon
EPS_DECAY = 10000         # How fast epsilon decays
TARGET_UPDATE = 5        # How often to update the target network (in episodes)
REPLAY_MEMORY_SIZE = 50000 # Max size of the replay buffer
LR = 0.0001               # Learning rate for the optimizer

# Transformer Model settings
EMBED_DIM = 128           # Embedding dimension for the Transformer
NUM_HEADS = 8             # Number of heads in Multi-Head Attention
NUM_LAYERS = 3            # Number of Transformer Encoder layers
DROPOUT_RATE = 0.1        # Dropout rate for regularization

# Training settings
NUM_EPISODES = 50        # Total number of episodes to train
MAX_STEPS_PER_EPISODE = 2000 # Max steps before an episode is "done"

