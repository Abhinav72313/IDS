import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from config import *

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
        
        return next_state, reward, done, true_label
