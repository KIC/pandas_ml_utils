import pandas as pd
import numpy as np
import gym
from gym import spaces
from typing import Tuple, Callable, List

from ..model.features_and_Labels import FeaturesAndLabels

INIT_ACTION = -1


class RowWiseGym(gym.Env):

    def __init__(self,
                 environment: Tuple[np.ndarray, np.ndarray, np.ndarray],
                 features_and_labels: FeaturesAndLabels,
                 action_reward_functions: List[Callable[[np.ndarray], float]],
                 observation_range: Tuple[int, int],
                 reward_range: Tuple[int, int]):
        super().__init__()
        self.environment = environment
        self.reward_range = reward_range
        self.action_reward_functions = action_reward_functions

        # start at the beginning of the frame
        self.state = 0

        # define spaces
        self.action_space = spaces.Discrete(len(action_reward_functions))
        self.observation_space = spaces.Box(low=observation_range[0], high=observation_range[1],
                                            shape=features_and_labels.shape()[0], dtype=np.float16)

        # define history
        self.reward_history = []
        self.action_history = []

    metadata = {'render.modes': ['human']}

    def reset(self):
        # Reset the state of the environment to an initial state and reset history
        self.reward_history = []
        self.action_history = []
        return self.step(INIT_ACTION)[0]

    def step(self, action):
        # Execute one time step within the environment
        if action is not INIT_ACTION:
            reward = self.action_reward_functions[action](self.environment[2][self.state])
            self.reward_history.append(reward)
            self.action_history.append(action)
            self.state += 1
        else:
            reward = 0
            self.state = 0

        done = self.state >= len(self.environment[1])
        obs = self.environment[1][self.state if not done else 0]

        return obs, reward, done, {}

    def render(self, mode='human', close=False):
        print(f"reward: {sum(self.reward_history)}")

    def get_history(self):
        return pd.DataFrame({"reward_history": self.reward_history,
                             "action_history": self.action_history},
                            index=self.environment[0]).sort_index()