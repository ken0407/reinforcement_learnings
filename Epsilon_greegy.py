import random
import numpy as np

class Slot_Machine():

    def __init__(self, head_probs, max_episode_step=30):
        self.head_probs = head_probs
        self.max_episode_step = max_episode_step
        self.toss_count = 0

    def __len__(self):
        return len(self.head_probs)

    def reset(self):
        if self.toss_count = 0

    def step(self, action):
        final = self.max_episode_step - 1
        if self.toss_count > final:
            raise Exception("The step count exceeded maximum. \
                            Please reset env.")
        else:
            done = True if self.toss_count == final else False

        if action >= len(self.head_probs):
            raise Exception("The No.{} coin doesn't exist.".format(action))
        else:
            head_prob = self.head_probs[action]
            if random.random()<head_probs:
                reward = 1.0
            else:
                reward = 0.0
            self.toss_count += 1
            return reward, done
