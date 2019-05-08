import random
import numpy as np

class Slot_Machine():
    #Slot_Machineクラスでmax_episode_steoとslot_countを定義。
    def __init__(self, win_probs, max_episode_step=30):
        self.win_probs = win_probs
        self.max_episode_step = max_episode_step
        self.slot_count = 0

    def __len__(self):
        return len(self.win_probs)

    def reset(self):
        self.slot_count = 0

    #スロットを一回回す関数。actionには選ばれたマシンの番号が渡される。
    def step(self, action):
        final = self.max_episode_step - 1
        if self.slot_count > final:
            raise Exception("The step count exceeded maximum. \
                            Please reset env.")
        else:
            done = True if self.slot_count == final else False

        if action >= len(self.win_probs):
            raise Exception("The No.{} machine doesn't exist.".format(action))
        else:
            win_prob = self.win_probs[action]
            if random.random()<win_prob:
                reward = 1.0
            else:
                reward = 0.0
            self.slot_count += 1
            return reward, done

class EpsilonGreedyAgent():

    def __init__(self, epsilon):
        self.epsilon = epsilon
        self.V = [] #Vには各スロットでこれまでに当たった確率が格納されていく。

    #どのスロットを回すかを選ぶ関数
    def policy(self):
        machines = range(len(self.V))
        #一様分布がεより小さくなったときスロットを選びなおす、つまり確率εでスロットを選びなおす。
        if random.random() < self.epsilon:
            return random.choice(machines)
        #確率が一番大きいもの、つまり一番当たるスロットを選ぶ
        else:
            return np.argmax(self.V)

    def play(self, env):
        # Initialize estimation.
        N = [0] * len(env)
        self.V = [0] * len(env)

        env.reset()
        done = False
        rewards = []
        while not done:
            selected_machine = self.policy()
            reward, done = env.step(selected_machine)
            rewards.append(reward)

            n = N[selected_machine]
            win_average = self.V[selected_machine]
            new_average = (win_average * n + reward) / (n + 1)
            N[selected_machine] += 1
            self.V[selected_machine] = new_average

        return rewards

if __name__ == "__main__":
    import pandas as pd
    import matplotlib.pyplot as plt

    def main(machine_num,epsilon):
        #各マシンの当たる確率をランダム生成
        win_probs = np.random.rand(machine_num)
        win_probs = np.round(win_probs,decimals=1)
        env = Slot_Machine(win_probs)

        game_steps = list(range(10,310,10))
        result = {}
        agent = EpsilonGreedyAgent(epsilon)
        means = []
        #スロットを10回回した時、20回回した時、…の結果を調べるためにSlot_Machineに様々なmax_episode_stepsを渡す。
        for s in game_steps:
            env.max_episode_steps = s
            rewards = agent.play(env)
            means.append(np.mean(rewards))
        result["epsilon={}".format(epsilon)] = means
        result["machine slot count"] = game_steps
        result = pd.DataFrame(result)
        result.set_index("machine slot count", drop=True, inplace=True)
        result.plot.line(figsize=(10,5))
        plt.show()

main(10,0.1)
