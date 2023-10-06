import os
import cv2
import json
import glob
import time
import shutil
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from models import DeepQNet
from scipy import interpolate
from collections import deque
from collections import namedtuple

Transition = namedtuple("Transition", ["state", "action", "next_state", "reward", "displacement", "velocity"])
Hyperparameters = namedtuple("Hyperparameters", ["lr", "batch_size", "gamma", "alpha", "target_update", "n_iterations"])


# Create command line argument parser
def create_parser():
    parser = argparse.ArgumentParser(description="Offline RL training")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--alpha", type=float, default=1)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--target_update", type=int, default=100)
    parser.add_argument("--n_iterations", type=int, default=5000)
    parser.add_argument("--etype", type=str, default="real")
    parser.add_argument("--weights", type=str, default="models/agent/dqn.pth")
    parser.add_argument("--n_frames", type=int, default=5)
    parser.add_argument("--n_actions", type=int, default=3)
    parser.add_argument("--rl_loss", type=str, default="smoothl1")
    parser.add_argument("--reg_loss", type=str, default="l2")
    parser.add_argument("--capacity", type=int, default=10000)
    parser.add_argument("--data_path", type=str, default="episodes")
    parser.add_argument("--save_path", type=str, default="models/agent")
    parser.add_argument("--checkpoint", type=int, default=1000)
    return parser


# Get a timestamp
def millis():
    return round(time.time() * 1000)


# Create a directory
def create_dir(folder, remove=True):
    if remove:
        try:
            shutil.rmtree(folder)
        except FileNotFoundError:
            pass
        os.makedirs(folder, exist_ok=True)


# Estimate the displacement and velocity of a saved episode
def estimate_numeric(path, model):
    gt_displacement = []
    gt_velocity = []
    displacement = []
    velocity = []
    with open(path, "r") as f:
        episode = json.load(f)
        transitions = episode["transitions"]
        for transition in transitions:
            state_paths = [transition[0]]
            state = transform_states(state_paths, torch.device("cpu"))
            _, state_numeric = model(state)
            displacement.append(state_numeric.squeeze(0)[0].item())
            gt_displacement.append(transition[4])
            gt_velocity.append(transition[5])
            velocity.append(state_numeric.squeeze(0)[1].item())
    plt.plot(displacement)
    plt.plot(gt_displacement)
    plt.show()
    plt.plot(velocity)
    plt.plot(gt_velocity)
    plt.show()

# Load the reference curves
def load_reference():
    with open("reference/velocity.json", "r") as f1:
        tck_v_json = json.load(f1)
    with open("reference/displacement.json", "r") as f2:
        tck_d_json = json.load(f2)
    tck_v = (np.array(tck_v_json[0]), np.array(tck_v_json[1]), tck_v_json[2])
    tck_d = (np.array(tck_d_json[0]), np.array(tck_d_json[1]), tck_d_json[2])
    return tck_v, tck_d


# Get the reference curve data at given time instance
def get_reference(tck_d, tck_v, t):
    if t < 9.5:
        v = interpolate.splev(t, tck_v, der=0)
        d = interpolate.splev(t, tck_d, der=0)
    else:
        v = 0
        d = 2.62
    return np.array([d, v], dtype=np.float64)


# Transform an action to an integer
def action2int(action, step=1):
    inverse_mapping = {0: 0, step: 1, -step: 2}
    return 3 ** 2 * inverse_mapping[action[0]] + 3 * inverse_mapping[action[1]] + inverse_mapping[action[2]]


# Buffer to sample previous experience
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    # Push back a new transition
    def push(self, transition):
        self.memory.append(transition)

    # Randomly sample a batch of transitions
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    # Get the current length of the memory buffer
    def __len__(self):
        return len(self.memory)


# Load the frames of a batch
def transform_states(states, device):
    tstates = []
    for state in states:
        tstate = []
        for path in state:
            frame = cv2.imread(path) / 255
            tstate.append(np.transpose(frame, (2, 0, 1)))
            if frame.shape[0] != 512:
                print(path)
        tstate = np.array(tstate)
        tstates.append(tstate)
    return torch.from_numpy(np.array(tstates)).to(device)


# Transform the actions of a batch
def transform_actions(actions, device):
    mapping = {0: 0, 3: 1, -3: 1}
    tactions = []
    for action in actions:
        taction = torch.tensor([mapping[action[0]], mapping[action[1]], mapping[action[2]]]).unsqueeze(0).to(device)
        tactions.append(taction)
    return tuple(tactions)


# Transform the numeric data of a batch
def transform_numeric(numerics, device):
    tnumerics = []
    for numeric in numerics:
        tnumeric = torch.tensor([numeric]).to(device)
        tnumerics.append(tnumeric)
    return tuple(tnumerics)


# Offline Reinforcement Learning training algorithm
class RLTrainer():
    def __init__(self,
                 etype,
                 weights,
                 rl_loss,
                 reg_loss,
                 capacity,
                 data_path,
                 save_path,
                 checkpoint,
                 n_frames,
                 n_actions,
                 hyperparameters):
        self.etype = etype
        self.weights = weights
        self.rl_loss = rl_loss
        self.reg_loss = reg_loss
        self.n_frames = n_frames
        self.n_actions = n_actions
        self.capacity = capacity
        self.data_path = data_path
        self.save_path = save_path
        self.checkpoint = checkpoint
        self.timestamp = str(millis())
        self.hyperparameters = hyperparameters
        self.memory = ReplayMemory(capacity)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DeepQNet(n_frames=n_frames, n_actions=n_actions).double()
        self.target_net = DeepQNet(n_frames=n_frames, n_actions=n_actions).double()
        self.policy_net.to(self.device)
        self.target_net.to(self.device)
        self.target_net.eval()
        self.initialize_weights()
        self.copy_weights()
        self.optimizer = optim.Adam(self.policy_net.parameters(), hyperparameters.lr)
        create_dir(os.path.join(self.save_path, self.timestamp))

    # Copy the weights of the policy network to the target network
    def copy_weights(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    # Save the weights of the Q-network
    def save_weights(self, i):
        model_path = os.path.join(self.save_path, self.timestamp, "dqn_" + str(i) + ".pth")
        torch.save(self.policy_net.state_dict(), model_path)

    # Weight initialization
    def initialize_weights(self):
        self.policy_net.load_state_dict(torch.load(self.weights, map_location=torch.device("cpu")))
        for n, p in self.policy_net.named_parameters():
            if "block" in n:
                p.requires_grad = False
            if "fc_ss" in n:
                p.requires_grad = False

    # Load real or generated data
    def load_episodes(self, etype):
        real_episodes = glob.glob(os.path.join(self.data_path, "training", "real", "*.json"))
        generated_episodes = glob.glob(os.path.join(self.data_path, "training", "generated", "*.json"))
        untrained_episodes = glob.glob(os.path.join(self.data_path, "training", "untrained", "*.json"))
        switcher = {"real": real_episodes, "generated": generated_episodes, "untrained": untrained_episodes, "all": real_episodes + generated_episodes}
        return switcher.get(etype, real_episodes)

    # Load the data in the replay memory buffer
    def load_memory(self):
        episodes = self.load_episodes(self.etype)
        for episode in episodes:
            with open(episode, "r") as f:
                data = json.load(f)
                transitions = data["transitions"]
                for transition in transitions:
                    self.memory.push(transition)
                    if len(self.memory) == self.capacity:
                        return

    # Sample a random batch from the replay memory buffer
    def load_batch(self, batch_size):
        transitions = self.memory.sample(batch_size)
        batch = Transition(*zip(*transitions))
        states = transform_states(batch.state, self.device)
        next_states = transform_states(batch.next_state, self.device)
        actions = torch.cat(transform_actions(batch.action, self.device))
        rewards = torch.cat(transform_numeric(batch.reward, self.device))
        displacements = torch.cat(transform_numeric(batch.displacement, self.device))
        velocities = torch.cat(transform_numeric(batch.velocity, self.device))
        return states, actions, next_states, rewards, displacements, velocities

    # Get the appropriate cost functions
    def get_criteria(self):
        switcher = {"l1": nn.L1Loss(), "l2": nn.MSELoss(),
                    "huber": nn.HuberLoss(), "smoothl1": nn.SmoothL1Loss(),
                    "ce": nn.CrossEntropyLoss(), "nll": nn.NLLLoss(),
                    }
        criterion1 = switcher.get(self.rl_loss, nn.SmoothL1Loss())
        criterion2 = switcher.get(self.reg_loss, nn.MSELoss())
        return criterion1, criterion2

    # Get the logging information dictionary
    def get_log(self):
        log = {"etype": self.etype, "rl_loss": self.rl_loss,
               "reg_loss": self.reg_loss, "n_frames": self.n_frames,
               "n_actions": self.n_actions, "capacity": self.capacity,
               "hyperparameters": self.hyperparameters._asdict()}
        return log

    # Get the predicted values
    def get_values(self, states, actions, next_states):
        _, next_state_values1, next_state_values2, next_state_values3 = self.target_net(next_states)
        state_numeric, state_action_values1, state_action_values2, state_action_values3,  = self.policy_net(states)
        state_action_values1 = state_action_values1.gather(1, actions[:, 0].unsqueeze(1)).squeeze(1).double()
        state_action_values2 = state_action_values2.gather(1, actions[:, 1].unsqueeze(1)).squeeze(1).double()
        state_action_values3 = state_action_values3.gather(1, actions[:, 2].unsqueeze(1)).squeeze(1).double()
        next_state_values1 = next_state_values1.max(1)[0].double()
        next_state_values2 = next_state_values2.max(1)[0].double()
        next_state_values3 = next_state_values3.max(1)[0].double()
        next_state_values = (next_state_values1, next_state_values2, next_state_values3)
        state_action_values = (state_action_values1, state_action_values2, state_action_values3)
        return state_action_values, next_state_values, state_numeric

    # Perform an optimization step
    def optimize(self, criterion1, criterion2):
        gamma = self.hyperparameters.gamma
        alpha = self.hyperparameters.alpha
        batch_size = self.hyperparameters.batch_size
        states, actions, next_states, rewards, displacements, velocities = self.load_batch(batch_size)
        state_action_values, next_state_values, state_numeric = self.get_values(states, actions, next_states)
        state_action_values1, state_action_values2, state_action_values3 = state_action_values
        next_state_values1, next_state_values2, next_state_values3 = next_state_values
        expected_state_action_values1 = (next_state_values1 * gamma) + rewards
        expected_state_action_values2 = (next_state_values2 * gamma) + rewards
        expected_state_action_values3 = (next_state_values3 * gamma) + rewards
        rl_loss1 = criterion1(state_action_values1.unsqueeze(1), expected_state_action_values1.unsqueeze(1))
        rl_loss2 = criterion1(state_action_values2.unsqueeze(1), expected_state_action_values2.unsqueeze(1))
        rl_loss3 = criterion1(state_action_values3.unsqueeze(1), expected_state_action_values3.unsqueeze(1))
        rl_loss = rl_loss1 + rl_loss2 + rl_loss3
        reg_loss = criterion2(torch.cat((displacements.unsqueeze(1), velocities.unsqueeze(1)), dim=1).double(), state_numeric)
        loss = (rl_loss1 + rl_loss2 + rl_loss3) + 0 * reg_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return rl_loss, reg_loss

    # Optimize the RL agent
    def train(self):
        self.load_memory()
        log = self.get_log()
        log["rl_loss"], log["reg_loss"] = [], []
        criterion1, criterion2 = self.get_criteria()
        n_iterations = self.hyperparameters.n_iterations
        target_update = self.hyperparameters.target_update
        for i in tqdm(range(n_iterations), ascii=True, desc="Training"):
            loss1, loss2 = self.optimize(criterion1, criterion2)
            log["rl_loss"].append(loss1.item())
            log["reg_loss"].append(loss2.item())
            if (i + 1) % target_update == 0:
                self.copy_weights()
            if (i + 1) % self.checkpoint == 0:
                self.save_weights((i + 1) // self.checkpoint)
        with open(os.path.join("log", "agent", self.timestamp + ".json"), "w") as f:
            json.dump(log, f)


def main():
    parser = create_parser()
    args = parser.parse_args()
    hyperparameters = Hyperparameters(gamma=args.gamma, alpha=args.alpha, lr=args.lr, batch_size=args.batch_size,
                                      target_update=args.target_update, n_iterations=args.n_iterations)
    trainer = RLTrainer(etype=args.etype,
                        weights=args.weights,
                        n_frames=args.n_frames,
                        n_actions=args.n_actions,
                        rl_loss=args.rl_loss,
                        reg_loss=args.reg_loss,
                        capacity=args.capacity,
                        data_path=args.data_path,
                        save_path=args.save_path,
                        checkpoint=args.checkpoint,
                        hyperparameters=hyperparameters)
    trainer.train()


if __name__ == "__main__":
    # main()
    time = []
    velocity = []
    ref_velocity = []
    diff_velocity = []
    tck_v, tck_d = load_reference()
    with open("log/agent/record/testing_case_1.json", "r") as f:
        data = json.load(f)
        numeric = data["numeric"]
        for n in numeric:
            time.append(n["time"])
            velocity.append(n["velocity"])
            d, v = get_reference(tck_d, tck_v, n["time"])
            ref_velocity.append(v)
            diff_velocity.append(-(v - velocity[-1]) ** 2)
    avg_reward = np.mean(diff_velocity)

    u_json_paths = glob.glob("log/agent/testing/untrained/case2/*.json")
    r_json_paths = glob.glob("log/agent/testing/real/case2/*.json")
    g_json_paths = glob.glob("log/agent/testing/generated/case2/*.json")
    u_json_paths.sort()
    r_json_paths.sort()
    g_json_paths.sort()
    u_avg_reward = []
    r_avg_reward = []
    g_avg_reward = []

    # Untrained model log 
    for u_path in u_json_paths:
        u_time = []
        u_reward = []
        u_velocity = []
        u_displacement = []
        u_ref_velocity = []
        u_ref_displacement = []
        u_diff_velocity = []
        u_diff_displacement = []
        with open(u_path, "r") as f:
            data = json.load(f)
            numeric = data["numeric"]
            for n in numeric:
                t = n["time"]
                u_time.append(t)
                u_reward.append(n["reward"])
                u_velocity.append(n["velocity"])
                u_displacement.append(n["displacement"])
                d, v = get_reference(tck_d, tck_v, t)
                u_ref_velocity.append(v)
                u_ref_displacement.append(d)
                u_diff_velocity.append(-(v - u_velocity[-1]) ** 2)
                u_diff_displacement.append(-(d - u_displacement[-1]) ** 2)
        u_avg_reward.append(np.mean(u_diff_velocity))

    # Real pretrained model log
    for r_path in r_json_paths:
        r_time = []
        r_reward = []
        r_velocity = []
        r_displacement = []
        r_ref_velocity = []
        r_ref_displacement = []
        r_diff_velocity = []
        r_diff_displacement = []
        with open(r_path, "r") as f:
            data = json.load(f)
            numeric = data["numeric"]
            for n in numeric:
                t = n["time"]
                r_time.append(t)
                r_reward.append(n["reward"])
                r_velocity.append(n["velocity"])
                r_displacement.append(n["displacement"])
                d, v = get_reference(tck_d, tck_v, t)
                r_ref_velocity.append(v)
                r_ref_displacement.append(d)
                r_diff_velocity.append(-(v - r_velocity[-1]) ** 2)
                r_diff_displacement.append(-(d - r_displacement[-1]) ** 2)
        r_avg_reward.append(np.mean(r_diff_velocity))

    # Real + Generated pretrained model log
    for g_path in g_json_paths:
        g_time = []
        g_reward = []
        g_velocity = []
        g_displacement = []
        g_ref_velocity = []
        g_ref_displacement = []
        g_diff_velocity = []
        g_diff_displacement = []
        with open(g_path, "r") as f:
            data = json.load(f)
            numeric = data["numeric"]
            for n in numeric:
                t = n["time"]
                g_time.append(t)
                g_reward.append(n["reward"])
                g_velocity.append(n["velocity"])
                g_displacement.append(n["displacement"])
                d, v = get_reference(tck_d, tck_v, t)
                g_ref_velocity.append(v)
                g_ref_displacement.append(d)
                g_diff_velocity.append(-(v - g_velocity[-1]) ** 2)
                g_diff_displacement.append(-(d - g_displacement[-1]) ** 2)
        g_avg_reward.append(np.mean(g_diff_velocity))

    u_avg_reward.insert(0, avg_reward)
    r_avg_reward.insert(0, avg_reward)
    g_avg_reward.insert(0, avg_reward)

    # Display the reward function
    plt.plot(u_avg_reward, "-b*", linewidth=1.5, label="Untrained")
    plt.plot(r_avg_reward, "-r^", linewidth=1.5, label="Real")
    plt.plot(g_avg_reward, "-gs", linewidth=1.5, label="Generated")
    plt.legend()
    plt.show()