import os
import cv2
import glob
import time
import json
import random
import shutil
import argparse

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from nema17 import Nema17
from models import DeepQNet
from scipy import interpolate
from collections import deque
from collections import namedtuple


# Definition of constants and named tuples
N_FRAMES = 5
N_ACTIONS = 3
DIM = (512, 512)
TERMINAL_TIME = 11
TERMINAL_VELOCITY = 0
TERMINAL_DISPLACEMENT = 2.62
Hyperparameters = namedtuple("Hyperparameters", ["lr", "batch_size", "gamma", "target_update"])
Transition = namedtuple("Transition", ["state", "action", "next_state", "reward", "displacement", "velocity"])


# Create command line argument parser
def create_parser():
    parser = argparse.ArgumentParser(description="Online RL training")

    # Q-Learning arguments
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--target_update", type=int, default=20)
    parser.add_argument("--epsilon", type=float, default=0.02)
    parser.add_argument("--capacity", type=int, default=10000)
    parser.add_argument("--rl_loss", type=str, default="smoothl1")

    # Webcam arguments
    parser.add_argument("--fps", type=float, default=13.5)
    parser.add_argument("--cap_id", type=int, default=0)
    parser.add_argument("--cap_width", type=int, default=960)
    parser.add_argument("--cap_height", type=int, default=720)
    parser.add_argument("--lim_width", type=int, default=310)
    parser.add_argument("--lim_height", type=int, default=700)
    parser.add_argument("--step", type=int, default=3)

    # Onset detection arguments
    parser.add_argument("--lag", type=int, default=20)
    parser.add_argument("--influence", type=float, default=1)
    parser.add_argument("--threshold", type=float, default=4)

    # Miscellaneous arguments
    parser.add_argument("--etype", type=str, default="training")
    parser.add_argument("--mtype", type=str, default="untrained")
    parser.add_argument("--data_path", type=str, default="episodes")
    parser.add_argument("--save_path", type=str, default="data")
    parser.add_argument("--log_path", type=str, default="log/agent")
    parser.add_argument("--transform", type=str, default="data/calibration/warp.npy")
    return parser


# Return a timestamp
def millis():
    return round(time.time() * 1000)


# Create a directory with the given name
def create_dir(folder, remove=True):
    if remove:
        try:
            shutil.rmtree(folder)
        except FileNotFoundError:
            pass
        os.makedirs(folder, exist_ok=True)


# Interpolate the reference curve
def interpolate_and_save():
    df = pd.read_csv("reference/reference.csv")
    time = df["TIME"].to_numpy()
    velocity = df["VELOCITY"].to_numpy()
    displacement = np.zeros_like(velocity)
    for i in range(1, len(velocity)):
        displacement[i] = time[i] * np.mean(velocity[:i])
    time = time[6:] - time[6]
    velocity, displacement = 1.10 * velocity[6:], 1.10 * displacement[6:]
    plt.plot(displacement)
    plt.show()
    tck_v = interpolate.splrep(time, velocity, k=3, s=0.002)
    tck_d = interpolate.splrep(time, displacement, k=1, s=0.003)
    tck_v_json = (tck_v[0].tolist(), tck_v[1].tolist(), tck_v[2])
    tck_d_json = (tck_d[0].tolist(), tck_d[1].tolist(), tck_d[2])
    with open("reference/velocity.json", "w") as f1:
        json.dump(tck_v_json, f1)
    with open("reference/displacement.json", "w") as f2:
        json.dump(tck_d_json, f2)


# Load the reference curves
def load_reference():
    with open("reference/velocity.json", "r") as f1:
        tck_v_json = json.load(f1)
    with open("reference/displacement.json", "r") as f2:
        tck_d_json = json.load(f2)
    tck_v = (np.array(tck_v_json[0]), np.array(tck_v_json[1]), tck_v_json[2])
    tck_d = (np.array(tck_d_json[0]), np.array(tck_d_json[1]), tck_d_json[2])
    return tck_v, tck_d


# Trenary representation of a number
def ternary(n):
    if n == 0:
        return '0'.zfill(3)
    nums = []
    while n:
        n, r = divmod(n, 3)
        nums.append(str(r))
    return ''.join(reversed(nums)).zfill(3)


# Transform integers to actions
def int2action(n, step):
    mapping = {'0': 0, '1': step, '2': -step}
    action_str = ternary(n)
    action = [mapping[action_str[0]], mapping[action_str[1]], mapping[action_str[2]]]
    return action


# Transform actions to integers
def action2int(action, step):
    inverse_mapping = {0: 0, step: 1, -step: 2}
    return 3 ** 2 * inverse_mapping[action[0]] + 3 * inverse_mapping[action[1]] + inverse_mapping[action[2]]


# Load the frames of a batch
def transform_states(states, device):
    tstates = []
    for state in states:
        tstate = []
        for path in state:
            frame = cv2.imread(path) / 255
            tstate.append(np.transpose(frame, (2, 0, 1)))
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


# On-line onset detection algorithm
class OnsetDetection:
    def __init__(self, threshold, lag, influence):
        self.y = []
        self.filteredY = []
        self.onset = 0
        self.length = len(self.y)
        self.threshold = threshold
        self.influence = influence
        self.lag = lag
        self.avg = 0
        self.std = 1

    def threshold_algo(self, val):
        self.y.append(val)
        i = len(self.y) - 1
        if i < self.lag:
            return 0
        elif i == self.lag:
            self.onset = 0
            self.filteredY = list(np.array(self.y))
            self.avg = np.mean(self.y)
            self.std = np.std(self.y)

        self.filteredY += [0]
        if np.abs(self.y[i] - self.avg) > self.threshold * self.std:
            if self.y[i] > self.avg:
                self.onset = 1
            else:
                self.onset = -1
            self.filteredY[i] = self.influence * self.y[i] + (1 - self.influence) * self.filteredY[i - 1]
            self.avg = np.mean(self.filteredY)
            self.std = np.std(self.filteredY)
        else:
            self.onset = 0
            self.filteredY[i] = self.y[i]
            self.avg = np.mean(self.filteredY)
            self.std = np.std(self.filteredY)

        return self.onset


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


# Online RL agent class
class RLAgent:
    def __init__(self,
                 fps,
                 cap_id,
                 cap_width,
                 cap_height,
                 dim,
                 lim_width,
                 lim_height,
                 step,
                 lag,
                 threshold,
                 influence,
                 mtype,
                 etype,
                 rl_loss,
                 epsilon,
                 capacity,
                 n_frames,
                 n_actions,
                 log_path,
                 data_path,
                 save_path,
                 hyperparameters,
                 transform=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tck_v, self.tck_d = load_reference()

        # Setting up camera parameters
        self.cap = cv2.VideoCapture(int(cap_id))
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.set_resolution(width=cap_width, height=cap_height)
        self.fps = fps

        # Onset detection parameters
        self.lag = lag
        self.influence = influence
        self.threshold = threshold
        self.terminal_time = TERMINAL_TIME
        self.terminal_velocity = TERMINAL_VELOCITY
        self.terminal_displacement = TERMINAL_DISPLACEMENT
        self.onset_detector = OnsetDetection(lag=self.lag, influence=self.influence, threshold=self.threshold)

        # Transform parameters
        self.dim = dim
        self.lim_width = lim_width
        self.lim_height = lim_height
        self.H = np.load(transform) if transform is not None else np.identity(3)

        # Reinforcement learning parameters
        self.mtype = mtype
        self.etype = etype
        self.log_path = log_path
        self.data_path = data_path
        self.save_path = save_path
        self.n_frames = n_frames
        self.n_actions = n_actions
        self.rl_loss = rl_loss
        self.a_counter = 0
        self.previous_action = [0, 0, 0]
        self.epsilon = epsilon
        self.capacity = capacity
        self.hyperparameters = hyperparameters
        self.memory = ReplayMemory(capacity)
        self.policy_net = DeepQNet(n_frames=n_frames, n_actions=n_actions).double()
        self.target_net = DeepQNet(n_frames=n_frames, n_actions=n_actions).double()
        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=hyperparameters.lr)
        self.image_data = {}
        self.state_buffer = []
        self.load_weights()
        self.load_memory()

        # Log parameters
        self.counter = 0
        self.timestamp = str(millis())
        if self.etype == "record":
            create_dir(os.path.join(save_path, self.etype, "stream_" + self.timestamp))
        else:
            create_dir(os.path.join(save_path, self.etype, self.mtype, "stream_" + self.timestamp))

        # Hardware parameters
        self.step = step
        self.nema17 = Nema17()

    # Define the destructor to release the webcam
    def __del__(self):
        self.cap.release()

    # Get the reference curve data at given time instance
    def get_reference(self, t):
        if t < 9.5:
            v = interpolate.splev(t, self.tck_v, der=0)
            d = interpolate.splev(t, self.tck_d, der=0)
        else:
            v = self.terminal_velocity
            d = self.terminal_displacement
        return np.array([d, v], dtype=np.float64)

    # Transform the captured frame to the given dimension
    def transform_frame(self, frame):
        if self.etype == "record":
            opath = os.path.join(self.save_path, self.etype, "stream_" + self.timestamp, "oframe_" + str(self.counter) + ".bmp")
            tpath = os.path.join(self.save_path, self.etype, "stream_" + self.timestamp, "frame_" + str(self.counter) + ".bmp")
        else:
            opath = os.path.join(self.save_path, self.etype, self.mtype, "stream_" + self.timestamp, "oframe_" + str(self.counter) + ".bmp")
            tpath = os.path.join(self.save_path, self.etype, self.mtype, "stream_" + self.timestamp, "frame_" + str(self.counter) + ".bmp")
        cv2.imwrite(opath, frame)
        frame = cv2.warpPerspective(frame, self.H, (self.lim_width, self.lim_height))
        frame = cv2.resize(frame, self.dim, cv2.INTER_AREA)
        cv2.imshow("Frame", frame)
        cv2.waitKey(1)
        cv2.imwrite(tpath, frame)
        self.image_data[tpath] = frame / 255
        self.counter += 1
        return frame / 255

    # Load the frames of a batch
    def transform_states(self, states, device):
        tstates = []
        for state in states:
            tstate = []
            for path in state:
                frame = self.image_data[path]
                tstate.append(np.transpose(frame, (2, 0, 1)))
            tstate = np.array(tstate)
            tstates.append(tstate)
        return torch.from_numpy(np.array(tstates)).to(device)

    # Save the weights of the policy network
    def save_weights(self):
        model_path = os.path.join("models", "agent", self.etype, self.mtype, self.timestamp + ".pth")
        torch.save(self.policy_net.state_dict(), model_path)
        return model_path

    # Load the pre-trained network weights
    def load_weights(self):
        print("Loading the weights of the network")
        mapping = {"untrained": "models/agent/dqn.pth",
                   "generated": "models/agent/g_dqn.pth",
                   "real": "models/agent/r_dqn.pth"}
        weight_paths = glob.glob(os.path.join("models", "agent", self.etype, self.mtype, "*"))
        weight_paths.sort(key=lambda s: int(s.split("/")[-1].split(".")[0]))
        if self.etype == "training":
            weights = weight_paths[-1] if weight_paths else mapping[self.mtype]
        elif self.etype == "testing":
            if weight_paths:
                weights = weight_paths[-1]
            else:
                weights = mapping[self.mtype]
        else:
            weights = mapping["untrained"]
        self.policy_net.load_state_dict(torch.load(weights, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.policy_net.double(), self.target_net.double()
        self.target_net.eval()
        self.policy_net.to(self.device)
        self.target_net.to(self.device)
        # Freeze the weights of the regression network
        for n, p in self.policy_net.named_parameters():
            if not ("fc1" in n) and not ("fc2" in n) and not ("fc3" in n):
                p.requires_grad = False
        for n, p in self.target_net.named_parameters():
            if not ("fc1" in n) and not ("fc2" in n) and not ("fc3" in n):
                p.requires_grad = False

    # Load real or generated data
    def load_episodes(self):
        training_episodes = glob.glob(os.path.join(self.data_path, "training", self.mtype, "*.json"))
        testing_episodes = glob.glob(os.path.join(self.data_path, "testing", self.mtype, "*.json"))
        return testing_episodes

    # Load all of the images in memory
    def load_images(self, episodes):
        training_dirs = glob.glob(os.path.join(self.save_path, "training", self.mtype, "*"))
        testing_dirs = glob.glob(os.path.join(self.save_path, "testing", self.mtype, "stream*"))
        for dir in testing_dirs:
           paths = glob.glob(os.path.join(dir, "*.bmp"))
           for path in paths:
               if "o" not in path:
                   self.image_data[path] = cv2.imread(path) / 255
        # if len(episodes) > 5:
        #     episodes = episodes[:5]
        # elif len(episodes) == 0:
        #     return
        # for episode in episodes:
        #     with open(episode, "r") as f:
        #         data = json.load(f)
        #         transitions = data["transitions"]
        #         for path in transitions[0][0]:
        #             self.image_data[path] = cv2.imread(path) / 255
        #         for transition in transitions[1:]:
        #             self.image_data[transition[0][-1]] = cv2.imread(transition[0][-1]) / 255
        #         self.image_data[transitions[-1][0][-1]] = cv2.imread(transitions[-1][0][-1]) / 255


    # Fill the memory buffer with the gathered experience
    def load_memory(self):
        episodes = self.load_episodes()
        self.load_images(episodes)
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
        states = self.transform_states(batch.state, self.device)
        next_states = self.transform_states(batch.next_state, self.device)
        actions = torch.cat(transform_actions(batch.action, self.device))
        rewards = torch.cat(transform_numeric(batch.reward, self.device))
        displacements = torch.cat(transform_numeric(batch.displacement, self.device))
        velocities = torch.cat(transform_numeric(batch.velocity, self.device))
        return states, actions, next_states, rewards, displacements, velocities

    # Get the appropriate cost functions
    def get_criterion(self):
        switcher = {"l1": nn.L1Loss(), "l2": nn.MSELoss(),
                    "huber": nn.HuberLoss(), "smoothl1": nn.SmoothL1Loss(),
                    "ce": nn.CrossEntropyLoss(), "nll": nn.NLLLoss(),
                    }
        criterion = switcher.get(self.rl_loss, nn.SmoothL1Loss())
        return criterion

    # Get the logging information dictionary
    def get_log(self):
        log = {"etype": self.mtype, "rl_loss": self.rl_loss,
               "n_frames": self.n_frames, "n_actions": self.n_actions,
               "capacity": self.capacity, "hyperparameters": self.hyperparameters._asdict()}
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
    def optimize(self, criterion):
        gamma = self.hyperparameters.gamma
        batch_size = self.hyperparameters.batch_size
        states, actions, next_states, rewards, displacements, velocities = self.load_batch(batch_size)
        state_action_values, next_state_values, state_numeric = self.get_values(states, actions, next_states)
        state_action_values1, state_action_values2, state_action_values3 = state_action_values
        next_state_values1, next_state_values2, next_state_values3 = next_state_values
        expected_state_action_values1 = (next_state_values1 * gamma) + rewards
        expected_state_action_values2 = (next_state_values2 * gamma) + rewards
        expected_state_action_values3 = (next_state_values3 * gamma) + rewards
        rl_loss1 = criterion(state_action_values1.unsqueeze(1), expected_state_action_values1.unsqueeze(1))
        rl_loss2 = criterion(state_action_values2.unsqueeze(1), expected_state_action_values2.unsqueeze(1))
        rl_loss3 = criterion(state_action_values3.unsqueeze(1), expected_state_action_values3.unsqueeze(1))
        rl_loss = rl_loss1 + rl_loss2 + rl_loss3
        self.optimizer.zero_grad()
        rl_loss.backward()
        self.optimizer.step()
        return rl_loss

    # Warm up images and load the frame buffer
    def warm_up(self):
        for _ in range(self.n_frames):
            ret, frame = self.cap.read()
            frame = self.transform_frame(frame)
            self.state_buffer.append(frame)

    # Set the resolution of the webcam
    def set_resolution(self, height=720, width=960):
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)

    def get_paths(self):
        return [os.path.join(self.save_path, self.etype, self.mtype,
                "stream_" + self.timestamp, "frame_" + str(i) + ".bmp")
                for i in range(self.counter - self.n_frames, self.counter)]

    # Construct the state tensor
    def get_state(self):
        state = np.asarray(self.state_buffer)
        state = np.transpose(state, (0, 3, 1, 2))
        state = torch.from_numpy(state).unsqueeze(0).contiguous().to(self.device)
        return state, self.get_paths()

    # Get the optimal action based on the current policy
    def get_action(self, state):
        mapping = {0:0, 1: self.step, 2: -self.step}
        sample = random.random()
        if sample > self.epsilon and self.a_counter == 0:
            with torch.no_grad():
                _, state_action_value1, state_action_value2, state_action_value3 = self.policy_net(state)
                action1 = mapping[state_action_value1.max(1)[1].view(1).item()]
                action2 = mapping[state_action_value2.max(1)[1].view(1).item()]
                action3 = mapping[state_action_value3.max(1)[1].view(1).item()]
                action = [action1, action2, action3]
        else:
            if self.a_counter == 0:
                action1 = mapping[random.randint(0, 2)]
                action2 = mapping[random.randint(0, 2)]
                action3 = mapping[random.randint(0, 2)]
                action = [action1, action2, action3]
                self.previous_action = action
                self.a_counter += 1
            elif self.a_counter < 4:
                action = self.previous_action
                self.a_counter += 1
            else:
                action = self.previous_action
                self.a_counter = 0
        return action

    # Calculate the reward
    def get_reward(self, t, state_numeric):
        reference = self.get_reference(t)
        state_numeric = state_numeric.squeeze(0).detach().cpu().numpy()
        state_numeric[1] = 1.20 * state_numeric[1]
        state_numeric[0] = 0.2 * state_numeric[0]
        reference[0] = 0.2 * reference[0]
        return - np.linalg.norm(reference - state_numeric) ** 2

    # Update the state buffer
    def update_buffer(self, frame):
        frame = self.transform_frame(frame)
        self.state_buffer.append(frame)
        self.state_buffer.pop(0)

    # Onset detection
    def detect_onset(self):
        ctr = 0
        while True:
            ctr += 1
            onset = 0
            ret, frame = self.cap.read()
            self.update_buffer(frame)
            state, _ = self.get_state()
            state_numeric, _, _, _ = self.target_net(state)
            # print(state_numeric.squeeze(0)[1].item())
            onset = self.onset_detector.threshold_algo(state_numeric.squeeze(0)[1].item() + 0.01 * random.random() - 0.015)
            if state_numeric.squeeze(0)[1].item() > 0.07 and ctr > 20:
                print("Downward movement has started...")
                break

    # Termination criteria
    def terminate(self, t, displacement, velocity):
        displacement_condition = displacement > self.terminal_displacement - 1E-2
        velocity_condition = (velocity > self.terminal_velocity - 1E-2) and (t > 10)
        time_condition = t > self.terminal_time
        termination = (displacement_condition and velocity_condition) or time_condition
        if termination:
            print("Movement has terminated...")
        return termination

    # Display the numerical data
    def display_numeric(self, log):
        timer = []
        reward = []
        velocity = []
        displacement = []
        ref_velocity = []
        ref_displacement = []
        numeric = log["numeric"]
        for data in numeric:
            reference = self.get_reference(data["time"])
            timer.append(data["time"])
            reward.append(data["reward"])
            velocity.append(data["velocity"])
            displacement.append(data["displacement"])
            ref_velocity.append(reference[1])
            ref_displacement.append(reference[0])

        # Display the estimated displacement
        plt.figure()
        plt.plot(timer, displacement, "-b*", linewidth=1.5, label="Displacement")
        plt.plot(timer, ref_displacement, "-k^", linewidth=1.8, label="Reference")
        plt.title("Estimated Displacement")
        plt.xlabel("t (sec)")
        plt.ylabel("z (m)")
        plt.legend()

        # Display the estimated velocity
        plt.figure()
        plt.plot(timer, velocity, "-r*", linewidth=1.5, label="Velocity")
        plt.plot(timer, ref_velocity, "-k^", linewidth=1.8, label="Reference")
        plt.title("Estimated Velocity")
        plt.xlabel("t (sec)")
        plt.ylabel("v (m/s)")
        plt.legend()

        # Display the reward function over time
        plt.figure()
        plt.stem(timer, reward, linefmt="gray", use_line_collection=True)
        plt.title("Estimated Reward")
        plt.xlabel("t (sec)")
        plt.ylabel("r")
        plt.show()

    # Record an episode without taking any actions
    def record(self):
        timer = []
        log = {"numeric": []}

        # Warm up camera and detect the onset of movement
        cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
        self.warm_up()
        self.detect_onset()
        ret, frame = self.cap.read()
        begin = time.time()

        # Get the initial state
        self.update_buffer(frame)

        # While the episode has not ended
        while self.cap.isOpened():
            # Capture a new frame and update the buffer
            ret, frame = self.cap.read()
            t = time.time() - begin
            self.update_buffer(frame)

            # Get the next state and the numerical information
            next_state, next_state_paths = self.get_state()
            next_state_numeric, _, _, _ = self.target_net(next_state)

            # Estimate the reward, velocity, and displacement of the movement
            reward = self.get_reward(t, next_state_numeric)
            displacement = next_state_numeric.squeeze(0)[0].item()
            velocity = 1.20 * (next_state_numeric.squeeze(0)[1].item() - 0.022)

            # Update the logger and check if the episode has terminated
            timer.append(t)
            numeric = {"time": t, "displacement": displacement,
                       "velocity": velocity, "reward": reward,
                       "path": os.path.join(self.save_path, self.etype, "stream_" + self.timestamp, "frame_" + str(self.counter) + ".bmp"),
                       "opath": os.path.join(self.save_path, self.etype, "stream_" + self.timestamp, "oframe_" + str(self.counter) + ".bmp")}
            log["numeric"].append(numeric)
            if self.terminate(t, displacement, velocity):
                with open(os.path.join(self.log_path, self.etype, self.timestamp + ".json"), "w") as f:
                    json.dump(log, f, indent=4)
                cv2.destroyAllWindows()
                self.display_numeric(log)
                break

    # Online Q-Learning algorithm
    def control(self):
        # Initialize learning parameters
        timer = [0]
        log = self.get_log()
        log["numeric"] = []
        episode = {"transitions": []}
        criterion = self.get_criterion()
        target_update = self.hyperparameters.target_update

        # Warm up camera and detect the onset of movement
        cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
        self.warm_up()
        self.detect_onset()
        ret, frame = self.cap.read()
        begin = time.time()

        # Get the initial state and action
        self.update_buffer(frame)
        state, state_paths = self.get_state()
        action = self.get_action(state)

        # Move the motors
        self.nema17.move(action[0], action[1], action[2])

        while self.cap.isOpened():
            # Capture a new frame and update the buffer
            ret, frame = self.cap.read()
            t = time.time() - begin
            self.update_buffer(frame)

            # Get the next state and the numerical information
            next_state, next_state_paths = self.get_state()
            next_state_numeric, _, _, _ = self.target_net(next_state)

            # Estimate the reward, velocity, and displacement of the movenet
            reward = self.get_reward(t, next_state_numeric)
            displacement = next_state_numeric.squeeze(0)[0].item()
            velocity = 1.10 * (next_state_numeric.squeeze(0)[1].item() - 0.022)

            # Append the memory buffer with a new transition
            transition = Transition(state_paths, action, next_state_paths, reward, displacement, velocity)
            episode["transitions"].append([state_paths, action, next_state_paths, reward, displacement, velocity])
            self.memory.push(transition)

            # Optimize the policy matrix
            self.optimize(criterion)
            if (self.counter % target_update) == 0:
                self.target_net.state_dict(self.policy_net.state_dict())

            # Update the state and select an action
            state = next_state
            state_paths = next_state_paths.copy()
            action = self.get_action(state)
            self.nema17.move(action[0], action[1], action[2])

            # Update the logger and check if the episode has terminated
            timer.append(t)
            numeric = {"time": t, "displacement": displacement,
                       "velocity": velocity, "action": action, "reward": reward,
                       "path": os.path.join(self.save_path, self.etype, self.mtype, "stream_" + self.timestamp, "frame_" + str(self.counter) + ".bmp"),
                       "opath": os.path.join(self.save_path, self.etype, self.mtype, "stream_" + self.timestamp, "frame_" + str(self.counter) + ".bmp")}
            log["numeric"].append(numeric)
            if self.terminate(t, displacement, velocity):
                log["model"] = self.save_weights()
                with open(os.path.join(self.data_path, self.etype, self.mtype, self.timestamp + ".json"), "w") as f:
                    json.dump(episode, f, indent=4)
                with open(os.path.join(self.log_path, self.etype, self.mtype, self.timestamp + ".json"), "w") as f:
                    json.dump(log, f, indent=4)
                cv2.destroyAllWindows()
                self.display_numeric(log)
                break


# Elevator weight control
def main():
    parser = create_parser()
    args = parser.parse_args()
    hyperparameters = Hyperparameters(lr=args.lr, batch_size=args.batch_size, gamma=args.gamma, target_update=args.target_update)
    agent = RLAgent(fps=args.fps,
                    cap_id=args.cap_id,
                    cap_width=args.cap_width,
                    cap_height=args.cap_height,
                    dim=DIM,
                    lim_width=args.lim_width,
                    lim_height=args.lim_height,
                    step=args.step,
                    lag=args.lag,
                    influence=args.influence,
                    threshold=args.threshold,
                    etype=args.etype,
                    mtype=args.mtype,
                    rl_loss=args.rl_loss,
                    epsilon=args.epsilon,
                    capacity=args.capacity,
                    n_frames=N_FRAMES,
                    n_actions=N_ACTIONS,
                    log_path=args.log_path,
                    data_path=args.data_path,
                    save_path=args.save_path,
                    transform=args.transform,
                    hyperparameters=hyperparameters
                    )
    begin = time.time()
    if args.etype == "record":
        agent.record()
    else:
        agent.control()
    end = time.time()
    print("Elapsed time: {}".format(end - begin))


if __name__ == "__main__":
    main()