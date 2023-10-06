import os
import cv2
import json
import time
import random
import shutil
import argparse
import numpy as np

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy import interpolate
from collections import namedtuple
from torch.utils.data import DataLoader

from models import DeepQNet, SoftSensor
from dataset import SoftSensingDataset
from rl_control import OnsetDetection


def create_parser():
    parser = argparse.ArgumentParser(description="Soft Sensing Training")
    parser.add_argument("--type", type=str, default="soft")
    parser.add_argument("--n_frames", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--features", type=int, default=32)
    parser.add_argument("--save_path", type=str, default="./models/agents")
    parser.add_argument("--log_path", type=str, default="./log/agents")
    parser.add_argument("--json_path", type=str, default="./data/frames/training/annotations.json")
    return parser


# Load the reference curves
def load_reference():
    with open("reference/velocity.json", "r") as f1:
        tck_v_json = json.load(f1)
    with open("reference/displacement.json", "r") as f2:
        tck_d_json = json.load(f2)
    tck_v = (np.array(tck_v_json[0]), np.array(tck_v_json[1]), tck_v_json[2])
    tck_d = (np.array(tck_d_json[0]), np.array(tck_d_json[1]), tck_d_json[2])
    return tck_d, tck_v


# Get the reference curve data at given time instance
def get_reference(t, tck_d, tck_v):
    if t < 10:
        v = interpolate.splev(t, tck_v, der=0)
        d = interpolate.splev(t, tck_d, der=0)
    else:
        v = 0
        d = 2.86
    return v, d


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

# Training Hyperparameters
Hyperparameters = namedtuple("Hyperparameters", ["n_epochs", "batch_size", "lr"])


# Soft sensing training class
class SSTrainer:
    def __init__(self,
                 model,
                 dataset,
                 hyperparameters,
                 save_path,
                 log_path):
        self.model = model
        self.dataset = dataset
        self.hyperparameters = hyperparameters
        self.save_path = save_path
        self.log_path = log_path
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hyperparameters.lr)
        self.timestamp = str(millis())

    # Weight initialization
    def initialize_weights(self):
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.GroupNorm, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # Split the dataset
    def split_dataset(self, ratio=0.8):
        length = len(self.dataset)
        train_length = int(ratio * length)
        val_length = length - train_length
        train_set, val_set = torch.utils.data.random_split(self.dataset, [train_length, val_length])
        train_loader = DataLoader(train_set, batch_size=self.hyperparameters.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=self.hyperparameters.batch_size, shuffle=True)
        return train_loader, val_loader

    # Make all network parameters trainable
    def make_trainable(self):
        for p in self.model.parameters():
            p.requires_grad = True

    # Single epoch training loop
    def train_loop(self, dataloader, criterion):
        epoch_train_loss = []
        self.model.train()
        for (X, y) in tqdm(dataloader, ascii=True, desc="Training"):
            X = X.to(self.device)
            y = y.to(self.device)
            self.optimizer.zero_grad()
            pred, _, _, _ = self.model(X)
            loss = criterion(pred, y)
            loss.backward()
            self.optimizer.step()
            epoch_train_loss.append(loss.item())
        return epoch_train_loss

    # Single epoch validation loop
    def val_loop(self, dataloader, criterion):
        epoch_val_loss = []
        self.model.eval()
        with torch.no_grad():
            for (X, y) in tqdm(dataloader, ascii=True, desc="Validation"):
                X = X.to(self.device)
                y = y.to(self.device)
                pred, _, _, _ = self.model(X)
                loss = criterion(pred, y).item()
                epoch_val_loss.append(loss)
        return epoch_val_loss

    # Model training function
    def train(self):
        log = {"Hyperparameters": self.hyperparameters._asdict()}
        self.model.to(self.device)
        self.make_trainable()
        self.initialize_weights()
        train_loss = np.zeros((self.hyperparameters.n_epochs,))
        val_loss = np.zeros((self.hyperparameters.n_epochs,))
        train_loader, val_loader = self.split_dataset()
        criterion = nn.SmoothL1Loss()
        statistics = []
        for t in range(self.hyperparameters.n_epochs):
            print("--------------- Epoch {} ---------------".format(t + 1))
            train_loss_t = self.train_loop(train_loader, criterion)
            val_loss_t = self.val_loop(val_loader, criterion)
            train_loss[t] = np.mean(train_loss_t)
            val_loss[t] = np.mean(val_loss_t)
            print("Training loss: {}, Validation loss: {}".format(train_loss[t], val_loss[t]))
            model_path = os.path.join(self.save_path, self.timestamp, "soft" + "_" + str(t + 1) + ".pth")
            torch.save(self.model.state_dict(), model_path)
            info = {"epoch": t + 1, "train_loss": train_loss[t], "val_loss": val_loss[t], "model_path": model_path}
            statistics.append(info)
        log["statistics"] = statistics
        with open(os.path.join(self.log_path, self.timestamp + ".json"), "w") as f:
            json.dump(log, f)


def inference(path, sensor, n_frames, display=False):
    with open(path, "r") as f:
        data = json.load(f)
        e_velocity = [0] * n_frames
        e_displacement = [0] * n_frames
        paths = data["paths"]
        velocity = data["velocity"]
        displacement = data["displacement"]
        for i in range(len(displacement)):
            displacement[i] = displacement[i] / 100
        onset_idx = onset_detection(displacement, velocity, display=True)
        state = [torch.from_numpy(np.transpose(cv2.imread(paths[i]) / 255, (2, 0, 1))).unsqueeze(0) for i in range(0, n_frames)]
        for i in range(n_frames, len(paths)):
            print(i)
            state.pop(0)
            state.append(torch.from_numpy(np.transpose(cv2.imread(paths[i]) / 255, (2, 0, 1))).unsqueeze(0))
            state_numeric, _, _, _ = sensor(torch.cat(state, dim=0).unsqueeze(0))
            e_velocity.append(state_numeric.squeeze(0)[1].item())
            e_displacement.append(state_numeric.squeeze(0)[0].item())

        onset_idx = onset_detection(e_displacement, e_velocity, display=False)
        print(np.sum(e_velocity) / 13.5)
        v_onset = e_velocity[:onset_idx+1] + 0.03 * np.random.randn(onset_idx+1)
        plt.plot(v_onset, "-b*", linewidth=1.5)
        plt.grid(axis="y")
        plt.plot([onset_idx], [v_onset[-1]], marker="o", markerfacecolor="red", markeredgecolor="red", label="Onset")
        plt.plot([0, onset_idx], [0.08, 0.08], "--k", linewidth=1.2)
        plt.plot([0, onset_idx], [-0.08, -0.08], "--k", linewidth=1.2)
        plt.show()

        # e_displacement = e_displacement[onset_idx:]
        # e_velocity = e_velocity[onset_idx:]
        # r_displacement = []
        # r_velocity = []
        # tck_d, tck_v = load_reference()
        # fs = 13.5
        # time = []
        # for i, (d, v) in enumerate(zip(e_displacement, e_velocity)):
        #     t = i / fs
        #     time.append(t)
        #     ref_v, ref_d = get_reference(t, tck_d, tck_v)
        #     r_displacement.append(ref_d)
        #     r_velocity.append(ref_v)
        # plt.plot(time, r_velocity, "-k", label="Reference", linewidth=1.5)
        # plt.plot(time, e_velocity, "-b", label="Estimated", linewidth=1.5)
        # plt.xlabel("Time", fontsize=14)
        # plt.ylabel("Velocity", fontsize=14)
        # plt.show()

        if display:
            plt.figure()
            plt.plot(velocity, label="Actual", linewidth=1.5, c="blue")
            plt.plot(e_velocity, label="Estimated", linewidth=1.5, c="orange")
            plt.plot([onset_idx], [e_velocity[onset_idx]], marker="o", markerfacecolor="red", label="Onset")
            plt.title("(c)", weight="bold")
            plt.ylabel("Velocity")
            plt.xlabel("Time")
            plt.legend()

            plt.figure()
            plt.plot(displacement, label="Actual", linewidth=1.5, c="blue")
            plt.plot(e_displacement, label="Estimated", linewidth=1.5, c="orange")
            plt.plot([onset_idx], [e_displacement[onset_idx]], marker="o", markerfacecolor="red", label="Onset")
            plt.title("(f)", weight="bold")
            plt.ylabel("Displacement")
            plt.xlabel("Time")
            plt.legend()
            plt.show()

        return displacement, velocity


# Detect and display the detected onset of motion
def onset_detection(displacement, velocity, display=False):
    onset_idx = len(displacement) - 1
    onset_detector = OnsetDetection(threshold=5, lag=20, influence=1)
    for i, v in enumerate(velocity):
        onset = onset_detector.threshold_algo(v + 0.05 * (random.random() - 0.5))
        if onset == 1:
            onset_idx = i
            break

    if display:
        plt.figure()
        plt.plot(velocity, label="Actual", linewidth=1.5, c="blue")
        plt.plot([onset_idx], [velocity[onset_idx]], marker="o", markerfacecolor="red", label="Onset")
        plt.title("Velocity Estimation")
        plt.ylabel("Velocity")
        plt.xlabel("Time")
        plt.legend()

        plt.figure()
        plt.plot(displacement, label="Actual", linewidth=1.5, c="blue")
        plt.plot([onset_idx], [displacement[onset_idx]], marker="o", markerfacecolor="red", label="Onset")
        plt.title("Displacement Estimation")
        plt.ylabel("Displacement")
        plt.xlabel("Time")
        plt.legend()
        plt.show()

    return onset_idx


def main():
    parser = create_parser()
    args = parser.parse_args()
    sensor = DeepQNet(channels=args.channels, features=args.features).double()
    dataset = SoftSensingDataset(json_path=args.json_path)
    hyperparameters = Hyperparameters(n_epochs=args.n_epochs, batch_size=args.batch_size, lr=args.lr)
    trainer = SSTrainer(
        model=sensor,
        dataset=dataset,
        hyperparameters=hyperparameters,
        save_path=args.save_path,
        log_path=args.log_path
    )
    trainer.train()


if __name__ == "__main__":
    # main()
    sensor = DeepQNet(channels=3, features=32).double()
    sensor.load_state_dict(torch.load("/home/apostolos/PycharmProjects/Generative-RL/models/agent/r_dqn.pth", map_location="cpu"))
    sensor.eval()

    # agent = DeepQNet(channels=3, features=32).double()
    # params1 = sensor.named_parameters()
    # params2 = agent.named_parameters()
    #
    # dict_params2 = dict(params2)
    #
    # for name1, param1 in params1:
    #     if name1 in dict_params2:
    #         if "block" in name1:
    #             dict_params2[name1].data.copy_(param1.data)
    #         if "fc2.weight" in name1:
    #             dict_params2["fc_ss.weight"].data.copy_(param1.data)
    #         if "fc2.bias" in name1:
    #             dict_params2["fc_ss.bias"].data.copy_(param1.data)
    # agent.load_state_dict(dict_params2)
    # torch.save(agent.state_dict(), "sensor.pth")

    displacement, velocity = inference("/home/apostolos/PycharmProjects/Generative-RL/data/frames/testing/stream_1661426290910/data.json", sensor, 5, display=True)
    # with open("/home/apostolos/PycharmProjects/Generative-RL/data/frames/training/annotations.json", "r") as f:
    #     annotations = json.load(f)
    #     for sample in annotations["samples"]:
    #         sample["velocity"] = 100 * sample["velocity"]
    # with open("/home/apostolos/PycharmProjects/Generative-RL/data/frames/training/annotations_.json", "w") as f:
    #     json.dump(annotations, f, indent=4)