import os
import json
import time
import shutil
import argparse
import numpy as np

import torch
import torch.nn as nn

from tqdm import tqdm
from collections import namedtuple
from torch.utils.data import DataLoader

from models import DeepQNet
from dataset import SoftSensingDataset


def create_parser():
    parser = argparse.ArgumentParser(description="Soft Sensing Training")
    parser.add_argument("--weights", type=str, default="")
    parser.add_argument("--type", type=str, default="soft")
    parser.add_argument("--n_frames", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--features", type=int, default=32)
    parser.add_argument("--save_path", type=str, default="./models/agents")
    parser.add_argument("--log_path", type=str, default="./log/agents")
    parser.add_argument("--json_path", type=str, default="./data/frames/training/annotations.json")
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
        create_dir(os.path.join(self.save_path, self.timestamp))
        create_dir(os.path.join(self.log_path, self.timestamp))

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
            _, pred = self.model(X)
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
                _, pred = self.model(X)
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
        criterion = nn.MSELoss()
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


def main():
    parser = create_parser()
    args = parser.parse_args()
    sensor = DeepQNet(channels=args.channels, features=args.features).double()
    if args.weights: sensor.load_state_dict(torch.load(args.weights))
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
    main()
