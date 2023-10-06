import os
import cv2
import json
import time
import shutil
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm

from torchmetrics import Accuracy
from collections import namedtuple
from abc import ABC, abstractmethod
from models import ResidualEncoder, ResidualDecoder
from dataset import SingleImageActionMaskDataset, DecoderDataset, EncoderDataset
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader


# Create command line argument parser
def create_parser():
    parser = argparse.ArgumentParser(description="Generative Video Training")
    parser.add_argument("--type", type=str, default="encoder")
    parser.add_argument("--n_past_frames", type=int, default=5)
    parser.add_argument("--n_future_frames", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--loss", type=str, default="ce")
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--gamma", type=float, default=0.2)
    parser.add_argument("--decay_every", type=int, default=20)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--features", type=int, default=32)
    parser.add_argument("--n_actions", type=int, default=3)
    parser.add_argument("--save_path", type=str, default="./models")
    parser.add_argument("--log_path", type=str, default="./log")
    parser.add_argument("--json_path", type=str, default="./dataset/frames512/dataset.json")
    parser.add_argument("--high_val", type=float, default=15.0)
    parser.add_argument("--low_val", type=float, default=1.0)
    return parser


Hyperparameters = namedtuple("Hyperparameters", ["n_epochs", "batch_size", "loss", "lr", "gamma", "decay_every"])


def bregman_loss(input, target, epsilon=0.1):
    probabilities = F.softmax(input, dim=1)
    target_probabilities = torch.zeros_like(input)
    for batch in range(target.shape[0]):
        target_probabilities[batch, target[batch]] = 1
    loss = - torch.sum((target_probabilities + epsilon) * torch.log(probabilities + epsilon), dim=1)
    return torch.mean(loss)


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


# Abstract training class
class Trainer(ABC):
    def __init__(self,
                 type,
                 model,
                 dataset,
                 hyperparameters,
                 save_path,
                 log_path):
        self.type = type
        self.model = model
        self.dataset = dataset
        self.hyperparameters = hyperparameters
        self.save_path = save_path
        self.log_path = log_path
        self.timestamp = str(millis())
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.hyperparameters.lr, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.hyperparameters.gamma)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        create_dir(os.path.join(self.save_path, self.timestamp))

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

    # Loss function definition
    def get_criterion(self):
        switcher = {"l1": nn.L1Loss(), "l2": nn.MSELoss(),
                    "huber": nn.HuberLoss(), "smoothl1": nn.SmoothL1Loss(),
                    "ce": nn.CrossEntropyLoss(), "nll": nn.NLLLoss(),
                    }
        return switcher.get(self.hyperparameters.loss, nn.MSELoss())

    # Make all network parameters trainable
    def make_trainable(self):
        for p in self.model.parameters():
            p.requires_grad = True

    @abstractmethod
    def train_loop(self, criterion, dataloader):
        pass

    @abstractmethod
    def val_loop(self, criterion, dataloader):
        pass

    # Model training function
    def train(self):
        log = {"Type": self.type, "Hyperparameters": self.hyperparameters._asdict()}
        self.model.to(self.device)
        self.make_trainable()
        self.initialize_weights()
        train_loss = np.zeros((self.hyperparameters.n_epochs,))
        val_loss = np.zeros((self.hyperparameters.n_epochs,))
        train_loader, val_loader = self.split_dataset()
        criterion = bregman_loss
        statistics = []
        for t in range(self.hyperparameters.n_epochs):
            print("--------------- Epoch {} ---------------".format(t + 1))
            train_loss_t = self.train_loop(train_loader, criterion)
            val_loss_t = self.val_loop(val_loader, criterion)
            train_loss[t] = np.mean(train_loss_t)
            val_loss[t] = np.mean(val_loss_t)
            print("Training loss: {}, Validation loss: {}".format(train_loss[t], val_loss[t]))
            model_path = os.path.join(self.save_path, self.timestamp, self.type + "_" + str(t + 1) + ".pth")
            torch.save(self.model.state_dict(), model_path)
            if (t + 1) % self.hyperparameters.decay_every == 0:
                self.scheduler.step()
            info = {"epoch": t + 1, "train_loss": train_loss[t], "val_loss": val_loss[t], "model_path": model_path}
            statistics.append(info)
        log["statistics"] = statistics
        with open(os.path.join(self.log_path, self.timestamp + ".json"), "w") as f:
            json.dump(log, f)


class DecoderTrainer(Trainer):
    def __init__(self,
                 model,
                 dataset,
                 hyperparameters,
                 save_path,
                 log_path,
                 high_val,
                 low_val):
        super(DecoderTrainer, self).__init__("decoder", model, dataset, hyperparameters, save_path, log_path)
        # Training parameters
        self.low_val = low_val
        self.high_val = high_val

    # Single epoch training loop
    def train_loop(self, dataloader, criterion):
        epoch_train_loss = []
        self.model.train()
        for (X, y, w) in tqdm(dataloader, ascii=True, desc="Training"):
            X = X.to(self.device)
            y = y.to(self.device)
            w = w.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(X)
            loss = criterion(torch.mul(pred, w), torch.mul(y, w))
            loss.backward()
            self.optimizer.step()
            epoch_train_loss.append(loss.item())
        return epoch_train_loss

    # Single epoch validation loop
    def val_loop(self, dataloader, criterion):
        epoch_val_loss = []
        self.model.eval()
        with torch.no_grad():
            for (X, y, w) in tqdm(dataloader, ascii=True, desc="Validation"):
                X = X.to(self.device)
                y = y.to(self.device)
                w = w.to(self.device)
                pred = self.model(X)
                loss = criterion(torch.mul(pred, w), torch.mul(y, w)).item()
                epoch_val_loss.append(loss)
        return epoch_val_loss


class EncoderTrainer(Trainer):
    def __init__(self,
                 model,
                 dataset,
                 hyperparameters,
                 save_path,
                 log_path):
        super(EncoderTrainer, self).__init__("encoder", model, dataset, hyperparameters, save_path, log_path)
        self.accuracy = Accuracy(task="multiclass", num_classes=3)

    # Single epoch training loop
    def train_loop(self, dataloader, criterion):
        epoch_train_loss = []
        epoch_train_accuracy = []
        self.model.train()
        for (X, a1, a2, a3) in tqdm(dataloader, ascii=True, desc="Training"):
            X = X.to(self.device)
            a1, a2, a3 = a1.to(self.device), a2.to(self.device), a3.to(self.device)
            self.optimizer.zero_grad()
            pred1, pred2, pred3 = self.model(X)
            loss1, loss2, loss3 = criterion(pred1, a1), criterion(pred2, a2), criterion(pred3, a3)
            accu1 = self.accuracy(torch.max(pred1, 1)[1], a1)
            accu2 = self.accuracy(torch.max(pred2, 1)[1], a2)
            accu3 = self.accuracy(torch.max(pred3, 1)[1], a3)
            accu = (accu1 + accu2 + accu3) / 3
            loss = loss1 + loss2 + loss3
            loss.backward()
            self.optimizer.step()
            epoch_train_loss.append(loss.item())
            epoch_train_accuracy.append(accu.item())
        return epoch_train_loss

    # Single epoch validation loop
    def val_loop(self, dataloader, criterion):
        epoch_val_loss = []
        epoch_val_accuracy = []
        self.model.eval()
        with torch.no_grad():
            for (X, a1, a2, a3) in tqdm(dataloader, ascii=True, desc="Validation"):
                X = X.to(self.device)
                a1, a2, a3 = a1.to(self.device), a2.to(self.device), a3.to(self.device)
                pred1, pred2, pred3 = self.model(X)
                loss1, loss2, loss3 = criterion(pred1, a1), criterion(pred2, a2), criterion(pred3, a3)
                accu1 = self.accuracy(torch.max(pred1, 1)[1], a1)
                accu2 = self.accuracy(torch.max(pred2, 1)[1], a2)
                accu3 = self.accuracy(torch.max(pred3, 1)[1], a3)
                accu = (accu1 + accu2 + accu3) / 3
                loss = loss1 + loss2 + loss3
                epoch_val_loss.append(loss.item())
                epoch_val_accuracy.append(accu.item())
        return epoch_val_loss


# Wrapper class for model training
class VideoGenerationTrainer:
    def __init__(self, model,
                 json_path,
                 n_frames,
                 batch_size,
                 n_epochs,
                 loss,
                 lr,
                 gamma,
                 lr_decay,
                 save_path,
                 log_path,
                 high_val,
                 low_val):
        self.model = model
        self.n_frames = n_frames
        self.train_loss = []
        self.val_loss = []
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.loss = loss
        self.lr = lr
        self.gamma = gamma
        self.lr_decay = lr_decay
        self.high_val = high_val
        self.low_val = low_val
        self.json_path = json_path
        self.timestamp = str(millis())
        self.dataset = SingleImageActionMaskDataset(json_path=self.json_path, high=self.high_val, low=self.low_val)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.gamma)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.save_path = os.path.join(save_path, self.timestamp)
        self.log_path = log_path
        create_dir(self.save_path)

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
        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=True)
        return train_loader, val_loader

    # Loss function definition
    def get_criterion(self):
        if self.loss == "l1":
            criterion = nn.L1Loss()
        elif self.loss == "l2":
            criterion = nn.MSELoss()
        elif self.loss == "huber":
            criterion = nn.HuberLoss()
        elif self.loss == "smoothl1":
            criterion = nn.SmoothL1Loss()
        else:
            criterion = nn.MSELoss()
        return criterion

    # Single epoch training loop
    def train_loop(self, dataloader, criterion):
        epoch_train_loss = []
        self.model.train()
        for (X, y, w, a) in tqdm(dataloader, ascii=True, desc="Training"):
            X = X.to(self.device)
            y = y.to(self.device)
            w = w.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(X)
            loss = criterion(torch.mul(pred, w), torch.mul(y, w))
            loss.backward()
            self.optimizer.step()
            epoch_train_loss.append(loss.item())
        return epoch_train_loss

    # Single epoch validation loop
    def val_loop(self, dataloader, criterion):
        epoch_val_loss = []
        self.model.eval()
        with torch.no_grad():
            for (X, y, w, a) in tqdm(dataloader, ascii=True, desc="Validation"):
                X = X.to(self.device)
                y = y.to(self.device)
                w = w.to(self.device)
                pred = self.model(X)
                loss = criterion(torch.mul(pred, w), torch.mul(y, w)).item()
                epoch_val_loss.append(loss.item())
        return epoch_val_loss

    # Model training function
    def train(self):
        log = {"batch_size": self.batch_size, "epochs": self.n_epochs,
               "loss": self.loss, "lr": self.lr,
               "gamma": self.gamma, "lr_decay": self.lr_decay,
               "high_val": self.high_val, "low_val": self.low_val}
        self.model.to(self.device)
        self.initialize_weights()
        for p in self.model.parameters():
            p.requires_grad = True
        train_loss = np.zeros((self.n_epochs,))
        val_loss = np.zeros((self.n_epochs,))
        train_loader, val_loader = self.split_dataset()
        criterion = self.get_criterion()
        statistics = []
        for t in range(self.n_epochs):
            print("--------------- Epoch {} ---------------".format(t + 1))
            train_loss_t = self.train_loop(train_loader, criterion)
            val_loss_t = self.val_loop(val_loader, criterion)
            train_loss[t] = np.mean(train_loss_t)
            val_loss[t] = np.mean(val_loss_t)
            print("Training loss: {}, Validation loss: {}".format(train_loss[t], val_loss[t]))
            model_path = os.path.join(self.save_path, "decoder_" + str(t + 1) + ".pth")
            torch.save(self.model.state_dict(), model_path)
            if (t + 1) % self.lr_decay == 0:
                self.scheduler.step()
            info = {"epoch": t + 1, "train_loss": train_loss[t], "val_loss": val_loss[t], "model_path": model_path}
            statistics.append(info)
        log["statistics"] = statistics
        with open(os.path.join(self.log_path, self.timestamp + ".json"), "w") as f:
            json.dump(log, f)


def predict(model, sample, display=False):
    model.eval()
    sample = sample.unsqueeze(0)
    prediction = model(sample)
    prediction = prediction.squeeze().detach().numpy()
    prediction = np.transpose(prediction, (0, 2, 3, 1))
    video = cv2.VideoWriter("prediction.avi", 0, 1, (256, 256))
    if display:
        cv2.namedWindow("Prediction", cv2.WINDOW_NORMAL)
        for i in range(prediction.shape[0]):
            pred = 255 * prediction[i, :, :, :]
            video.write(pred.astype("uint8"))
            cv2.imwrite("prediction.jpg", pred.astype("uint8"))
            cv2.imshow("Prediction", prediction[i, :, :, :])
            cv2.waitKey(0)
    cv2.destroyAllWindows()
    video.release()
    return prediction


def decoder_training(args):
    hyperparameters = Hyperparameters(args.n_epochs, args.batch_size, args.loss, args.lr, args.gamma, args.decay_every)
    decoder = ResidualDecoder(args.channels, args.features).double()
    dataset = DecoderDataset(args.json_path, high=args.high_val, low=args.low_val)
    trainer = DecoderTrainer(
        model=decoder,
        dataset=dataset,
        hyperparameters=hyperparameters,
        high_val=args.high_val,
        low_val=args.low_val,
        save_path=args.save_path,
        log_path=args.log_path,
    )
    trainer.train()


def encoder_training(args):
    hyperparameters = Hyperparameters(args.n_epochs, args.batch_size, args.loss, args.lr, args.gamma, args.decay_every)
    encoder = ResidualEncoder(args.channels, args.features, args.n_past_frames + args.n_future_frames, args.n_actions).double()
    dataset = EncoderDataset(args.json_path, args.n_past_frames, args.n_future_frames)
    trainer = EncoderTrainer(
        model=encoder,
        dataset=dataset,
        hyperparameters=hyperparameters,
        save_path=args.save_path,
        log_path=args.log_path
    )
    trainer.train()


def main():
    parser = create_parser()
    args = parser.parse_args()
    if args.type == "encoder":
        encoder_training(args)
    elif args.type == "decoder":
        decoder_training(args)
    else:
        print("Invalid training type.")


if __name__ == "__main__":
    main()
    # with open("dataset/frames512/dataset.json", "r") as f:
    #     actions = []
    #     data = json.load(f)
    #     for sample in data["samples"]:
    #         actions.append(sample[3])
    # actions = list(filter(lambda action: action == 26, actions))
