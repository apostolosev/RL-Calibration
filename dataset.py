import os
from abc import ABC

import cv2
import copy
import time
import glob
import json
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt


from encoding import spectrogram
from processing import estimate_velocity
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


# Convert an integer to trenary
def ternary(n):
    if n == 0:
        return '0'.zfill(3)
    nums = []
    while n:
        n, r = divmod(n, 3)
        nums.append(str(r))
    return ''.join(reversed(nums)).zfill(3)


def load_episode(episode):
    time = []
    displacement = []
    transitions = episode["transitions"]
    for transition in transitions:
        state = torch.tensor(transition["transition"][0], dtype=torch.float)
        displacement.append(state[0][3].item())
        time.append(state[0][0].item())
    return time, displacement


def create_dataset(root_dir="./episodes", save_dir="dataset"):
    paths = glob.glob(os.path.join(root_dir, "*.json"))
    paths.sort()
    M = np.load("./data/calibration/scale.npy")
    for path in paths:
        with open(path, "r") as f:
            episode = json.load(f)
            time, displacement = load_episode(episode)
            S = spectrogram(time, displacement)
            height, width = episode["height"], episode["width"]
            H = np.array(episode["transform"])
            transitions = episode["transitions"]
            for transition in transitions:
                bbox = transition["bbox"]
                frame = cv2.imread(transition["path"])
                start = (int(bbox[0][0][0]), int(bbox[0][0][1]))
                end = (int(bbox[0][3][0]), int(bbox[0][3][1]))
                cv2.rectangle(frame, start, end, (255, 0, 0), 2)
                cv2.imshow("Frame", frame)
                cv2.waitKey(0)


def track_and_annotate(root_dir="./episodes", dim=(256, 256), n_frames=10):
    paths = glob.glob(os.path.join(root_dir, "*.json"))
    paths.sort()
    tracking_params = cv2.TrackerCSRT_Params()
    setattr(tracking_params, "admm_iterations", 600)
    setattr(tracking_params, "template_size", 16)
    tracker = cv2.TrackerCSRT_create()
    j = 0
    k = 0
    samples = []
    for path in paths:
        with open(path, "r") as f:
            episode = json.load(f)
        H = np.array(episode["transform"])
        width = episode["width"]
        height = episode["height"]
        frame = cv2.imread(episode["transitions"][0]["path"])
        frame = cv2.warpPerspective(frame, H, (width, height))
        frame = cv2.resize(frame, dim, cv2.INTER_AREA)
        bbox = cv2.selectROI(frame, False)
        print(bbox[0] + bbox[2] / 2)
        print(bbox[1] + bbox[3] / 2)
        ok = tracker.init(frame, bbox)
        # cv2.imwrite("./dataset/frames180/frame_" + str(j) + ".jpg", frame)
        # np.save("./dataset/frames180/frame_" + str(j) + ".npy", bbox)
        action = episode["transitions"][0]["transition"][1][0]
        displacement = max(episode["transitions"][0]["transition"][0][0][3], 0)
        samples.append(
            ["./dataset/frames180/frame_" + str(j) + ".jpg",
             "./dataset/frames180/frame_" + str(j) + "_mask.jpg",
             displacement, action])
        j += 1
        time, displacement = load_episode(episode)
        S = spectrogram(time, displacement)
        for i, transition in enumerate(episode["transitions"][1:-n_frames]):
            action = transition["transition"][1][0]
            print(transition)
            displacement = max(transition["transition"][0][0][3], 0)
            # Spectrogram
            print(i + 1)
            start, end = time[i] * S.shape[0] / time[-1], time[i + n_frames - 1] * S.shape[0] / time[-1]
            Si = cv2.resize(S[:, int(start):int(end), :], dim, cv2.INTER_LINEAR)
            Si = 255 * np.dstack((Si, np.zeros(Si.shape[:2])))
            Si = Si.astype("uint8")
            # cv2.imwrite("./dataset/spectrograms180/spectrogram_" + str(k) + ".jpg", Si)

            # Frame
            frame = cv2.imread(transition["path"])
            frame = cv2.warpPerspective(frame, H, (width, height))
            frame = cv2.resize(frame, dim, cv2.INTER_AREA)
            # cv2.imwrite("./dataset/frames180/frame_" + str(j) + ".jpg", frame)
            ok, bbox = tracker.update(frame)
            # np.save("./dataset/frames180/frame_" + str(j) + ".npy", bbox)
            samples.append(
                ["./dataset/frames180/frame_" + str(j) + ".jpg",
                 "./dataset/frames180/frame_" + str(j) + "_mask.jpg",
                 displacement,
                 action])
            if ok:
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
            cv2.imshow("Tracking", frame)
            cv2.waitKey(1)
            j += 1
            k += 1
        for i, transition in enumerate(episode["transitions"][-n_frames:]):
            # Frame
            action = transition["transition"][1][0]
            print(transition)
            displacement = max(transition["transition"][0][0][3], 0)
            print(i + 1)
            frame = cv2.imread(transition["path"])
            frame = cv2.warpPerspective(frame, H, (width, height))
            frame = cv2.resize(frame, dim, cv2.INTER_AREA)
            # cv2.imwrite("./dataset/frames180/frame_" + str(j) + ".jpg", frame)
            ok, bbox = tracker.update(frame)
            # np.save("./dataset/frames180/frame_" + str(j) + ".npy", bbox)
            # samples.append(
            #     ["./dataset/frames180/frame_" + str(j) + ".jpg",
            #      "./dataset/frames180/frame_" + str(j) + ".npy",
            #      displacement,
            #      action])
            if ok:
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
            cv2.imshow("Tracking", frame)
            cv2.waitKey(1)
            j += 1
        cv2.destroyAllWindows()
        with open("dataset.json", "w") as f:
            dataset = {"samples": samples}
            json.dump(dataset, f, indent=4)


def create_json_annotations():
    n_samples = len(glob.glob("./dataset/plots180/*"))
    dataset = {"samples": []}
    k = 0
    for i in range(n_samples):
        if not i % 181 and i != 0:
            k += 9
        frame_paths = [("./dataset/frames180/frame_" + str(j) + ".jpg", "./dataset/frames180/frame_" + str(j) + ".npy")
                       for j in range(k, k + 10)]
        spectrogram_path = "./dataset/plots180/plot_" + str(i) + ".jpg"
        dataset["samples"].append((spectrogram_path, frame_paths))
        k += 1
    with open("./dataset/plots180.json", "w") as f:
        json.dump(dataset, f, indent=4)


def save_dataset(root_dir, n_frames, dim):
    data_json = {"samples": []}
    dataset = VideoGenerationDataset(n_frames=n_frames, root_dir=root_dir, dim=dim)
    save_dir = "./dataset"
    P, frames = dataset[0]
    # S = 255 * np.dstack((S, np.zeros(S.shape[:2])))
    # S = S.astype("uint8")
    plot_path = save_dir + "/plots180/plot_" + str(0) + ".jpg"
    cv2.imwrite(plot_path, P)
    frame_counter = 0
    frame_paths = []
    for i, frame in enumerate(frames):
        frame_path = save_dir + "/frames180/frame_" + str(frame_counter) + ".jpg"
        frame_paths.append(frame_path)
        # cv2.imwrite(frame_path, frame)
        frame_counter += 1
    data_json["samples"].append((plot_path, copy.deepcopy(frame_paths)))
    for i in range(1, len(dataset)):
        P, frames = dataset[i]
        plot_path = save_dir + "/plots180/plot_" + str(i) + ".jpg"
        frame_path = save_dir + "/frames180/frame_" + str(frame_counter) + ".jpg"
        # cv2.imwrite(frame_path, frames[-1])
        frame_paths.pop(0)
        frame_paths.append(frame_path)
        frame_counter += 1
        # S = 255 * np.dstack((S, np.zeros(S.shape[:2])))
        # S = S.astype("uint8")
        cv2.imwrite(plot_path, P)
        data_json["samples"].append((plot_path, copy.deepcopy(frame_paths)))
    with open("dataset/dataset.json", "w") as f:
        json.dump(data_json, f, indent=4)


# Dataset class for video generation
class VideoGenerationDataset(Dataset):
    def __init__(self, n_frames, root_dir="./episodes", dim=(256, 256), transform=None, target_transform=None):
        self.dim = dim
        self.n_frames = n_frames
        self.root_dir = root_dir
        self.paths = glob.glob(os.path.join(self.root_dir, "*.json"))
        self.paths.sort()
        self.transform = transform
        self.target_transform = target_transform

    # Get the total number of elements
    def __len__(self):
        length = 0
        for path in self.paths:
            with open(path, "r") as f:
                episode = json.load(f)
                length += len(episode["transitions"]) - self.n_frames + 1
        return length

    # Retrieve a specific item from the dataset
    def __getitem__(self, item):
        length = 0
        for i, path in enumerate(self.paths):
            with open(path, "r") as f:
                episode = json.load(f)
                length += len(episode["transitions"]) - self.n_frames + 1
                if item < length:
                    index = item - (length - len(episode["transitions"]) + self.n_frames - 1)
                    frames, frame_paths = self.load_frames(index, episode, display=False)
                    S = self.load_spectrogram(index, episode, display=False)
                    P = self.load_plot(index, episode)
                    self.load_plot(index, episode)
                    break
        if self.transform:
            S = self.transform(S)
        if self.target_transform:
            frames = self.target_transform(frames)
        # return S, torch.from_numpy(frames).double()
        return P, frames

    # Load the frames associated with the retrieved element
    def load_frames(self, index, episode, display=False):
        frames = []
        frame_paths = []
        transitions = episode["transitions"]
        H = np.array(episode["transform"])
        width = episode["width"]
        height = episode["height"]
        for i, transition in enumerate(transitions):
            if index <= i < index + self.n_frames:
                frame = cv2.imread(transition["path"])
                frame = cv2.warpPerspective(frame, H, (width, height))
                frames.append(cv2.resize(frame, self.dim, cv2.INTER_AREA))
                frame_paths.append(transition["path"])
        if display:
            for frame in frames:
                cv2.imshow("Image", frame)
                cv2.waitKey(100)
            cv2.destroyAllWindows()
        frames = np.array(frames)
        return frames, frame_paths

    # Load the spectrogram associated with the retrieved element
    def load_spectrogram(self, index, episode, display=False):
        time, displacement = load_episode(episode)
        S = spectrogram(time, displacement, dim=self.dim)
        start, end = time[index] * S.shape[0] / time[-1], time[index + self.n_frames - 1] * S.shape[0] / time[-1]
        S[:, :int(start), :] = 0
        S[:, int(end):, :] = 0
        print("Start: {}".format(int(start)))
        print("End: {}".format(int(end)))
        S = cv2.resize(S[:, int(start):int(end), :], self.dim, cv2.INTER_LINEAR)
        if display:
            plt.pcolormesh(S[:, :, 0], shading='gouraud')
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [sec]')
            plt.show()

            plt.pcolormesh(S[:, :, 1], shading='gouraud')
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [sec]')
            plt.show()
        return S

    # Load the 2D plot associated with the retrieved element
    def load_plot(self, index, episode, display=False):
        time, displacement = load_episode(episode)
        velocity = 700 * estimate_velocity(time, displacement)
        start, end = index, index + self.n_frames
        fig = plt.figure()
        fig.add_subplot(111)
        plt.plot(time[start:end], displacement[start:end], "-b", linewidth=2.5)
        plt.plot(time[start:end], velocity[start:end], "-g", linewidth=2.5)
        plt.ylim([-0.1, 3.1])
        plt.axis("off")
        fig.canvas.draw()
        plot_data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        plot_data = plot_data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plot_data = cv2.cvtColor(plot_data, cv2.COLOR_RGB2BGR)
        plot_data = cv2.resize(plot_data, self.dim, cv2.INTER_AREA)
        plt.close()
        return plot_data


class SingleImageActionMaskDataset(Dataset):
    def __init__(self, json_path="./dataset/frames180/dataset.json", dim=(3, 256, 256), high=10, low=1):
        self.dim = dim
        self.high = high
        self.low = low
        self.json_path = json_path
        with open(self.json_path, "r") as f:
            self.dataset = json.load(f)

    def __len__(self):
        return len(self.dataset["samples"])

    def __getitem__(self, item):
        frame_path, mask_path, displacement, action = self.dataset["samples"][item]
        frame = cv2.imread(frame_path) / 255
        frame = torch.from_numpy(np.transpose(frame, (2, 0, 1))).double()
        encoding = torch.tensor([displacement])
        weight = cv2.imread(mask_path)
        weight[weight != 0] = self.high
        weight[weight == 0] = self.low
        weight = torch.from_numpy(np.transpose(weight, (2, 0, 1)))
        return encoding, frame, weight, action


class DecoderDataset(Dataset):
    def __init__(self,
                 json_path="./dataset/frames180/dataset.json",
                 dim=(3, 256, 256),
                 high=10,
                 low=1):
        self.dim = dim
        self.high = high
        self.low = low
        with open(json_path, "r") as f:
            self.dataset = json.load(f)

    def __len__(self):
        return len(self.dataset["samples"])

    def __getitem__(self, item):
        frame_path, mask_path, displacement, _ = self.dataset["samples"][item]
        frame = cv2.imread(frame_path) / 255
        frame = torch.from_numpy(np.transpose(frame, (2, 0, 1))).double()
        encoding = torch.tensor([displacement], dtype=torch.double)
        weight = cv2.imread(mask_path)
        weight[weight > 0] = self.high
        weight[weight == 0] = self.low
        weight = torch.from_numpy(np.transpose(weight, (2, 0, 1)))
        return encoding, frame, weight


class EncoderDataset(Dataset):
    def __init__(self,
                 json_path="./dataset/frames512/dataset.json",
                 n_past_frames=5,
                 n_future_frames=2,
                 episode_length=180):
        self.json_path = json_path
        self.n_past_frames = n_past_frames
        self.n_future_frames = n_future_frames
        self.episode_length = episode_length
        self.transform = torchvision.transforms.ColorJitter()
        with open(self.json_path, "r") as f:
            self.dataset = json.load(f)

    def __len__(self):
        return (len(self.dataset["samples"]) // self.episode_length) * \
               (self.episode_length - self.n_past_frames - self.n_future_frames)

    def __getitem__(self, item):
        episode_id = item // (self.episode_length - self.n_past_frames - self.n_future_frames)
        start_index = item + (self.n_future_frames + self.n_past_frames) * episode_id
        end_index = start_index + self.n_past_frames + self.n_future_frames
        action = ternary(self.dataset["samples"][start_index + self.n_past_frames][3])
        action1 = torch.tensor(int(action[0]))
        action2 = torch.tensor(int(action[1]))
        action3 = torch.tensor(int(action[2]))
        frames = []
        for i in range(start_index, end_index):
            frame = self.transform(cv2.imread(self.dataset["samples"][i][0])) / 255
            frames.append(frame)
        frames = np.array(frames)
        frames = torch.from_numpy(np.transpose(frames, (0, 3, 1, 2))).double()
        return frames, action1, action2, action3


# Dataset to load the soft sensing data
class SoftSensingDataset(Dataset):
    def __init__(self, json_path="data/frames/training/annotations.json"):
        self.json_path = json_path
        with open(self.json_path, "r") as f:
            self.annotations = json.load(f)

    def __len__(self):
        return len(self.annotations["samples"])

    def __getitem__(self, item):
        state = []
        sample = self.annotations["samples"][item]
        state_paths = sample["state"]
        state_numeric = torch.tensor([sample["displacement"], sample["velocity"]], dtype=torch.double)
        for path in state_paths:
            frame = cv2.imread(path) / 255
            frame = torch.from_numpy(np.transpose(frame, (2, 0, 1))).double().unsqueeze(0)
            state.append(frame)
        state = torch.cat(state, dim=0)
        return state, state_numeric


if __name__ == "__main__":
    # dataset = SingleImageDataset()
    # dataset[190]
    # track_and_annotate()
    # save_dataset(root_dir="./episodes", n_frames=10, dim=(256, 256))
    create_json_annotations()
    # dataset = SingleImageDataset()
    # P, frames, weights = dataset[5 * 181]
    dataset = SoftSensingDataset()
