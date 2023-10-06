import os
import cv2
import json
import glob
import time
import copy
import torch
import scipy
import shutil
import random
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import interpolate
from scipy.signal import butter
from scipy.signal import filtfilt
from collections import namedtuple

from models import ResidualDecoder, ResidualEncoder

GenerationParameters = namedtuple("GenerationParameters",
                                  ["a1", "a2", "tc", "fs", "ymax"])


def create_parser():
    parser = argparse.ArgumentParser(description="Create synthetic RL episodes for offline training")
    parser.add_argument("--encoder_weights", type=str, default="models/encoders/encoder.pth")
    parser.add_argument("--decoder_weights", type=str, default="models/decoders/decoder.pth")
    parser.add_argument("--save_path", type=str, default="data/generated")
    parser.add_argument("--state_length", type=int, default=5)
    parser.add_argument("--n_past_frames", type=int, default=5)
    parser.add_argument("--n_future_frames", type=int, default=2)
    parser.add_argument("--fs", type=float, default=13.5)
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


def reference_curve(fps):
    df = pd.read_csv("reference/reference.csv")
    time = df["TIME"].to_numpy()
    velocity = df["VELOCITY"].to_numpy()
    T = time[-1]
    tck = interpolate.splrep(time, velocity, k=2, s=0.005)
    time_resampled = np.linspace(0, T, int(T * fps))
    velocity_resampled = 1.165 * interpolate.splev(time_resampled, tck, der=0)
    displacement_resampled = np.zeros_like(velocity_resampled)
    for i in range(1, len(velocity_resampled)):
        displacement_resampled[i] =  time_resampled[i] * np.mean(velocity_resampled[:i])
    return time_resampled, displacement_resampled, velocity_resampled


# Calculate the negative squared distance
def calculate_reward(x1, v1, x2, v2, alpha=0.1):
    return - (1 - alpha) * (x1 - x2) ** 2 - alpha * (v1 - v2) ** 2


# Convert an integer to trenary
def ternary(n):
    if n == 0:
        return '0'.zfill(3)
    nums = []
    while n:
        n, r = divmod(n, 3)
        nums.append(str(r))
    return ''.join(reversed(nums)).zfill(3)


# Convert an integer to a set of actions
def int2action(n, step=1):
    mapping = {'0': 0, '1': step, '2': -step}
    action_str = ternary(n)
    action = [mapping[action_str[0]], mapping[action_str[1]], mapping[action_str[2]]]
    return action


# Extend the given data to a specific length
def extend_data(time, displacement, velocity, length):
    Dt = time[1] - time[0]
    time, displacement, velocity = time.tolist(), displacement.tolist(), velocity.tolist()
    while len(time) < length:
        velocity.append(0)
        time.append(copy.copy(time[-1]) + Dt)
        displacement.append(copy.copy(displacement[-1]))
    return np.array(time, dtype=np.double), np.array(displacement, dtype=np.double), np.array(velocity, dtype=np.double)


# Butterworth low-pass filtering
def butter_lowpass_filter(data, cutoff, order):
    coeffs = butter(order, cutoff, btype='low', analog=False)
    y = filtfilt(coeffs[0], coeffs[1], data)
    return y


# Define a random walk process
def random_noise(N, mean=0, epsilon=1.0):
    X = [mean]
    for _ in range(10):
        X.append(0)
    for _ in range(1, N-20):
        dX = epsilon * (random.random() - 0.5)
        X.append(dX)
    for _ in range(10):
        X.append(0)
    X = butter_lowpass_filter(X, 0.05, 5)
    return np.array(X)


# Compute the numerical integral
def integrate(x, fs):
    ix = [0]
    for i in range(1, len(x)):
        ix.append(pint(x[:i], fs))
    return ix


# Pointwise integration
def pint(x, fs):
    return 1 / fs * np.sum(x)


# Randomization constraints
def get_l1(t1, t2):
    return 0.435 / (t2 - t1)


# Generate a randomized curve
def generate_randomized(fs):
    dt = 1 / fs
    ymax = 2.75
    bin_length = 10
    l1, l2 = 0.35, -0.41
    t1, t2, t3, t4 = 0.21, 1.45, 5.8, 6.52
    _t2 = t2 + 0.8 * t2 * (random.random() - 0.1)
    _l2 = l2 + 0.1 * l2 * (random.random() - 0.5)
    _t3 = t3 + 0.05 * t3 * (random.random() - 0.5)
    _t4 = t4 + 0.05 * t4 * (random.random() - 0.5)
    _l1 = get_l1(t1, _t2)
    _l2 = l2 + 0.5 * l2 * (random.random() - 0.5)
    return generate_curve_1(t1=t1, t2=_t2, t3=_t3, t4=_t4, l1=_l1, l2=_l2, dt=dt, ymax=ymax, bin_length=bin_length)


def generate_curve_1(t1, t2, t3, t4, l1, l2, dt, ymax, bin_length):
    t = [0]
    x = [0]
    ix = [0]
    while ix[-1] < ymax:
        t.append(copy.copy(t[-1]) + dt)
        if t[-1] < t1:
            x.append(0)
            ix.append(pint(x, 1 / dt))
        elif t[-1] < t2:
            x.append(l1 * (copy.copy(t[-1]) - t1))
            ix.append(pint(x, 1 / dt))
        elif t[-1] < t3:
            x.append(l1 * (t2 - t1))
            ix.append(pint(x, 1 / dt))
        elif t[-1] < t4:
            x.append(l2 * (copy.copy(t[-1]) - t3) + l1 * (t2 - t1))
            ix.append(pint(x, 1 / dt))
        else:
            x.append(l2 * (t4 - t3) + l1 * (t2 - t1))
            ix.append(pint(x, 1 / dt))

    for _ in range(10):
        t.append(copy.copy(t[-1]) + dt)
        x.append(0)
        ix.append(ix[-1])

    while len(x) % bin_length != 0:
        t.append(copy.copy(t[-1]) + dt)
        x.append(0)
        ix.append(ix[-1])

    window = scipy.signal.windows.parzen(len(x))
    x_noise = window * random_noise(len(x), 0, 0.3)
    ix_noise = integrate(x_noise, 1 / dt)
    t = np.array(t, dtype=np.float32)
    x = np.array(x, dtype=np.float32) + np.array(x_noise, dtype=np.float32)
    ix = np.array(ix, dtype=np.float32) + np.array(ix_noise, dtype=np.float32)

    return t, ix, x


def transform_parameters(t1, ah, vh, th, al, vl):
    t2 = vh / ah + t1
    t3 = t2 + th
    t4 = (vh - vl) / al + t3
    return t2, t3, t4


def generate_curve_2(t1, ah, vh, th, al, vl, dt, ymax, bin_length):
    t2, t3, t4 = transform_parameters(t1, ah, vh, th, al, vl)
    return generate_curve_1(t1, t2, t3, t4, ah, -al, dt, ymax, bin_length)


def predict_single(model, sample, transform, display=False):
    model.eval()
    if transform is not None:
        sample = transform(sample).double().unsqueeze(0)
    else:
        sample = sample.unsqueeze(0)
    prediction = model(sample)
    prediction = prediction.squeeze().detach().numpy()
    prediction = np.transpose(prediction, (1, 2, 0))
    if display:
        video = cv2.VideoWriter("prediction.avi", 0, 1, (256, 256))
        cv2.namedWindow("Prediction", cv2.WINDOW_NORMAL)
        for i in range(prediction.shape[0]):
            pred = 255 * prediction[i, :, :, :]
            video.write(pred.astype("uint8"))
            cv2.imwrite("/home/apostolos/PycharmProjects/Generative-RL/test/frame_" + str(i) + ".jpg",
                        pred.astype("uint8"))
            cv2.imwrite("prediction.jpg", pred.astype("uint8"))
            cv2.imshow("Prediction", prediction[i, :, :, :])
            cv2.waitKey(1)
        cv2.destroyAllWindows()
        video.release()
    return prediction


# Generate frames given a displacement curve
def create_frames(displacement, model):
    predictions = []
    # cv2.namedWindow("Prediction", cv2.WINDOW_NORMAL)
    for i in range(len(displacement)):
        encoding = torch.tensor([displacement[i]])
        prediction = predict_single(model, encoding, transform=None)
        predictions.append(np.expand_dims(prediction, axis=0))
    predictions = np.concatenate(predictions, axis=0)
    return predictions


# Generate frames and actions given a displacement curve
def predict_actions(displacement, encoder, decoder, n_future_frames, n_past_frames):
    mapping = {0: 0, 1: 3, 2: -3}
    frames = create_frames(displacement, decoder)
    encoder.eval()
    actions = []
    for i in range(n_past_frames, frames.shape[0] - n_future_frames):
        bin = torch.from_numpy(
            np.transpose(frames[i - n_past_frames:i + n_future_frames, :, :, :], (0, 3, 1, 2))).unsqueeze(0)
        action1, action2, action3 = encoder(bin)
        action1, action2, action3 = torch.argmax(action1), torch.argmax(action2), torch.argmax(action3)
        action1, action2, action3 = action1.squeeze().item(), action2.squeeze().item(), action3.squeeze().item()
        actions.append([mapping[action1], mapping[action2], mapping[action3]])
    return frames, actions


# Load the encoder model
def load_encoder(weights, channels=3, features=32, n_frames=7, n_actions=3, device="cpu"):
    encoder = ResidualEncoder(channels, features, n_frames, n_actions)
    encoder.load_state_dict(torch.load(weights, map_location=device))
    return encoder


# Load the decoder model
def load_decoder(weights, channels=3, features=32, device="cpu"):
    decoder = ResidualDecoder(channels, features)
    decoder.load_state_dict(torch.load(weights, map_location=device))
    return decoder


# Wrapper class to create synthetic episodes
class EpisodeGenerator:
    def __init__(self,
                 encoder_weights,
                 decoder_weights,
                 n_future_frames,
                 n_past_frames,
                 save_path,
                 state_length,
                 fs
                 ):
        self.fs = fs
        self.state_length = state_length
        self.n_future_frames = n_future_frames
        self.n_past_frames = n_past_frames
        self.decoder = load_decoder(decoder_weights)
        self.encoder = load_encoder(encoder_weights, n_frames=n_future_frames + n_past_frames)
        self.save_path = save_path
        self.timestamp = str(millis())
        self.time, self.displacement, self.velocity = reference_curve(self.fs)
        create_dir(os.path.join(self.save_path, "strean_" + self.timestamp))

    # Generate synthetic data
    def generate_data(self, display=False):
        time, displacement, velocity = generate_randomized(self.fs)
        frames, actions = predict_actions(displacement, self.encoder, self.decoder, self.n_future_frames,
                                          self.n_past_frames)
        if display:
            self.display_generated_curve(time, displacement, velocity)
        return frames, actions, time, displacement, velocity

    # Display the generated data
    def display_generated_curve(self, time, displacement, velocity):
        fig1 = plt.figure()
        plt.plot(time, displacement, label="Generated")
        plt.plot(self.time, self.displacement, label="Reference")
        plt.title("Generated Displacement Curve")
        plt.xlabel("Time (s)")
        plt.ylabel("Displacement (m)")
        plt.legend()
        fig1.show()

        fig2 = plt.figure()
        plt.plot(time, velocity, label="Generated")
        plt.plot(self.time, self.velocity, label="Reference")
        plt.title("Generated Velocity Curve")
        plt.xlabel("Time (s)")
        plt.ylabel("Velocity (m/s)")
        plt.legend()
        fig2.show()
        plt.show()

    # Write the generated data to a json file
    def create_episode(self):
        transitions = []
        save_path = os.path.join(self.save_path, "strean_" + self.timestamp)
        frames, actions, time, displacement, velocity = self.generate_data()
        if len(self.displacement) > len(displacement):
            time, displacement, velocity = extend_data(time, displacement, velocity, length=len(self.time))
        elif len(self.displacement < len(displacement)):
            self.time, self.displacement, self.velocity = extend_data(self.time, self.displacement, self.velocity, length=len(time))
        self.display_generated_curve(time, displacement, velocity)
        for i in range(frames.shape[0]):
            frame = 255 * frames[i, :, :, :]
            frame = frame.astype("uint8")
            cv2.imwrite(os.path.join(save_path, "frame_" + str(i) + ".jpg"), frame)
        state = [os.path.join(save_path, "frame_" + str(i) + ".jpg")
                 for i in range(0, self.n_past_frames)]
        next_state = [os.path.join(save_path, "frame_" + str(i) + ".jpg")
                      for i in range(1, self.n_past_frames + 1)]
        reward = calculate_reward(displacement[1], velocity[1], self.displacement[1], self.velocity[1])
        transition = [copy.deepcopy(state), actions[0], copy.deepcopy(next_state),
                      reward, float(displacement[1]), float(velocity[1])]
        transitions.append(transition)
        for i in range(2, len(actions)):
            state = copy.deepcopy(next_state)
            next_state.pop(0)
            next_state.append(os.path.join(save_path, "frame_" + str(i + self.n_past_frames) + ".jpg"))
            reward = calculate_reward(displacement[i], velocity[i], self.displacement[i], self.velocity[i])
            transition = [state, actions[i], copy.deepcopy(next_state), reward, float(displacement[i]), float(velocity[i])]
            transitions.append(transition)
        episode = {"transitions": transitions}
        with open(os.path.join("episodes", "generated", self.timestamp + ".json"), "w") as f:
            json.dump(episode, f, indent=4)


def generate_data(save_path, a1, a2, tc, ymax, fs, model):
    bin_length = 10
    desired_length = 250
    timestamp = str(millis())
    save_dir = os.path.join(save_path, "gstream_" + timestamp)
    create_dir(save_dir)
    create_dir(os.path.join(save_dir, "transformed"))
    time, displacement, velocity = generate_randomized(fs)
    while len(time) > desired_length:
        time, displacement, velocity = generate_randomized(fs)
    time, displacement, velocity = extend_data(time, displacement, velocity, desired_length)
    plt.plot(time, velocity, "-b")
    plt.xlabel("Time (s)", fontsize=14)
    plt.ylabel("Velocity (m/s)", fontsize=14)
    plt.show()
    frames = create_frames(displacement, model)
    paths = []
    for i in range(frames.shape[0]):
        timestamp_i = str(millis())
        frame = 255 * frames[i, :, :, :]
        path = os.path.join(os.path.join(save_dir, "transformed"),  "webcam_" + timestamp_i + ".jpg")
        paths.append(path)
        cv2.imwrite(path, frame.astype("uint8"))
    with open(os.path.join(save_dir, "refined_data_.json"), "w") as f:
        data = {"paths": paths, "time": time.tolist(), "displacement": displacement.tolist(), "velocity": velocity.tolist()}
        json.dump(data, f, indent=4)


def main():
    parser = create_parser()
    args = parser.parse_args()
    generator = EpisodeGenerator(encoder_weights=args.encoder_weights,
                                 decoder_weights=args.decoder_weights,
                                 n_future_frames=args.n_future_frames,
                                 n_past_frames=args.n_past_frames,
                                 save_path=args.save_path,
                                 state_length=args.state_length,
                                 fs=args.fs
                                 )
    generator.create_episode()


if __name__ == "__main__":
    main()
    # decoder = ResidualDecoder().double()
    # decoder.load_state_dict(torch.load("models/decoders/decoder.pth", map_location="cpu"))
    # fs = 14
    # time, displacement, velocity = reference_curve(fs)
    # plt.figure(figsize=(8,6))
    # plt.plot(time, velocity, "-k", label="Reference", linewidth=2.5)
    # colors = ["b", "c", "g", "m", "r"]
    # for i in range(5):
    #     t, x, v = generate_randomized(fs)
    #     plt.plot(t, v, alpha=0.8, color=colors[i], linewidth=1.5)
    # plt.legend()
    # plt.grid(axis="y")
    # plt.show()

    # generate_data("data/frames/training", a1, a2, tc, ymax, fs, decoder)
    # list_frames = glob.glob("/home/apostolos/PycharmProjects/Generative-RL/data/frames/training/stream_1661426393438/original/*")
    # print(list_frames)
    # list_frames.sort(key=lambda s: int(s.split("/")[-1].split(".")[0].split("_")[1]))
    # out = cv2.VideoWriter(
    #     "test.mp4",
    #     cv2.VideoWriter_fourcc(*'mp4v'), 13.5, (960, 720))
    
    # for filename in list_frames:
    #     img = cv2.imread(filename)
    #     out.write(img)
    # #
    # out.release()
