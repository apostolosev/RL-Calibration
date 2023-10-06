import os
import cv2
import json
import time
import copy
import torch
import shutil
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import interpolate
from collections import namedtuple
from torchvision.transforms import ToTensor

from models import ResidualDecoder, ResidualEncoder
from processing import load_json, estimate_velocity

GenerationParameters = namedtuple("GenerationParameters",
                                  ["a1", "a2", "tc", "fs", "ymax"])


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
    plt.plot(time, velocity)
    plt.show()
    T = time[-1]
    tck = interpolate.splrep(time, velocity, k=2, s=0.005)
    time_resampled = np.linspace(0, T, int(T * fps))
    velocity_resampled = 1.21 * interpolate.splev(time_resampled, tck, der=0)
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


# Generate a piecewise linear curve
def generate_curve(a1, a2, tc, ymax, bin_length, fs=10):
    dt = 1 / fs
    t = [0]
    y = [0]
    b2 = (a1 - a2) * tc
    for i in range(10):
        t.append(copy.copy(t[-1] + dt))
        y.append(0)
    while y[-1] < ymax:
        t.append(copy.copy(t[-1] + dt))
        y.append(a1 * (t[-1] - t[10]) if t[-1] < tc else a2 * t[-1] + b2 -  a1 * t[10])
    while len(y) % bin_length != 0:
        t.append(copy.copy(t[-1] + dt))
        y.append(copy.copy(y[-1]))
    dy = []
    for i in range(len(y)):
        if i < 10:
            dy.append(0)
        elif i / fs < tc:
            dy.append(a1)
        else:
            dy.append(a2)
    for i in range(10):
        t.append(copy.copy(t[-1] + dt))
        y.append(copy.copy(y[-1]))
        dy.append(0)
    return np.array(t, dtype=np.float32), np.array(y, dtype=np.float32), np.array(dy, dtype=np.float32)


# Split a time series into segments
def split_time_series(time, displacement, velocity, segment_length=10, dim=(256, 256)):
    segments = []
    while len(displacement) % segment_length != 0:
        displacement.append(displacement[-1])
    num_segments = len(displacement) // segment_length
    for i in range(num_segments):
        start, end = i * segment_length, (i + 1) * segment_length
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
        plot_data = cv2.resize(plot_data, dim, cv2.INTER_AREA)
        plt.close()
        segments.append(plot_data)
    return segments


def predict_single(model, sample, transform, display=False):
    model.eval()
    if transform is not None:
        sample = transform(sample).double().unsqueeze(0)
    else:
        sample = sample.unsqueeze(0)
    prediction = model(sample)
    prediction = prediction.squeeze().detach().numpy()
    prediction = np.transpose(prediction, (1, 2, 0))
    video = cv2.VideoWriter("prediction.avi", 0, 1, (256, 256))
    if display:
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


def plot_inference(time, displacement, model):
    velocity = 700 * estimate_velocity(time, displacement)
    segments = split_time_series(time, displacement, velocity)
    predictions = []
    for segment in segments:
        predictions.append(predict_single(model, segment, ToTensor()))
    prediction = np.concatenate(predictions, axis=0)
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
    frames = create_frames(displacement, decoder)
    encoder.eval()
    actions = []
    for i in range(n_past_frames, frames.shape[0] - n_future_frames):
        bin = torch.from_numpy(
            np.transpose(frames[i - n_past_frames:i + n_future_frames, :, :, :], (0, 3, 1, 2))).unsqueeze(0)
        action = torch.argmax(encoder(bin))
        actions.append(int2action(action.squeeze().item()))
    return frames, actions


# Load the encoder model
def load_encoder(weights, channels=3, features=32, n_frames=10, n_actions=27, device="cpu"):
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
                 generation_params,
                 n_future_frames,
                 n_past_frames,
                 variance,
                 save_path,
                 state_length,
                 ):
        self.a1 = generation_params.a1
        self.a2 = generation_params.a2
        self.tc = generation_params.tc
        self.fs = generation_params.fs
        self.ymax = generation_params.ymax
        self.state_length = state_length
        self.n_future_frames = n_future_frames
        self.n_past_frames = n_past_frames
        self.decoder = load_decoder(decoder_weights)
        self.encoder = load_encoder(encoder_weights)
        self.save_path = save_path
        self.variance = variance
        self.timestamp = str(millis())
        self.time, self.displacement, self.velocity = reference_curve(self.fs)
        create_dir(os.path.join(self.save_path, "strean_" + self.timestamp))

    # Generate synthetic data
    def generate_data(self, display=True):
        a1 = self.a1 + self.variance[0] * (random.random() - 0.5)
        a2 = self.a2 + self.variance[1] * (random.random() - 0.5)
        tc = self.tc + self.variance[2] * (random.random() - 0.5)
        time, displacement, velocity = generate_curve(a1, a2, tc, self.ymax, self.n_future_frames + self.n_past_frames,
                                                self.fs)
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
        plt.plot(time, velocity, "-b")
        plt.xlabel("Time")
        plt.ylabel("Velocity")
        plt.show()
        for i in range(frames.shape[0]):
            frame = 255 * frames[i, :, :, :]
            frame = frame.astype("uint8")
            cv2.imwrite(os.path.join(save_path, "frame_" + str(i) + ".jpg"), frame)
        state = [os.path.join(save_path, "frame_" + str(i) + ".jpg")
                 for i in range(0, self.n_future_frames + self.n_past_frames)]
        next_state = [os.path.join(save_path, "frame_" + str(i) + ".jpg")
                      for i in range(1, self.n_future_frames + self.n_past_frames + 1)]
        transition = [state, actions[0], copy.deepcopy(next_state)]
        transitions.append(transition)
        for i in range(2, len(actions)):
            state = copy.deepcopy(next_state)
            next_state.pop(0)
            next_state.append(os.path.join(save_path, "frame_" + str(i + self.n_past_frames + n_future_frames) + ".jpg"))
            transition = [state, actions[i], copy.deepcopy(next_state)]
            transitions.append(transition)
        episode = {"transitions": transitions}
        with open(os.path.join("episodes", "generated", self.timestamp + ".json"), "w") as f:
            json.dump(episode, f, indent=4)


if __name__ == "__main__":
    encoder_weights = "models/encoders/encoder_8.pth"
    decoder_weights = "models/decoders/decoder_99.pth"
    save_path = "data/generated"
    n_past_frames = 2
    n_future_frames = 8
    state_length = 10
    variance = [0.1, 0.05, 0.1]
    generation_params = GenerationParameters(0.45, 0.12, 6.2, 10, 2.86)
    generator = EpisodeGenerator(encoder_weights=encoder_weights,
                                 decoder_weights=decoder_weights,
                                 generation_params=generation_params,
                                 n_future_frames=n_future_frames,
                                 n_past_frames=n_past_frames,
                                 variance=variance,
                                 save_path=save_path,
                                 state_length=state_length,
                                 )
    generator.create_episode()
