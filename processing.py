import os
import cv2
import copy
import json
import glob
import time
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import interpolate
from scipy.signal import butter
from scipy.signal import filtfilt

from models import DeepQNet


# Return a timestamp
def millis():
    return round(time.time() * 1000)


# Generate a piecewise linear curve
def generate_curve(a1, a2, tc, ymax, bin_length, fs=10.0):
    dt = 1 / fs
    t = [0]
    y = [0]
    b2 = (a1 - a2) * tc
    for i in range(30):
        t.append(copy.copy(t[-1] + dt))
        y.append(0)
    while y[-1] < ymax:
        t.append(copy.copy(t[-1] + dt))
        y.append(a1 * (t[-1] - t[30]) if t[-1] < tc else a2 * t[-1] + b2 -  a1 * t[30])
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
    for i in range(30):
        t.append(copy.copy(t[-1] + dt))
        y.append(copy.copy(y[-1]))
        dy.append(0)
    return np.array(t, dtype=np.float32), np.array(y, dtype=np.float32), np.array(dy, dtype=np.float32)


def calculate_reward(state, alpha=0.5):
    return - alpha * (state[1] - state[2]) ** 2 - (1 - alpha) * (state[3] - state[4]) ** 2


def derivative(t, x):
    return (np.mean(t * x) - np.mean(x) * np.mean(t)) / (np.mean(t ** 2) - np.mean(t) ** 2)


def butter_lowpass_filter(data, cutoff, order):
    # Get the filter coefficients
    r = butter(order, cutoff, btype='low', analog=False)
    y = filtfilt(r[0], r[1], data)
    return y


def estimate_velocity(time, displacement, h=10):
    velocity = np.zeros_like(displacement, dtype=np.float)
    displacement_pad = np.pad(displacement, (int(h / 2), int(h / 2)), mode="edge") / 100
    time_pad = np.pad(time, (int(h / 2), int(h / 2)), mode="reflect", reflect_type="odd")
    for i in range(0, len(time)):
        ti = time_pad[i:i + h]
        di = displacement_pad[i:i + h]
        velocity[i] = derivative(ti, di)
    return velocity


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


def reference_curve(fps):
    df = pd.read_csv("reference/reference.csv")
    time = df["TIME"].to_numpy()
    velocity = df["VELOCITY"].to_numpy()
    displacement = np.zeros_like(velocity)
    for i in range(1, len(velocity)):
        displacement[i] = time[i] * np.mean(velocity[:i])
    plt.plot(displacement)
    plt.show()
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


def load_json(path):
    velocity = []
    displacement = []
    with open(path, "r") as f:
        episode = json.load(f)
        transitions = episode["transitions"]
        for transition in transitions:
            # state = torch.tensor(transition["transition"][0], dtype=torch.float)
            displacement.append(transition[4])
            velocity.append(transition[5])
    return velocity, displacement


def load_data(path):
    time = []
    velocity = []
    displacement = []
    ref_velocity = []
    ref_displacement = []
    tck_v, tck_d = load_reference()
    with open(path, "r") as f:
        episode = json.load(f)
        numeric = episode["numeric"]
        for num in numeric:
            time.append(num["time"])
            velocity.append(num["velocity"])
            displacement.append(num["displacement"])
            d, v = get_reference(num["time"], tck_v, tck_d)
            ref_velocity.append(v)
            ref_displacement.append(d)

    # Display the estimated displacement
    plt.figure()
    plt.plot(time, displacement, "-b*", linewidth=1.5, label="Displacement")
    plt.plot(time, ref_displacement, "-k^", linewidth=1.8, label="Reference")
    plt.title("Estimated Displacement")
    plt.xlabel("t (sec)")
    plt.ylabel("z (m)")
    plt.legend()

    # Display the estimated velocity
    plt.figure()
    plt.plot(time, velocity, "-r*", linewidth=1.5, label="Velocity")
    plt.plot(time, ref_velocity, "-k^", linewidth=1.8, label="Reference")
    plt.title("Estimated Velocity")
    plt.xlabel("t (sec)")
    plt.ylabel("v (m/s)")
    plt.legend()
    plt.show()


def transform_frames(path):
    lim_width = 310
    lim_height = 700
    dim = (512, 512)
    fps = 14

    # Load the frames paths
    warp = np.load("data/calibration/warp.npy")
    scale = np.load("data/calibration/scale.npy")
    paths = glob.glob(path + "original/*.jpg")
    paths.sort()

    # Initialize the trakker
    tracking_params = cv2.TrackerCSRT_Params()
    setattr(tracking_params, "admm_iterations", 600)
    setattr(tracking_params, "template_size", 16)
    tracker = cv2.TrackerCSRT_create()
    frame = cv2.imread(paths[0])
    frame = cv2.warpPerspective(frame, warp, (lim_width, lim_height))
    bbox0 = cv2.selectROI(frame, False)
    starting_position = np.array([scale[0, 0] * (bbox0[0] + bbox0[2] / 2), scale[1, 1] * (bbox0[1] + bbox0[3] / 2)])
    ok = tracker.init(frame, bbox0)
    displacement = [0]
    transformed_frame_path = path + "transformed/" + paths[0].split("/")[-1]
    transformed_frame_paths = [transformed_frame_path]
    cv2.imwrite(transformed_frame_path, cv2.resize(frame, dim, cv2.INTER_AREA))
    # Track weights for the remaining frames
    for i, frame_path in enumerate(paths[1:]):
        frame = cv2.imread(frame_path)
        frame = cv2.warpPerspective(frame, warp, (lim_width, lim_height))
        ok, bbox1 = tracker.update(frame)
        bbox = (bbox0[0], bbox1[1], bbox0[2], bbox0[3])
        position = np.array([scale[0, 0] * (bbox[0] + bbox[2] / 2), scale[1, 1] * (bbox[1] + bbox[3] / 2)])
        displacement.append(np.linalg.norm(position - starting_position) / 100)
        transformed_frame_path = path + "transformed/" + frame_path.split("/")[-1]
        transformed_frame_paths.append(transformed_frame_path)
        # data["data"].append({"path": transformed_frame_path, "displacement": displacement[i], ""})
        cv2.imwrite(transformed_frame_path, cv2.resize(frame, dim, cv2.INTER_AREA))
        if ok:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 10, 20)
        cv2.imshow("Frame", frame)
        cv2.waitKey(1)
        frame = cv2.resize(frame, (512, 512), cv2.INTER_AREA)
        cv2.imwrite("test/frame_" + str(i) + ".jpg", frame)
    cv2.destroyAllWindows()

    time = np.linspace(0, len(displacement) / fps, len(displacement))
    velocity = estimate_velocity(time, displacement, h=12)
    velocity = butter_lowpass_filter(velocity, 0.5, 3)
    data = {"paths": transformed_frame_paths, "time": time.tolist(), "displacement": displacement, "velocity": velocity.tolist()}
    with open(path + "data.json", "w") as f:
        json.dump(data, f, indent=4)

    plt.plot(time, velocity, "-b", linewidth=1.4)
    plt.title("Estimated Velocity", fontsize=16)
    plt.xlabel("$t$ ($s$)", fontsize=16)
    plt.ylabel("$v$ ($m/s$)", fontsize=16)
    plt.show()

    plt.plot(time, displacement, "-b", linewidth=1.4)
    plt.title("Estimated Displacement", fontsize=16)
    plt.xlabel("$t$ ($s$)", fontsize=16)
    plt.ylabel("$z$ ($m$)", fontsize=16)
    plt.show()
    return time, displacement, velocity, transformed_frame_paths


def create_synthetic(path, d_displacement, d_velocity):
    # directory_paths = glob.glob("data/frames/training/*")
    # synthetic_paths = len(d_displacement) * ["."]
    # synthetic_displacement = len(d_displacement) * [float("inf")]
    # for dir_path in directory_paths:
    #     with open(dir_path + "/data.json") as f:
    #         data = json.load(f)
    #         paths = data["paths"]
    #         displacement = np.array(data["displacement"])
    #         displacement = 2.86 / np.max(displacement) * displacement
    #         for i in range(len(d_displacement)):
    #             j = min(np.argmin(np.abs(displacement - d_displacement[i])), len(d_displacement) - 1)
    #             if abs(displacement[j] - d_displacement[i]) < abs(synthetic_displacement[i] - d_displacement[i]):
    #                 synthetic_displacement[i] = displacement[j]
    #                 synthetic_paths[i] = paths[j]
    # plt.plot(d_displacement)
    # plt.plot(synthetic_displacement)
    # plt.show()
    with open(path, "r") as f:
        data = json.load(f)
        paths = data["paths"]
        displacement = np.array(data["displacement"])
        displacement = 2.86 / np.max(displacement) * displacement
        synthetic_paths = []
        synthetic_displacement = []
        for i in range(len(d_displacement)):
            j = np.argmin(np.abs(displacement - d_displacement[i]))
            synthetic_displacement.append(displacement[j])
            synthetic_paths.append(paths[j])
    for spath in synthetic_paths:
        frame = cv2.imread(spath)
        cv2.imshow("Frame", frame)
        cv2.waitKey(100)
    cv2.destroyAllWindows()
    plt.plot(synthetic_displacement)
    plt.plot(d_displacement)
    plt.show()

    data = {"paths": synthetic_paths, "displacement": synthetic_displacement, "velocity": d_velocity.tolist()}
    epath = "data/frames/training/"+ str(millis()) + ".json"
    with open(epath, "w") as f:
        json.dump(data, f, indent=4)


def change_ext(path):
    with open(path, "r") as f:
        episode = json.load(f)
        transitions = episode["transitions"]
        transition = copy.deepcopy(transitions[0])
        transition[0] = [transition[0][0] for _ in range(10)]
        transition[2] = [transition[0][0] for _ in range(10)]
        transition[4] = 0.1 * random.random()
        transition[5] = 0
        for _ in range(15):
            transitions.insert(0, copy.deepcopy(transition))
        # for transition in transitions:
        #     transition[1] = [3 * a for a in transition[1]]
        # for i, spath in enumerate(state_paths):
        #     spath = spath.split(".")[0] + ".bmp"
        #     state_paths[i] = spath
        # for i, spath in enumerate(next_state_paths):
        #     spath = spath.split(".")[0] + ".bmp"
        #     next_state_paths[i] = spath
    with open(path.split(".")[0] + "_.json", "w") as f:
        json.dump(episode, f, indent=4)


def process_json(path):
    time, displacement = load_json(path)
    velocity = estimate_velocity(time, displacement)
    velocity = velocity.tolist()
    velocity.append(0)
    with open(path, "r") as f:
        episode = json.load(f)
    with open(path.split(".")[0] + "_.json", "w") as f:
        # episode["width"] = 310
        # episode["height"] = 700
        # episode["transform"] = np.load("./data/calibration/warp.npy").tolist()
        transitions = episode["transitions"]
        for i, transition in enumerate(transitions):
            episode["transitions"][i]["transition"][0][0][3] = transition["transition"][0][0][3] / 100
            episode["transitions"][i]["transition"][0][0][1] = velocity[i]
            episode["transitions"][i]["transition"][2][0][3] = transition["transition"][2][0][3] / 100
            episode["transitions"][i]["transition"][2][0][1] = velocity[i + 1]
            episode["transitions"][i]["transition"][3][0] = calculate_reward(
                episode["transitions"][i]["transition"][2][0])
        json.dump(episode, f, indent=4)


def extend_json(path, max_length):
    with open(path, "r") as f:
        episode = json.load(f)
        transitions = episode["transitions"]
        while len(transitions) < max_length:
            transition = copy.deepcopy(transitions[-1])
            transition = transition
            dt = transition["transition"][2][0][0] - transition["transition"][0][0][0]
            transition["transition"][0][0][0] += dt
            transition["transition"][2][0][0] += dt
            transitions.append(transition)
        episode["transitions"] = transitions
    with open(path.split(".")[0] + "_.json", "w") as f:
        json.dump(episode, f, indent=4)


def modify_json(path, seg_length):
    klee_json = {"id": "000", "episode_id": str(millis()), "segments": []}


    def ternary(n):
        if n == 0:
            return '0'.zfill(3)
        nums = []
        while n:
            n, r = divmod(n, 3)
            nums.append(str(r))
        return ''.join(reversed(nums)).zfill(3)

    def int2action(n, step):
        mapping = {'0': 0, '1': step, '2': -step}
        action_str = ternary(n)
        action = [mapping[action_str[0]], mapping[action_str[1]], mapping[action_str[2]]]
        return action

    with open(path, "r") as f:
        segment = {"valve_1": 0, "valve_2": 0, "valve_3": 0, "nrmse": 0}
        episode = json.load(f)
        transitions = episode["transitions"]
        counter = 0
        for transition in transitions:
            if counter == seg_length:
                klee_json["segments"].append(copy.deepcopy(segment))
                segment = {"valve_1": 0, "valve_2": 0, "valve_3": 0, "nrmse": 0}
                counter = 0
            action = int2action(transition["transition"][1][0], step=3)
            segment["valve_1"] += action[0]
            segment["valve_2"] += action[1]
            segment["valve_3"] += action[2]
            segment["nrmse"] -= transition["transition"][3][0]
            counter += 1
    with open("klee_json.json", "w") as f:
        json.dump(klee_json, f, indent=4)


def move_frames(path):
    frame_paths = glob.glob(os.path.join(path, "transformed", "*"))
    for frame_path in frame_paths:
        frame = cv2.imread(frame_path)
        new_path = os.path.join(path, frame_path.split("/")[-1].split(".")[0] + ".bmp")
        cv2.imwrite(new_path, frame)


def modify_episode(path):
    sensor = DeepQNet(channels=3, features=32).double()
    sensor.load_state_dict(torch.load("models/agent/soft_25.pth", map_location="cpu"))
    sensor.eval()
    with open(path, "r") as f:
        episode = json.load(f)
        transitions = episode["transitions"]
        displacement = []
        velocity = []
        for transition in transitions:
            # transition[0] = transition[0][5:]
            # transition[2] = transition[2][5:]
            state = []
            for frame_path in transition[0]:
                frame = torch.from_numpy(np.transpose(cv2.resize(cv2.imread(frame_path), (512, 512), cv2.INTER_AREA), (2, 0, 1)) / 255).unsqueeze(0)
                state.append(frame)
            state = torch.cat(state, dim=0).unsqueeze(0)
            _, state_numeric = sensor(state)
            displacement.append(state_numeric.squeeze()[0].item())
            velocity.append(state_numeric.squeeze()[1].item())
        plt.plot(displacement)
        plt.show()
        plt.plot(velocity)
        plt.show()

    with open(path.split(".")[0] + "_.json", "w") as f:
        episode["transitions"] = transitions
        json.dump(episode, f, indent=4)


def display_json(path):
    time = []
    velocity = []
    displacement = []
    reference_velocity = []
    reference_displacement = []
    with open(path, "r") as f:
        episode = json.load(f)
        transitions = episode["transitions"]
        for transition in transitions:
            state = torch.tensor(transition["transition"][0], dtype=torch.float)
            time.append(state[0][0])
            velocity.append(state[0][1])
            displacement.append(state[0][3])
            reference_velocity.append(state[0][2])
            reference_displacement.append(state[0][4])
    # Displacement plot
    plt.plot(time, displacement, "-b", label="Displacement")
    plt.plot(time, reference_displacement, "-k", label="Reference")
    plt.legend()
    plt.show()
    # Velocity plot
    plt.plot(time, velocity, "-b", label="Velocity")
    plt.plot(time, reference_velocity, "-k", label="Reference")
    plt.legend()
    plt.show()
    return time, displacement, velocity


def move_json_masks():
    boxes = glob.glob("./dataset/frames180/*.npy")
    boxes.sort()
    box_0 = np.load(boxes[0])
    center_0 = [box_0[0] + box_0[2] / 2, box_0[1] + box_0[3] / 2]
    with open("./dataset/frames180/frame_0.json", "r") as f1:
        mask = json.load(f1)
    shapes = mask["shapes"]
    moved_shapes = copy.deepcopy(shapes)
    for i, box_path in enumerate(boxes):
        box = np.load(box_path)
        center = [box[0] + box[2] / 2, box[1] + box[3] / 2]
        displacement = [center[0] - center_0[0], center[1] - center_0[1]]
        for i, shape in enumerate(shapes):
            moved_points = []
            points = shape["points"]
            for pnt in points:
                moved_points.append([min(pnt[0] + displacement[0], 256), min(pnt[1] + displacement[1], 256)])
            moved_shapes[i]["points"] = moved_points
        mask_copy = copy.deepcopy(mask)
        mask_copy["shapes"] = copy.deepcopy(moved_shapes)
        mask_copy["imagePath"] = "frame_" + box_path.split(".")[1].split("_")[1] + ".jpg"
        if i != 0:
            with open(box_path[:-4] + ".json", "w") as f2:
                json.dump(mask_copy, f2, indent=4)


def create_binary_masks():
    masks = glob.glob("./dataset/frames180/*.json")
    for i, mask_path in enumerate(masks):
        print(i)
        with open(mask_path, "r") as f:
            mask = json.load(f)
            points = mask["shapes"][0]["points"]
            bg1 = mask["shapes"][1]["points"]
            bg2 = mask["shapes"][2]["points"]
            bg3 = mask["shapes"][3]["points"]
            bg4 = mask["shapes"][4]["points"]
            bg5 = mask["shapes"][5]["points"]
            height, width = mask["imageHeight"], mask["imageWidth"]
            raw_dist = np.zeros((mask["imageHeight"], mask["imageWidth"]), dtype=np.uint8)
            mask_name = mask["imagePath"].split(".")[0] + "_mask.jpg"
            start = time.time()
            for i in range(height):
                for j in range(width):
                    if cv2.pointPolygonTest(np.array(points, dtype=np.float32), (j, i), False) >= 0:
                        raw_dist[i, j] = 255
                        if cv2.pointPolygonTest(np.array(bg1, dtype=np.float32), (j, i), False) > 0:
                            raw_dist[i, j] = 0
                        if cv2.pointPolygonTest(np.array(bg2, dtype=np.float32), (j, i), False) > 0:
                            raw_dist[i, j] = 0
                        if cv2.pointPolygonTest(np.array(bg3, dtype=np.float32), (j, i), False) > 0:
                            raw_dist[i, j] = 0
                        if cv2.pointPolygonTest(np.array(bg4, dtype=np.float32), (j, i), False) > 0:
                            raw_dist[i, j] = 0
                        if cv2.pointPolygonTest(np.array(bg5, dtype=np.float32), (j, i), False) > 0:
                            raw_dist[i, j] = 0
            end = time.time()
            print("Elapsed time: {}".format(end - start))
            raw_dist_color = np.empty((height, width, 3))
            for i in range(3):
                raw_dist_color[:, :, i] = raw_dist
            # cv2.imshow("Mask", raw_dist)
            # cv2.waitKey(0)
            cv2.imwrite(os.path.join("./dataset/frames180", mask_name), raw_dist_color)
    # cv2.desrtoyAllWindows()


def create_list_of_images(path):
    list_of_frames = []
    for i in range(300,330):
        img = path +'/'+"frame{:05}.jpg".format(i)
        # img = path +'/'+"{}.jpg".format(i)
        print(img)
        list_of_frames.append(img)
    return list_of_frames


def display_rewards():
    avg_reward = []
    tck_d, tck_v = load_reference()
    paths = glob.glob("episodes/real/*")
    # paths.sort(key=lambda s: int(s.split(".")[0].split("/")[-1]))
    paths = paths[5:]
    for path in paths:
        reward = []
        velocity = []
        reference = []
        with open(path, "r") as f:
            data = json.load(f)
            numeric = data["numeric"]
            for n in numeric:
                t = n["time"]
                r = n["reward"]
                v = n["displacement"]
                velocity.append(v)
                ref_d, ref_v = get_reference(t, tck_d, tck_v)
                reference.append(ref_d)
                reward.append(r)
        avg_reward.append(np.sum(reward) / 100)
        plt.plot(reference, "-k")
        plt.plot(velocity, "-b")
        plt.show()
        print(np.sum(velocity) / 13.5)
    plt.plot(avg_reward, "-b*", linewidth=1.5)
    plt.xlabel("Episode", fontsize=14)
    plt.title("Average Episode Reward", fontsize=16)
    # plt.grid()
    plt.show()



if __name__ == "__main__":
    # display_json("/home/apostolos/PycharmProjects/Generative-RL/episodes/episode_0_.json")
    # modify_json("/home/apostolos/PycharmProjects/Generative-RL/episodes/episode_15.json", 20)
    # create_binary_masks()
    # extend_json("/home/apostolos/PycharmProjects/Generative-RL/episodes/episode_16.json", max_length=190)
    # create_binary_masks()
    # transform_frames("data/frames/training/stream_1661425320254/")
    display_rewards()
    # t, d, v = reference_curve(15)
    # plt.plot(t, v, "-b")
    # plt.xlabel("Time", fontsize=16)
    # plt.ylabel("Velocity", fontsize=16)
    # plt.show()

    # ymax = 2.86
    # a1 = 0.42 + 0.04 * random.random()
    # a2 = 0.12 + 0.04 * random.random()
    # tc = 7.20 + 1.2 * random.random()
    # t, d, v = generate_curve(a1, a2, tc, ymax, bin_length=10, fs=13.5)
    # create_synthetic("data/frames/training/stream_1661425135934/data.json", d, v)
    # time, displacement, velocity = display_json("/home/apostolos/PycharmProjects/klee/log/visit_6/episode_5_.json")
    # transform_frames("data/frames/training/stream_1661425701346/")
    # modify_episode("/home/apostolos/PycharmProjects/Generative-RL/episodes/real/episode_10_.json")
    # move_frames("/home/apostolos/PycharmProjects/Generative-RL/data/real/stream_1661428536148")
    # v, d = load_json("/home/apostolos/PycharmProjects/Generative-RL/episodes/real/episode_3.json")
    # plt.figure()
    # plt.plot(v)
    # plt.figure()
    # plt.plot(d)
    # plt.show()
    # reference_curve(14)
    # t = np.linspace(0, 10, 512)
    # ref_displacement = []
    # ref_velocity = []
    # tck_d, tck_v = load_reference()
    # for i in range(len(t)):
    #     d, v = get_reference(t[i], tck_d, tck_v)
    #     ref_displacement.append(d)
    #     ref_velocity.append(v)
    # plt.plot(ref_displacement)
    # plt.show()
