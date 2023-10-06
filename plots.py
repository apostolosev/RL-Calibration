import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt 
import scipy.interpolate as interpolate
from scipy.signal import butter
from scipy.signal import filtfilt


def butter_lowpass_filter(data, cutoff, order):
    normal_cutoff = cutoff
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

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

# Get the average reward for the initial state
def get_initial_reward(path, color="blue"):
    time = []
    velocity = []
    ref_velocity = []
    reward = []
    tck_v, tck_d = load_reference()
    with open(path, "r") as f:
        data = json.load(f)
        numeric = data["numeric"]
        for n in numeric:
            time.append(n["time"])
            velocity.append(n["velocity"])
            _, v = get_reference(tck_d, tck_v, n["time"])
            ref_velocity.append(v)
            reward.append(-(n["velocity"] - v) ** 2)
    plt.figure()
    plt.plot(time, velocity, "-", color=color, marker="*", linewidth=1.5, label="Velocity")
    plt.plot(time, ref_velocity, "-", color="black", marker="^", linewidth=1.8, label="Reference")
    # plt.title("Estimated Velocity")
    plt.xlabel("t (sec)", fontsize=14)
    plt.ylabel("v (m/s)", fontsize=14)
    plt.legend(fontsize=14)
    plt.show()
    avg_reward = -np.linalg.norm(np.array(velocity) - np.array(ref_velocity)) ** 2
    return reward, avg_reward


# Load the data of all recorden episodes
def load_episode_data(source_path, color):
    tck_v, tck_d = load_reference()
    json_paths = glob.glob(os.path.join(source_path, "*.json"))
    json_paths.sort()
    avg_reward = []
    reward = []
    for path in json_paths:
        time = []
        velocity = []
        displacement = []
        ref_velocity = []
        ref_displacement = []
        with open(path, "r") as f:
            data = json.load(f)
            numeric = data["numeric"]
            for n in numeric:
                t = n["time"]
                time.append(t)
                velocity.append(n["velocity"])
                displacement.append(n["displacement"])
                d, v = get_reference(tck_d, tck_v, t)
                ref_velocity.append(v)
                ref_displacement.append(d)
                reward.append(-(n["velocity"] - v) ** 2)
        velocity = butter_lowpass_filter(velocity, 0.2, 10)
        displacement = butter_lowpass_filter(displacement, 0.2, 10)
        # plt.figure()
        # plt.plot(time, velocity, "-", color=color, marker="*", linewidth=1.5, label="Velocity")
        # plt.plot(time, ref_velocity, "-", color="black", marker="^", linewidth=1.8, label="Reference")
        # # plt.title("Estimated Velocity")
        # plt.xlabel("t (sec)", fontsize=14)
        # plt.ylabel("v (m/s)", fontsize=14)
        # plt.legend()
        # plt.show()
        avg_reward.append(-np.linalg.norm(np.array(velocity) - np.array(ref_velocity)) ** 2)
    return reward, avg_reward


def main():
    # Initial recorded episode
    source_path = "log/agent/record/testing_case_2.json"
    init_reward, avg_init_reward = get_initial_reward(source_path, color="green")

    u_source_path = "log/agent/testing/untrained/case2"
    r_source_path = "log/agent/testing/real/case2"
    g_source_path = "log/agent/testing/generated/case2"
    u_reward, u_avg_reward = load_episode_data(u_source_path, color="blue")
    r_reward, r_avg_reward = load_episode_data(r_source_path, color="red")
    g_reward, g_avg_reward = load_episode_data(g_source_path, color="green")
    u_reward = init_reward + u_reward
    r_reward = init_reward + r_reward
    g_reward = init_reward + g_reward

    print(len(u_avg_reward))

    print("Agent 1 MSE 1: {}".format(u_avg_reward[0]))
    print("Agent 2 MSE 1: {}".format(r_avg_reward[0]))
    print("Agent 3 MSE 1: {}".format(g_avg_reward[0]))
    print("\n")

    print("Agent 1 MSE 2: {}".format(u_avg_reward[1]))
    print("Agent 2 MSE 2: {}".format(r_avg_reward[1]))
    print("Agent 3 MSE 2: {}".format(g_avg_reward[1]))
    print("\n")

    print("Agent 1 MSE 3: {}".format(u_avg_reward[2]))
    print("Agent 2 MSE 3: {}".format(r_avg_reward[2]))
    print("Agent 3 MSE 3: {}".format(g_avg_reward[2]))
    print("\n")

    print("Agent 1 MSE 4: {}".format(u_avg_reward[3]))
    print("Agent 2 MSE 4: {}".format(r_avg_reward[3]))
    print("Agent 3 MSE 4: {}".format(g_avg_reward[3]))
    print("\n")

    print("Agent 1 MSE 5: {}".format(u_avg_reward[4]))
    print("Agent 2 MSE 5: {}".format(r_avg_reward[4]))
    print("Agent 3 MSE 5: {}".format(g_avg_reward[4]))
    print("\n")



    u_avg_reward.insert(0, avg_init_reward)
    r_avg_reward.insert(0, avg_init_reward)
    g_avg_reward.insert(0, avg_init_reward)

    u_avg_reward, u_reward = np.array(u_avg_reward), np.array(u_reward)
    r_avg_reward, r_reward = np.array(r_avg_reward), np.array(r_reward)
    g_avg_reward, g_reward = np.array(g_avg_reward), np.array(g_reward)

    print(np.arange(0, u_avg_reward.shape[0]))

    u_avg_reward = np.interp(np.linspace(0, u_avg_reward.shape[0], len(u_reward)), 
                             np.arange(0, u_avg_reward.shape[0]), u_avg_reward)[:600]
    r_avg_reward = np.interp(np.linspace(0, r_avg_reward.shape[0], len(r_reward)), 
                             np.arange(0, r_avg_reward.shape[0]), r_avg_reward)[:600]
    g_avg_reward = np.interp(np.linspace(0, g_avg_reward.shape[0], len(g_reward)), 
                             np.arange(0, g_avg_reward.shape[0]), g_avg_reward)[:600]
    

    # Set style and size of the figure
    plt.style.use('seaborn-darkgrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(np.arange(0, u_avg_reward.shape[0]), u_avg_reward, color='#e74c3c', label='Agent-1', marker="o", linewidth=3, alpha=0.8, markevery=20)
    ax.plot(np.arange(0, r_avg_reward.shape[0]), r_avg_reward, color='#2980b9', label='Agent-2', marker="s", linewidth=3, alpha=0.8, markevery=20)
    ax.plot(np.arange(0, g_avg_reward.shape[0]), g_avg_reward, color='#27ae60', label='Agent-3', marker="^", linewidth=3, alpha=0.8, markevery=20)

    ax.vlines(x=110, ymin=-0.325, ymax=-0.13, colors="black", linewidth=0.8, linestyles="dashed")
    ax.vlines(x=220, ymin=-0.325, ymax=-0.13, colors="black", linewidth=0.8, linestyles="dashed")
    ax.vlines(x=330, ymin=-0.325, ymax=-0.13, colors="black", linewidth=0.8, linestyles="dashed")
    ax.vlines(x=440, ymin=-0.325, ymax=-0.13, colors="black", linewidth=0.8, linestyles="dashed")
    ax.vlines(x=550, ymin=-0.325, ymax=-0.13, colors="black", linewidth=0.8, linestyles="dashed")

    ax.text(x=115, y=-0.3, s="20%", fontsize=16)
    ax.text(x=225, y=-0.3, s="40%", fontsize=16)
    ax.text(x=335, y=-0.3, s="60%", fontsize=16)
    ax.text(x=445, y=-0.3, s="80%", fontsize=16)
    ax.text(x=555, y=-0.3, s="100%", fontsize=16)

    # Set axis labels and title
    ax.set_xlabel('Iterations', fontsize=14, fontweight='bold')
    ax.set_ylabel('Reward', fontsize=14, fontweight='bold')
    ax.set_title('Average episode reward', fontsize=16, fontweight='bold')

    # Add a legend to the plot
    ax.legend(fontsize=12)

    # Add grid lines to the plot
    ax.grid(alpha=0.3)


    fig.show()
    plt.show()



if __name__ == "__main__":
    main()

    