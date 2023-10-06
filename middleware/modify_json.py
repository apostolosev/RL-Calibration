import json
import copy
import time
import numpy as np

from scipy import interpolate

TERMINAL_TIME = 11
TERMINAL_VELOCITY = 0
TERMINAL_DISPLACEMENT = 2.62

# Return a timestamp
def millis():
    return round(time.time() * 1000)

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
def get_reference(t):
    tck_v, tck_d = load_reference()
    if t < TERMINAL_TIME:
        v = interpolate.splev(t, tck_v, der=0)
        d = interpolate.splev(t, tck_d, der=0)
    else:
        v = TERMINAL_VELOCITY
        d = TERMINAL_DISPLACEMENT
    return np.array([d, v], dtype=np.float64)

# Modify the recorded json file for DSS visualization
def modify_json_dss(in_path, out_path):
    # Read the input json data
    with open(in_path, "r") as f:
        in_log = json.load(f)
    time = list()
    velocity = list()
    displacement = list()
    reference_velocity = list()
    reference_displacement = list()
    numeric = in_log["numeric"]

    # Modify the imput json
    for data in numeric:
        time.append(data["time"])
        velocity.append(data["velocity"])
        displacement.append(data["displacement"])
        reference = get_reference(data["time"])
        reference_velocity.append(reference[1]) 
        reference_displacement.append(reference[0])

    out_log = dict(time=time, 
                   velocity=velocity, 
                   reference_velocity=reference_velocity)

    # Write the output
    with open(out_path, "w") as f:
        json.dump(out_log, f, indent=4)


# Modify the recorded json for Blockchain upload
def modify_json_blockchain(in_path, out_path, seg_length):
    out_log = {"id": "000", "episode_id": str(millis()), "segments": []}

    # Convert a natural number to trenary
    def ternary(n):
        if n == 0:
            return '0'.zfill(3)
        nums = []
        while n:
            n, r = divmod(n, 3)
            nums.append(str(r))
        return ''.join(reversed(nums)).zfill(3)

    # Convert an integer to action
    def int2action(n, step):
        mapping = {'0': 0, '1': step, '2': -step}
        action_str = ternary(n)
        action = [mapping[action_str[0]], mapping[action_str[1]], mapping[action_str[2]]]
        return action

    # Load the input json file
    with open(in_path, "r") as f:
        episode = json.load(f)

    # Modify the input json file
    segment = {"valve_1": 0, "valve_2": 0, "valve_3": 0, "nrmse": 0}
    transitions = episode["transitions"]
    counter = 0
    for transition in transitions:
        if counter == seg_length:
            out_log["segments"].append(copy.deepcopy(segment))
            segment = {"valve_1": 0, "valve_2": 0, "valve_3": 0, "nrmse": 0}
            counter = 0
        action = transition[1]
        segment["valve_1"] += action[0]
        segment["valve_2"] += action[1]
        segment["valve_3"] += action[2]
        segment["nrmse"] -= transition[3]
        counter += 1

    with open(out_path, "w") as f:
        json.dump(out_log, f, indent=4)


def main():
    path_in = "episodes//training/real/1675950261486.json"
    path_out = "middleware/data/blockchain/1675950261486.json"
    modify_json_blockchain(path_in, path_out, seg_length=10)


if __name__ == "__main__":
    main()
    

