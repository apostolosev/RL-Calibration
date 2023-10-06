import copy
import scipy
import random
import numpy as np

from scipy.signal import butter
from scipy.signal import filtfilt


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


if __name__ == "__main__":
    fs = 13.5
    dt = 1 / fs
    ymax = 2.75
    bin_length = 10
    l1, l2 = 0.35, -0.41
    t1, t2, t3, t4 = 0.21, 1.45, 5.8, 6.52
    t, d, v = generate_curve_1(t1, t2, t3, t4, l1, l2, dt, ymax, bin_length)