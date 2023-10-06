import cv2
import numpy as np
import matplotlib.pyplot as plt

from scipy import signal
from processing import load_json


# Calculate the integral of a signal
def integral(t, x):
    y = np.zeros_like(x)
    for i in range(1, len(x)):
        y[i] = np.trapz(x[:i], dx=t[i] - t[i - 1])
    return y


# Get the Butterworth filter coefficients
def butter_lowpass(fc, fs, order=5):
    nyq = 0.5 * fs
    fc_n = fc / nyq
    b, a = signal.butter(order, fc_n, btype="low", analog=False)
    return b, a


# Low-pass filter a signal
def lowpass_filter(data, fc, fs, order=5):
    b, a = butter_lowpass(fc, fs, order)
    y = signal.lfilter(b, a, data)
    return y


# Frequency modulation
def modulate_fm(t, x, alpha=4):
    return np.cos(2 * np.pi * alpha * integral(t, x))


# Phase modulation
def modulate_pm(t, x, alpha=32):
    return np.cos(2 * np.pi * alpha * x)


# Calculate the spectrogram of a signal after modulation
def spectrogram(t, x, nx=1024, fc=0.02, nperseg=70, noverlap=50, display=False, dim=(256, 256)):
    t = np.linspace(0, t[-1], nx)
    x = signal.resample_poly(x, up=nx, down=len(x))
    x = lowpass_filter(x, fc, 1)
    pm = modulate_pm(t, x)
    fm = modulate_fm(t, x)
    window = signal.get_window("parzen", Nx=nperseg)
    _, _, Spm = signal.spectrogram(pm, nperseg=nperseg, noverlap=noverlap, nfft=nx, window=window)
    _, _, Sfm = signal.spectrogram(fm, nperseg=nperseg, noverlap=noverlap, nfft=nx, window=window)
    Spm = cv2.resize(Spm[:256, :], dim, interpolation=cv2.INTER_LINEAR)
    Sfm = cv2.resize(Sfm[:256, :], dim, interpolation=cv2.INTER_LINEAR)
    Spm = (Spm - np.min(Spm)) / (np.max(Spm) - np.min(Spm))
    Sfm = (Sfm - np.min(Sfm)) / (np.max(Sfm) - np.min(Sfm))

    if display:
        # PM modulation spectrogram
        plt.pcolormesh(Spm[:265, :], shading='gouraud')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.show()

        # FM modulation spectrogram
        plt.pcolormesh(Sfm[:256, :], shading='gouraud')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.show()

    S = np.transpose(np.array([Sfm, Spm], dtype=np.float), (1, 2, 0))
    return S


# Define a simple encoding by copying the columns of a vector
def encode(x, v, dim=(256, 256), display=False):
    height, width = dim
    x = np.tile(np.array(x), (height, 1))
    encoding = cv2.resize(x, dim, cv2.INTER_AREA)
    encoding = (encoding - np.min(encoding)) / (np.max(encoding) - np.min(encoding))
    if display:
        cv2.namedWindow("Encoding", cv2.WINDOW_NORMAL)
        cv2.imshow("Encoding", encoding)
        cv2.waitKey(0)
    return encoding


if __name__ == "__main__":
    t, x = load_json("/home/apostolos/PycharmProjects/Generative-RL/episodes/episode_4.json")
    plt.plot(t, x)
    plt.show()
    # t = t[:int(len(t) / 2)]
    # x = x[:int(len(x) / 2)]
    S = encode(t, x, display=True)
