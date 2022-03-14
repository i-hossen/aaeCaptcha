# coding: utf-8
import sys
import os
import shutil
import subprocess
import tempfile
import numpy as np
from scipy import signal
from scipy.io import wavfile as wav
import torch
import torch.nn.functional as F

def QT(audio, param=128, bits=16):
    q = param
    if audio.max() < 1:
        audio = audio * 32768

    audio_new = np.round(audio / q) * q
    return audio_new

def MS(audio, param=3):
    win_len = param
    pad_len = (win_len - 1) // 2
    audio = torch.from_numpy(audio)
    audio = F.pad(audio, (pad_len, pad_len), mode="constant", value=0.0)
    roll = audio.unfold(-1, win_len, 1)
    audio_new, _ = torch.median(roll, -1)
    return audio_new.numpy()


def AS(audio, param=3):
    kernel_size = param
    assert kernel_size % 2 == 1

    kernel_weights = np.ones(kernel_size) / kernel_size
    kernel_weights = kernel_weights.reshape((1, -1))
    audio_new = signal.convolve(audio, kernel_weights, mode="same")
    return audio_new


def DS(audio, param, fs=16000):
    ffmpeg_bin = "/usr/bin/ffmpeg"

    tmp_dir = tempfile.mkdtemp()
    audios_new = np.zeros(audio.shape)

    for i in range(audio.shape[0]):
        orig_audio_path = os.path.join(
            tmp_dir, next(tempfile._get_candidate_names()) + ".wav"
        )
        wav.write(orig_audio_path, fs, audio[i].astype(np.int16))

        out_path = os.path.join(tmp_dir, next(tempfile._get_candidate_names()))
        out_path = "{}.{}".format(out_path, ".wav")

        cmd = "{} -y -i {} -ac 1 -ar {} {}".format(
            ffmpeg_bin, orig_audio_path, param, out_path
        )
        args = cmd.split()
        try:
            proc = subprocess.Popen(
                args, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL
            )
        except OSError:
            raise
        else:
            proc.wait()

        target_audio_path = os.path.join(
            tmp_dir, next(tempfile._get_candidate_names()) + ".wav"
        )
        cmd = "{} -y -i {} -ac 1 -ar {} -c:a pcm_s16le {}".format(
            ffmpeg_bin, out_path, fs, target_audio_path
        )
        args = cmd.split()
        try:
            proc = subprocess.Popen(
                args, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL
            )
        except:
            raise
        else:
            proc.wait()

        sr, audio_new = wav.read(target_audio_path, fs)
        audios_new[i][: len(audio_new)] = audio_new

    shutil.rmtree(tmp_dir)
    return audios_new


def butter_lowpass_filter(audio, param, fs=16000, order=4):
    cutoff = param
    nyq_freq = 0.5 * fs
    normal_cutoff = float(cutoff) / nyq_freq
    #     ws = normal_cutoff
    #     wp = 0.5 * ws
    #     print(wp, ws)
    #     N, Wn = signal.buttord(wp, ws, 3.0, 40.0)
    #     b, a = signal.butter(N, Wn, btype='lowpass', analog=False)
    b, a = signal.butter(order, normal_cutoff, btype="lowpass", analog=False)
    audio_new = signal.filtfilt(b, a, audio)
    return audio_new


def butter_bandpass_filter(audio, lowcut, highcut, fs=16000, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    
    b, a = signal.butter(
        order, [low, high], analog=False, btype="bandpass", output="ba"
    )
    audio_new = signal.filtfilt(b, a, audio)
    return audio_new


# def butter_bandpass(lowcut, highcut, samplingrate, order=4):
#     """Source: https://github.com/MTgeophysics/mtpy/blob/develop/mtpy/processing/filter.py"""
#     nyq = 0.5 * samplingrate
#     low = lowcut / nyq
#     high = highcut / nyq

#     if high >= 1.0 and low == 0.0:
#         b = np.array([1.0])
#         a = np.array([1.0])

#     elif high < 0.95 and low > 0.0:
#         wp = [1.05 * low, high - 0.05]
#         ws = [0.95 * low, high + 0.05]

#         order, wn = signal.buttord(wp, ws, 3.0, 40.0)
#         b, a = signal.butter(order, wn, btype="bandpass", analog=False, output="ba")

#     elif high >= 0.95:
#         print("highpass", low, 1.2 * low, 0.8 * low)
#         order, wn = signal.buttord(15 * low, 0.05 * low, gpass=0.0, gstop=10.0)
#         print(order, wn)
#         b, a = signal.butter(order, wn, btype="high", analog=False, output="ba")

#     elif low <= 0.05:
#         print("lowpass", high)
#         order, wn = signal.buttord(high - 0.05, high + 0.05, gpass=0.0, gstop=10.0)
#         b, a = signal.butter(order, wn, analog=False, btype="low", output="ba")

#     return b, a


# def butter_bandpass_filter(audio, lowcut, highcut, fs=16000, order=4):
#     data = audio
#     b, a = butter_bandpass(lowcut, highcut, fs, order=order)
#     y = signal.lfilter(b, a, data)
#     return y
