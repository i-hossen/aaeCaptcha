#!/usr/bin/env python
# coding: utf-8
import numpy as np
import tempfile
import os
import sys
import argparse
import multiprocessing
from multiprocessing import Process
from scipy.io import wavfile as wav

from helper import LoadAudioData
from audio_preprocessing import (
    QT,
    AS,
    MS,
    DS,
    butter_lowpass_filter,
    butter_bandpass_filter,
)

CLIP_MIN = -(2 ** 15)
CLIP_MAX = 2 ** 15 - 1

TEST_FILE = "test_pgd_{}_{}_{}.csv".format(
    os.environ["EPSILON"], os.environ["STEPS"], os.environ["ALPHA"]
)
TEST_FILE = os.path.join('data', TEST_FILE)
STORE_ROOT = "data/audio_captchas"

def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument(
        "--input",
        type=str,
        required=False,
        default=TEST_FILE,
        help="location of input data",
    )
    parser.add_argument(
        "--batch_size", type=int, required=False, default=25, help="batch size"
    )
    parser.add_argument(
        "--n_jobs", type=int, required=False, default=20, help="number of jobs"
    )
    parser.add_argument(
        "--attack_type",
        type=str,
        required=True,
        default=None,
        help="attack type: [QT, MS, AS, DS, LPF, BPF]",
    )
    parser.add_argument(
        "--param",
        type=int,
        required=True,
        default=None,
        help="parameter for the specified attack",
    )
    parser.add_argument(
        "--param2",
        type=int,
        required=False,
        default=None,
        help="higher cutoff frequency for bandpass filter",
    )
    parser.add_argument("--save", action="store_true", help="whether to save the preprocessed audio samples")
    args = parser.parse_args()
    return args


def main(data_part, attack_type, attack_method, attack_param):
    args = get_args()
    data = data_part
    num = len(data)
    num_loops = num // args.batch_size
    assert num % args.batch_size == 0

    print("number of batches: ", num_loops)

    for l in range(num_loops):
        print("processing batch: ", l)
        data_sub = data[l * args.batch_size : (l + 1) * args.batch_size]
        audios, lengths = LoadAudioData(data=data_sub, batch_size=args.batch_size)
        if attack_type == "BPF":
            audios_new = attack_method(audio=audios, **attack_param)
        else:
            audios_new = attack_method(audios, attack_param)
        if args.save:
            for i in range(audios_new.shape[0]):
                if attack_type == "BPF":
                    path = os.path.basename(data_sub[i][0]).split(".")[0]
                    path = "{}_{}_{}_{}.{}".format(
                        path,
                        attack_type,
                        attack_param["lowcut"],
                        attack_param["highcut"],
                        "wav",
                    )
                    path = os.path.join(STORE_ROOT, path)
                    wav.write(path, 16000, audios_new[i][: lengths[i]].astype(np.int16))
                else:
                    path = os.path.basename(data_sub[i][0]).split(".")[0]
                    path = "{}_{}_{}.{}".format(path, attack_type, attack_param, "wav")
                    path = os.path.join(STORE_ROOT, path)
                    audio_temp = np.clip(
                        audios_new[i][: lengths[i]], CLIP_MIN, CLIP_MAX
                    )
                    wav.write(path, 16000, audio_temp.astype(np.int16))
    print("processed ", num_loops, " loops")


def run_script(attack_type, attack_method, attack_param, parallel=False):
    args = get_args()
    data = np.loadtxt(args.input, delimiter=",", skiprows=1, dtype=str)

    if parallel:
        n_jobs = args.n_jobs
        n_samples = len(data)
        s_per_job = n_samples // n_jobs
        assert n_samples % n_jobs == 0

        procs = []

        for i in range(n_jobs):
            data_part = data[i * s_per_job : (i + 1) * s_per_job]
            p = Process(
                target=main, args=(data_part, attack_type, attack_method, attack_param)
            )
            procs.append(p)
            p.start()
        for p in procs:
            p.join()
    else:
        main(data, attack_type, attack_method, attack_param)


if __name__ == "__main__":
    args = get_args()
    valid_attack_types = ["QT", "AS", "MS", "DS", "LPF", "BPF"]

    if args.attack_type not in valid_attack_types:
        raise ValueError("invalid attack type")

    if args.attack_type == "QT":
        attack = QT
    elif args.attack_type == "AS":
        attack = AS
    elif args.attack_type == "MS":
        attack = MS
    elif args.attack_type == "DS":
        attack = DS
    elif args.attack_type == "LPF":
        attack = butter_lowpass_filter
    elif args.attack_type == "BPF":
        attack = butter_bandpass_filter
        lowcut = args.param
        highcut = args.param2
        if args.param2 == None:
            raise ValueError("--param2 (highcut frequency) is required.")

    if args.attack_type == "BPF":
        run_script(
            attack_type=args.attack_type,
            attack_method=attack,
            attack_param={"lowcut": lowcut, "highcut": highcut},
            parallel=False,
        )
    else:
        attack_param = args.param
        run_script(
            attack_type=args.attack_type,
            attack_method=attack,
            attack_param=attack_param,
            parallel=False,
        )
