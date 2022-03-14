#!/usr/bin/env python
# coding: utf-8
import numpy as np
import os
import sys
import argparse
import subprocess
import multiprocessing
from multiprocessing import Process
from scipy.io import wavfile as wav

from helper import LoadAudioData
from audio_compression import Audio_Compression

CLIP_MIN = -(2 ** 15)
CLIP_MAX = 2 ** 15 - 1

TEST_FILE = "test_pgd_{}_{}_{}.csv".format(
    os.environ["EPSILON"], os.environ["STEPS"], os.environ["ALPHA"]
)
TEST_FILE = os.path.join("data", TEST_FILE)
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
        "--to",
        type=str,
        required=True,
        default=None,
        help="output format for conversion",
    )
    parser.add_argument(
        "--bit_rate", type=int, required=False, default=None, help="bit rate"
    )
    parser.add_argument(
        "--save", action="store_true", help="whether to save the output audio files"
    )
    args = parser.parse_args()
    return args


def main(data_part):
    args = get_args()
    data = data_part
    num = len(data)
    num_loops = num // args.batch_size
    assert num % args.batch_size == 0

    print("number of batches: {}".format(num_loops))

    for l in range(num_loops):
        print("processing batch: {}".format(l))
        data_sub = data[l * args.batch_size : (l + 1) * args.batch_size]
        audios, lengths = LoadAudioData(data=data_sub, batch_size=args.batch_size)
        audios_new = Audio_Compression(
            audios, lengths, out_format=args.to, bit_rate=args.bit_rate
        )

        if args.save:
            for i in range(audios_new.shape[0]):
                path = os.path.basename(data_sub[i][0]).split(".")[0]
                path = path + "_AC_" + args.to.upper() + ".wav"
                path = os.path.join(STORE_ROOT, path)
                audio_temp = np.clip(audios_new[i][: lengths[i]], CLIP_MIN, CLIP_MAX)
                wav.write(path, 16000, audio_temp.astype(np.int16))
    print("processed {} batches".format(num_loops))


def run_script(parallel=False):
    args = get_args()
    n_jobs = args.n_jobs
    data = np.loadtxt(args.input, delimiter=",", skiprows=1, dtype=str)

    if parallel:
        n_samples = len(data)
        s_per_job = n_samples // n_jobs
        assert n_samples % n_jobs == 0

        procs = []

        for i in range(n_jobs):
            data_part = data[i * s_per_job : (i + 1) * s_per_job]
            p = Process(target=main, args=(data_part,))
            procs.append(p)
            p.start()
        for p in procs:
            p.join()
    else:
        main(data_part=data)


if __name__ == "__main__":
    run_script(parallel=True)
