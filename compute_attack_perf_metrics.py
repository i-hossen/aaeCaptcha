#!/usr/bin/env python
# coding: utf-8
import sys
import argparse
import numpy as np
from scipy.io import wavfile


def levenshtein(a, b):
    """Calculates the Levenshtein distance between a and b.
    The code was taken from: http://hetland.org/coding/python/levenshtein.py
    """
    n, m = len(a), len(b)
    if n > m:
        # Make sure n <= m, to use O(min(n,m)) space
        a, b = b, a
        n, m = m, n
    current = list(range(n + 1))
    for i in range(1, m + 1):
        previous, current = current, [i] + [0] * n
        for j in range(1, n + 1):
            add, delete = previous[j] + 1, current[j - 1] + 1
            change = previous[j - 1]
            if a[j - 1] != b[i - 1]:
                change = change + 1
            current[j] = min(add, delete, change)
    return current[n]


def evaluate_wer(preds, labels):
    total_dist = 0.0
    total_count = 0.0
    wer_per_sample = np.empty(shape=len(labels))

    empty_preds = 0
    for idx in range(len(labels)):
        audio_filename = labels[idx][0]
        label = labels[idx][-1].strip()
        label = " ".join(label.split())
        pred = preds[idx][-1].strip()
        pred = " ".join(pred.split())
        dist = levenshtein(label.lower().split(), pred.lower().split())
        if pred == "":
            empty_preds += 1
        if pred != "":
            total_dist += dist
            total_count += len(label.split())
        wer_per_sample[idx] = dist / len(label.split())
    print("# empty preds: {}".format(empty_preds))
    wer = total_dist / total_count
    return wer, wer_per_sample


def SNR(normal, adv, bits=16):
    if normal.max() > 1:
        normal = normal / (2 ** (bits - 1))
    normal = normal.flatten()
    if adv.max() > 1:
        adv = adv / (2 ** (bits - 1))
    adv = adv.flatten()

    assert len(normal) == len(adv)

    power_normal = np.sum(normal ** 2)
    distortion = adv - normal
    power_distortion = np.sum(distortion ** 2)
    if power_distortion <= 0.0:
        return np.infty
    snr = 10 * np.log10(power_normal / power_distortion)
    return snr


def compute_snr_and_distortion(labels, preds):
    snr = []
    l1_pert = []

    for i in range(len(labels)):
        _, x = wavfile.read(labels[i, 0])
        _, adv_x = wavfile.read(preds[i, 0])
        snr.append(SNR(x, adv_x))   
        l1_pert.append(np.mean(np.abs(adv_x - x)))
    return np.mean(snr), np.mean(l1_pert)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument(
        "--orig_path",
        type=str,
        required=False,
        default="data/test.csv",
        help="location of the original grouth truth file",
    )
    parser.add_argument(
        "--pred_path",
        type=str,
        required=True,
        default="",
        help="location of the prediction file",
    )
    args = parser.parse_args()

    labels = np.loadtxt(args.orig_path, skiprows=1, delimiter=",", dtype=str)
    preds = np.loadtxt(args.pred_path, skiprows=1, delimiter=",", dtype=str)

    assert len(labels) == len(preds)

    wer, wer_samples = evaluate_wer(preds, labels)
    t = 0.0
    correct = 0.0
    for wer_sample in wer_samples:
        if wer_sample <= t:
            correct += 1
    sroa = correct / len(labels)

    snr, l1_pert = compute_snr_and_distortion(labels, preds)

    print("WER: {:.4f}".format(wer))
    print("SRoA: {:.2f}".format(t, sroa))
    print("SNR: {:.2f}".format(snr))
    print("{}: {:.2f}".format("||\u03B4||\u2081", l1_pert))
