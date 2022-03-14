#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import itertools
import json
import csv
import numpy as np
import os
import pandas
import progressbar
import sys
import tables
import tensorflow as tf

sys.path.insert(0, "DeepSpeech")

from attrdict import AttrDict
from collections import namedtuple
from ds_ctcdecoder import ctc_beam_search_decoder_batch, Scorer
from multiprocessing import Pool, cpu_count
from six.moves import zip, range
from util.audio import audiofile_to_input_vector
from util.config import Config, initialize_globals
from util.flags import create_flags, FLAGS
from util.logging import log_error
from util.preprocess import pmap, preprocess
from util.text import Alphabet, ctc_label_dense_to_sparse, wer, levenshtein


def split_data(dataset, batch_size):
    remainder = len(dataset) % batch_size
    if remainder != 0:
        dataset = dataset[:-remainder]

    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size]


def pad_to_dense(jagged):
    maxlen = max(len(r) for r in jagged)
    subshape = jagged[0].shape

    padded = np.zeros((len(jagged), maxlen) + subshape[1:], dtype=jagged[0].dtype)
    for i, row in enumerate(jagged):
        padded[i, : len(row)] = row
    return padded


def process_decode_result(item):
    label, decoding, distance, loss = item
    sample_wer = wer(label, decoding)
    return AttrDict(
        {
            "src": label,
            "res": decoding,
            "loss": loss,
            "distance": distance,
            "wer": sample_wer,
            "levenshtein": levenshtein(label.split(), decoding.split()),
            "label_length": float(len(label.split())),
        }
    )


def calculate_report(labels, decodings, distances, losses):
    r"""
    This routine will calculate a WER report.
    It'll compute the `mean` WER and create ``Sample`` objects of the ``report_count`` top lowest
    loss items from the provided WER results tuple (only items with WER!=0 and ordered by their WER).
    """
    samples = pmap(process_decode_result, zip(labels, decodings, distances, losses))

    total_levenshtein = sum(s.levenshtein for s in samples)
    total_label_length = sum(s.label_length for s in samples)

    # Getting the WER from the accumulated levenshteins and lengths
    samples_wer = total_levenshtein / total_label_length

    # Order the remaining items by their loss (lowest loss on top)
    # samples.sort(key=lambda s: s.loss)

    # Then order by WER (highest WER on top)
    # samples.sort(key=lambda s: s.wer, reverse=True)

    return samples_wer, samples


def evaluate(test_data, inference_graph):
    scorer = Scorer(
        FLAGS.lm_alpha,
        FLAGS.lm_beta,
        FLAGS.lm_binary_path,
        FLAGS.lm_trie_path,
        Config.alphabet,
    )

    def create_windows(features):
        num_strides = len(features) - (Config.n_context * 2)

        # Create a view into the array with overlapping strides of size
        # numcontext (past) + 1 (present) + numcontext (future)
        window_size = 2 * Config.n_context + 1
        features = np.lib.stride_tricks.as_strided(
            features,
            (num_strides, window_size, Config.n_input),
            (features.strides[0], features.strides[0], features.strides[1]),
            writeable=False,
        )

        return features

    # Create overlapping windows over the features
    test_data["features"] = test_data["features"].apply(create_windows)

    with tf.Session(config=Config.session_config) as session:
        inputs, outputs, layers = inference_graph

        # Transpose to batch major for decoder
        transposed = tf.transpose(outputs["outputs"], [1, 0, 2])

        labels_ph = tf.placeholder(
            tf.int32, [FLAGS.test_batch_size, None], name="labels"
        )
        label_lengths_ph = tf.placeholder(
            tf.int32, [FLAGS.test_batch_size], name="label_lengths"
        )

        sparse_labels = tf.cast(
            ctc_label_dense_to_sparse(
                labels_ph, label_lengths_ph, FLAGS.test_batch_size
            ),
            tf.int32,
        )
        loss = tf.nn.ctc_loss(
            labels=sparse_labels,
            inputs=layers["raw_logits"],
            sequence_length=inputs["input_lengths"],
        )

        # Create a saver using variables from the above newly created graph
        mapping = {
            v.op.name: v
            for v in tf.global_variables()
            if not v.op.name.startswith("previous_state_")
        }
        saver = tf.train.Saver(mapping)

        # Restore variables from training checkpoint
        checkpoint = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if not checkpoint:
            log_error(
                "Checkpoint directory ({}) does not contain a valid checkpoint state.".format(
                    FLAGS.checkpoint_dir
                )
            )
            exit(1)

        checkpoint_path = checkpoint.model_checkpoint_path
        saver.restore(session, checkpoint_path)

        logitses = []
        losses = []

        print("Computing acoustic model predictions...")
        batch_count = len(test_data) // FLAGS.test_batch_size
        bar = progressbar.ProgressBar(
            max_value=batch_count, widget=progressbar.AdaptiveETA
        )

        # First pass, compute losses and transposed logits for decoding
        for batch in bar(split_data(test_data, FLAGS.test_batch_size)):
            session.run(outputs["initialize_state"])

            features = pad_to_dense(batch["features"].values)
            features_len = batch["features_len"].values
            labels = pad_to_dense(batch["transcript"].values)
            label_lengths = batch["transcript_len"].values

            logits, loss_ = session.run(
                [transposed, loss],
                feed_dict={
                    inputs["input"]: features,
                    inputs["input_lengths"]: features_len,
                    labels_ph: labels,
                    label_lengths_ph: label_lengths,
                },
            )

            logitses.append(logits)
            losses.extend(loss_)

    ground_truths = []
    predictions = []

    print("Decoding predictions...")
    bar = progressbar.ProgressBar(max_value=batch_count, widget=progressbar.AdaptiveETA)

    # Get number of accessible CPU cores for this process
    try:
        num_processes = cpu_count()
    except:
        num_processes = 1

    # Second pass, decode logits and compute WER and edit distance metrics
    for logits, batch in bar(
        zip(logitses, split_data(test_data, FLAGS.test_batch_size))
    ):
        seq_lengths = batch["features_len"].values.astype(np.int32)
        decoded = ctc_beam_search_decoder_batch(
            logits,
            seq_lengths,
            Config.alphabet,
            FLAGS.beam_width,
            num_processes=num_processes,
            scorer=scorer,
        )

        ground_truths.extend(Config.alphabet.decode(l) for l in batch["transcript"])
        predictions.extend(d[0][1] for d in decoded)

    distances = [levenshtein(a, b) for a, b in zip(ground_truths, predictions)]

    wer, samples = calculate_report(ground_truths, predictions, distances, losses)
    mean_edit_distance = np.mean(distances)
    mean_loss = np.mean(losses)

    # Take only the first report_count items
    report_samples = itertools.islice(samples, FLAGS.report_count)

    # print('Test - WER: %f, CER: %f, loss: %f' %
    #       (wer, mean_edit_distance, mean_loss))
    # print('-' * 80)
    # for sample in report_samples:
    #     print('WER: %f, CER: %f, loss: %f' %
    #           (sample.wer, sample.distance, sample.loss))
    #     print(' - src: "%s"' % sample.src)
    #     print(' - res: "%s"' % sample.res)
    #     print('-' * 80)

    return samples


def main(_):
    initialize_globals()

    if not FLAGS.test_files:
        log_error(
            "You need to specify what files to use for evaluation via "
            "the --test_files flag."
        )
        exit(1)

    # sort examples by length, improves packing of batches and timesteps
    test_data = preprocess(
        FLAGS.test_files.split(","),
        FLAGS.test_batch_size,
        alphabet=Config.alphabet,
        numcep=Config.n_input,
        numcontext=Config.n_context,
        hdf5_cache_path=FLAGS.hdf5_test_set,
    )

    from DeepSpeech import create_inference_graph

    graph = create_inference_graph(batch_size=FLAGS.test_batch_size, n_steps=-1)

    samples = evaluate(test_data, graph)

    field_names = ["wav_filename", "transcript"]
    csvfile = open(FLAGS.pred_file, "w")
    writer = csv.DictWriter(csvfile, fieldnames=field_names)
    writer.writeheader()
    data = np.loadtxt(
        FLAGS.test_files.split(",")[0], delimiter=",", skiprows=1, dtype=str
    )
    for i in range(len(samples)):
        writer.writerow({"wav_filename": data[i, 0], "transcript": samples[i].res})

    if FLAGS.test_output_file:
        # Save decoded tuples as JSON, converting NumPy floats to Python floats
        json.dump(
            samples, open(FLAGS.test_output_file, "w"), default=lambda x: float(x)
        )

    print("model's predictions have been written to ", FLAGS.pred_file)    

if __name__ == "__main__":
    create_flags()
    tf.app.flags.DEFINE_string(
        "hdf5_test_set", "", "path to hdf5 file to cache test set features"
    )
    tf.app.flags.DEFINE_string(
        "test_output_file",
        "",
        "path to a file to save all src/decoded/distance/loss tuples",
    )

    tf.app.flags.DEFINE_string(
        "pred_file", "", "path to a file to save all predictions"
    )

    keys_list = ["lm_binary_path", "lm_trie_path"]
    for keys in keys_list:
        FLAGS.__delattr__(keys)
    tf.app.flags.DEFINE_string("lm_binary_path", "DeepSpeech/data/lm/lm.binary", "")
    tf.app.flags.DEFINE_string("lm_trie_path", "DeepSpeech/data/lm/trie", "")

    tf.app.run(main)
