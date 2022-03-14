# coding: utf-8

import numpy as np
import argparse
from tqdm import tqdm
import time
import tensorflow as tf
import scipy.io.wavfile as wav
import os
import sys
from multiprocessing import Process

sys.path.append("DeepSpeech")
import random
import librosa

###########################################################################
# This section of code is credited to:
## Copyright (C) 2017, Nicholas Carlini <nicholas@carlini.com>.

# Okay, so this is ugly. We don't want DeepSpeech to crash.
# So we're just going to monkeypatch TF and make some things a no-op.
# Sue me.
tf.load_op_library = lambda x: x
generation_tmp = os.path.exists
os.path.exists = lambda x: True


class Wrapper:
    def __init__(self, d):
        self.d = d

    def __getattr__(self, x):
        return self.d[x]


class HereBeDragons:
    d = {}
    FLAGS = Wrapper(d)

    def __getattr__(self, x):
        return self.do_define

    def do_define(self, k, v, *x):
        self.d[k] = v


tf.app.flags = HereBeDragons()
import DeepSpeech

os.path.exists = generation_tmp

# More monkey-patching, to stop the training coordinator setup
DeepSpeech.TrainingCoordinator.__init__ = lambda x: None
DeepSpeech.TrainingCoordinator.start = lambda x: None

# from util.text import ctc_label_dense_to_sparse
from tensorflow.python.keras.backend import ctc_label_dense_to_sparse
from tf_logits import compute_mfcc, get_logits


# These are the tokens that we're allowed to use.
# The - token is special and corresponds to the epsilon
# value in CTC decoding, and can not occur in the phrase.
toks = " abcdefghijklmnopqrstuvwxyz'-"

###########################################################################

from attack.audio_preprocessing import QT

MAX_AUDIO_LEN = 104240
MAX_T_LEN = 77
CLIP_MIN = -(2 ** 15)
CLIP_MAX = 2 ** 15 - 1
STORE_ROOT = "data/audio_captchas"


def LoadAudioData(data, batch_size):

    audios = np.zeros([batch_size, MAX_AUDIO_LEN], dtype=np.float32)
    trans = np.zeros([batch_size, MAX_T_LEN], dtype=np.int32)
    lengths = np.zeros(batch_size, dtype=np.int32)
    tran_lengths = np.zeros(batch_size, dtype=np.int32)

    for i in range(batch_size):
        _, audio_temp = wav.read(data[i, 0])

        if max(audio_temp) < 1:
            audio_np = audio_temp * 32768
        else:
            audio_np = audio_temp

        tran = data[i, -1]
        tran = [toks.index(x) for x in tran]

        length = len(audio_np)
        t_length = len(tran)

        audios[i, :length] = audio_np.astype(float)
        trans[i, :t_length] = tran

        lengths[i] = length
        tran_lengths[i] = t_length

    return (
        audios,
        lengths,
        trans,
        tran_lengths,
    )


class aaeCaptcha:
    def __init__(self, sess, batch_size, epsilon, restore_path, seed):
        self.sess = sess
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.seed = seed
        self.restore_path = restore_path
        self.max_len = MAX_AUDIO_LEN
        self.max_t_len = MAX_T_LEN

        self.delta = tf.Variable(
            np.zeros((self.batch_size, self.max_len), dtype=np.float32),
            name="aae_delta",
        )
        self.original = tf.Variable(
            np.zeros((self.batch_size, self.max_len), dtype=np.float32),
            name="aae_orig",
        )
        self.mask = tf.Variable(
            np.zeros((batch_size, self.max_len), dtype=np.float32), name="aae_mask"
        )
        self.lengths = lengths = tf.Variable(
            np.zeros(self.batch_size, dtype=np.int32), name="aae_lengths"
        )
        self.phrases = tf.placeholder(
            tf.int32, shape=[self.batch_size, self.max_t_len], name="aae_phrases"
        )
        self.phrase_lengths = tf.placeholder(
            tf.int32, shape=[self.batch_size], name="aae_phrase_lengths"
        )
        self.learning_rate = tf.placeholder(tf.float32, name="aae_lr")

        self.apply_delta = tf.clip_by_value(self.delta, -epsilon, epsilon) * self.mask
        self.pass_in = tf.clip_by_value(
            self.original + self.apply_delta, CLIP_MIN, CLIP_MAX
        )
        self.logits = get_logits(self.pass_in, self.lengths)

        saver = tf.train.Saver(
            [x for x in tf.global_variables() if "aae" not in x.name]
        )
        saver.restore(self.sess, self.restore_path)

        target = ctc_label_dense_to_sparse(self.phrases, self.phrase_lengths)

        self.ctcloss = tf.nn.ctc_loss(
            labels=tf.cast(target, tf.int32),
            inputs=self.logits,
            sequence_length=self.lengths,
        )
        self.loss = self.ctcloss

        start_vars = set(x.name for x in tf.global_variables())

        # self.grad = tf.gradients(self.loss, [self.delta])[0]
        # self.optim_step = tf.assign(self.delta, self.delta + self.learning_rate * tf.sign(self.grad))
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        grad, var = optimizer.compute_gradients(-self.loss, [self.delta])[0]
        self.train = optimizer.apply_gradients([(tf.sign(grad), var)])

        end_vars = tf.global_variables()
        new_vars = [x for x in end_vars if x.name not in start_vars]
        self.sess.run(tf.variables_initializer(new_vars + [self.delta]))

        # self.decoded, _ = tf.nn.ctc_beam_search_decoder(
        #     self.logits, self.lengths, merge_repeated=False, beam_width=100
        # )
        # self.decoded = tf.convert_to_tensor(
        #                 [tf.sparse.to_dense(sparse_tensor) for sparse_tensor in self.decoded], name='output_node'
        # )

    def pgd(
        self,
        audios,
        lengths,
        phrases,
        phrase_lens,
        eps=350.0,
        alpha=40.0,
        steps=50,
        rand_init=False,
        BPDA=False,
    ):
        sess = self.sess
        sess.run(tf.variables_initializer([self.delta]))
        sess.run(self.original.assign(audios))
        sess.run(
            self.mask.assign(
                np.array(
                    [[1 if i < l else 0 for i in range(self.max_len)] for l in lengths]
                )
            )
        )
        sess.run(
            self.lengths.assign(
                [(audios.shape[1] - 1) // 320 for i in range(self.batch_size)]
            )
        )
        # sess.run(self.phrase_lengths.assign(phrase_lens))

        np.random.seed(seed=self.seed)
        tf.set_random_seed(self.seed)

        if rand_init:
            delta_init = np.random.uniform(-eps, eps, size=audios.shape)
            sess.run(self.delta.assign(delta_init))
        else:
            self.delta.assign(
                np.zeros((self.batch_size, self.max_len), dtype=np.float32)
            )

        feed_dict = {
            self.phrases: phrases,
            self.phrase_lengths: phrase_lens,
            self.learning_rate: alpha,
        }

        # PGD
        for i in tqdm(range(1, steps + 1)):

            if BPDA:
                new_input = sess.run(self.pass_in)
                QTed = QT(new_input, param=1024)
                feed_dict[self.pass_in] = QTed

            # sess.run(self.optim_step, feed_dict=feed_dict)
            sess.run(self.train, feed_dict=feed_dict)

            if i % 10 == 0:
                loss = sess.run(self.loss, feed_dict=feed_dict)
                print("CTC loss: ", loss.mean())

        adv = sess.run(self.pass_in)
        return adv


def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument(
        "--input",
        type=str,
        required=False,
        default="data/test.csv",
        help="location of input data",
    )
    parser.add_argument(
        "--batch_size", type=int, required=False, default=25, help="batch size"
    )
    parser.add_argument("--seed", type=int, required=False, default=5, help="seed")
    parser.add_argument(
        "--epsilon",
        type=float,
        required=False,
        default=350.0,
        help="perturbation constraint",
    )
    parser.add_argument(
        "--alpha", type=float, required=False, default=40.0, help="PGD learning rate"
    )
    parser.add_argument(
        "--steps",
        type=int,
        required=False,
        default=50,
        help="maximum number of PGD iterations",
    )
    parser.add_argument(
        "--bpda",
        action="store_const",
        const=True,
        required=False,
        help="generate audio preprocessing attack-resistant adversarial examples",
    )
    parser.add_argument(
        "--restore_path",
        type=str,
        required=False,
        default="deepspeech-0.4.1-checkpoint/model.v0.4.1",
        help="path to the DeepSpeech checkpoint (ending in model0.4.1)",
    )
    parser.add_argument(
        "--n_gpus", type=int, required=False, default=4, help="number of gpus to use"
    )
    args = parser.parse_args()
    return args


def main(device, data_part):
    args = get_args()
    data = data_part

    num = len(data)
    num_loops = num // args.batch_size
    assert num % args.batch_size == 0
    print("number of batches: ", num_loops)

    with tf.device(device):
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            aae_captha = aaeCaptcha(
                sess=sess,
                batch_size=args.batch_size,
                epsilon=args.epsilon,
                restore_path=args.restore_path,
                seed=args.seed,
            )

            start = time.time()
            l1_pertub = []

            for l in range(num_loops):
                print("processing batch: ", l)
                data_sub = data[l * args.batch_size : (l + 1) * args.batch_size]
                audios, lengths, trans, tran_lengths = LoadAudioData(
                    data=data_sub, batch_size=args.batch_size
                )
                adv = aae_captha.pgd(
                    audios=audios,
                    lengths=lengths,
                    phrases=trans,
                    phrase_lens=tran_lengths,
                    eps=args.epsilon,
                    alpha=args.alpha,
                    steps=args.steps,
                    rand_init=True,
                    BPDA=args.bpda,
                )

                for i in range(adv.shape[0]):
                    path = os.path.basename(data_sub[i][0]).split(".")[0]
                    path = "{}_pgd_{}_{}_{}".format(
                        path, int(args.epsilon), int(args.steps), int(args.alpha)
                    )

                    if args.bpda:
                        path = path + "_BPDA.wav"
                    else:
                        path = path + ".wav"

                    path = os.path.join(STORE_ROOT, path)
                    wav.write(path, 16000, adv[i][: lengths[i]].astype(np.int16))
                    l1_pertub.append(
                        np.mean(np.abs(adv[i][: lengths[i]] - audios[i][: lengths[i]]))
                    )
                print(
                    "{} for current batch: {}".format(
                        "||\u03B4||\u2081",
                        np.mean(
                            l1_pertub[l * args.batch_size : (l + 1) * args.batch_size]
                        ),
                    )
                )

            print("final {}: {}".format("||\u03B4||\u2081", np.mean(l1_pertub)))
            end = time.time()
            print("processed {} batches in {:.2f} secs".format(num_loops, end - start))


def run_script(use_mult_gpu=False):
    args = get_args()
    data = np.loadtxt(args.input, delimiter=",", skiprows=1, dtype=str)

    if use_mult_gpu:
        n_gpus = args.n_gpus
        n_samples = len(data)
        s_per_gpu = n_samples // n_gpus
        assert s_per_gpu % args.batch_size == 0

        procs = []

        for i in range(n_gpus):
            device = "gpu:{}".format(i)
            data_part = data[i * s_per_gpu : (i + 1) * s_per_gpu]
            p = Process(target=main, args=(device, data_part))
            procs.append(p)
            p.start()
        for p in procs:
            p.join()
    else:
        main("gpu:0", data)


if __name__ == "__main__":
    run_script(use_mult_gpu=True)
