import os
import numpy as np
import argparse
import time
import os
import sys
import csv
from multiprocessing import Process
import scipy.io.wavfile as wav
import tensorflow as tf

sys.path.append("DeepSpeech")
import DeepSpeech

# os.environ['CUDA_VISIBLE_DEVICES'] = ''

from tf_logits import get_logits

from attack.helper import LoadAudioData

# These are the tokens that we're allowed to use.
# The - token is special and corresponds to the epsilon
# value in CTC decoding, and can not occur in the phrase.
toks = " abcdefghijklmnopqrstuvwxyz'-"

MAX_AUDIO_LEN = 104240
MAX_T_LEN = 77


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
        "--output",
        type=str,
        required=False,
        default="ds_out.txt",
        help="location of the prediction file",
    )
    parser.add_argument(
        "--batch_size", type=int, required=False, default=25, help="batch size"
    )
    parser.add_argument(
        "--n_gpus", type=int, required=False, default=4, help="number of gpus to use"
    )
    parser.add_argument(
        "--restore_path",
        type=str,
        required=False,
        default="deepspeech-0.4.1-checkpoint/model.v0.4.1",
        help="path to the DeepSpeech checkpoint (ending in model0.4.1)",
    )
    args = parser.parse_args()
    return args


class Transcribe:
    def __init__(self, sess, batch_size):
        self.args = get_args()
        self.sess = sess
        self.batch_size = batch_size

        self.new_input = tf.placeholder(tf.float32, [self.batch_size, MAX_AUDIO_LEN])
        self.lengths = tf.placeholder(tf.int32, [self.batch_size])

        self.logits = get_logits(self.new_input, self.lengths)

        saver = tf.train.Saver()
        saver.restore(self.sess, self.args.restore_path)

        self.decoded, _ = tf.nn.ctc_beam_search_decoder(
            self.logits, self.lengths, merge_repeated=False, beam_width=500
        )
        self.decoded = tf.convert_to_tensor(
            [
                tf.sparse_tensor_to_dense(sparse_tensor)
                for sparse_tensor in self.decoded
            ],
        )[0]

    def do_transcribe(self, audios, lengths):
        sess = self.sess

        feed_dict = {
            self.new_input: audios,
            self.lengths: np.array(
                [(lengths[i] - 1) // 320 for i in range(len(lengths))]
            ),
        }
        decoded = sess.run(self.decoded, feed_dict=feed_dict)
        return decoded


def main(data_part, device):
    args = get_args()
    data = data_part
    batch_size = args.batch_size

    num = len(data)
    num_loops = num // args.batch_size
    assert num % args.batch_size == 0
    print("number of batches: ", num_loops)

    with tf.device("gpu:0"):
        config = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(config=config) as sess:
            transcribe = Transcribe(sess=sess, batch_size=batch_size)

            for l in range(num_loops):
                print("processing batch: ", l)
                data_sub = data[l * args.batch_size : (l + 1) * args.batch_size]
                audios, lengths = LoadAudioData(
                    data=data_sub, batch_size=args.batch_size
                )
                decoded = transcribe.do_transcribe(audios, lengths)

                for i in range(len(decoded)):
                    prediction = "".join([toks[x] for x in decoded[i]])
                    # print("{},{}".format(data_sub[i, 0], prediction))
                    writer.writerow(
                        {"wav_filename": data[i, 0], "transcript": prediction.strip()}
                    )


if __name__ == "__main__":
    args = get_args()
    data = np.loadtxt(args.input, delimiter=",", skiprows=1, dtype=str)

    field_names = ["wav_filename", "transcript"]
    csvfile = open(args.output, "w")
    writer = csv.DictWriter(csvfile, fieldnames=field_names)
    writer.writeheader()

    main(data, "gpu:0")

    print("model's predictions have been written to ", args.output)
