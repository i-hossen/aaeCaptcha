import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
from lingvo import model_imports
from lingvo import model_registry
from lingvo.core import asr_frontend
from lingvo.core import py_utils
import numpy as np
import scipy.io.wavfile as wav
import time
import sys
import os
import csv
from lingvo.core import cluster_factory
from absl import flags
from absl import app

flags.DEFINE_string(
    "input",
    "",
    "the file containing the input audio data paths and transcriptions",
)

flags.DEFINE_string(
    "output",
    "",
    "the file to save model predictions",
)

flags.DEFINE_integer("batch_size", "1", "batch_size to do the testing")
flags.DEFINE_string("checkpoint", "./model/ckpt-00908156", "location of checkpoint")

FLAGS = flags.FLAGS

def _MakeLogMel(audio, sample_rate):
    audio = tf.expand_dims(audio, axis=0)
    static_sample_rate = 16000
    mel_frontend = _CreateAsrFrontend()
    with tf.control_dependencies([tf.assert_equal(sample_rate, static_sample_rate)]):
        log_mel, _ = mel_frontend.FPropDefaultTheta(audio)
    return log_mel


def _CreateAsrFrontend():
    p = asr_frontend.MelFrontend.Params()
    p.sample_rate = 16000.0
    p.frame_size_ms = 25.0
    p.frame_step_ms = 10.0
    p.num_bins = 80
    p.lower_edge_hertz = 125.0
    p.upper_edge_hertz = 7600.0
    p.preemph = 0.97
    p.noise_scale = 0.0
    p.pad_end = False
    # Stack 3 frames and sub-sample by a factor of 3.
    p.left_context = 2
    p.output_stride = 3
    return p.cls(p)


def create_features(input_tf, sample_rate_tf):  # , mask_freq):
    """
    Return:
        A tensor of features with size (batch_size, max_time_steps, 80)
    """

    features_list = []
    # unstact the features with placeholder
    input_unpack = tf.unstack(input_tf, axis=0)
    for i in range(len(input_unpack)):
        features = _MakeLogMel(input_unpack[i], sample_rate_tf)
        features = tf.reshape(features, shape=[-1, 80])
        features = tf.expand_dims(features, dim=0)
        features_list.append(features)
    features_tf = tf.concat(features_list, axis=0)
    features_tf = features_tf
    return features_tf


def create_inputs(model, features, tgt, batch_size):
    tgt_ids, tgt_labels, tgt_paddings = model.GetTask().input_generator.StringsToIds(
        tgt
    )

    # we expect src_inputs to be of shape [batch_size, num_frames, feature_dim, channels]
    src_paddings = tf.zeros(
        [tf.shape(features)[0], tf.shape(features)[1]], dtype=tf.float32
    )
    src_frames = tf.expand_dims(features, dim=-1)

    inputs = py_utils.NestedMap()
    inputs.tgt = py_utils.NestedMap(
        ids=tgt_ids,
        labels=tgt_labels,
        paddings=tgt_paddings,
        weights=1.0 - tgt_paddings,
    )
    inputs.src = py_utils.NestedMap(src_inputs=src_frames, paddings=src_paddings)
    inputs.sample_ids = tf.zeros([batch_size])
    return inputs


def Read_input(data, batch_size):
    audios = []
    lengths = []
    for i in range(batch_size):
        name = data[0]

        sample_rate_np, audio_temp = wav.read(name)

        # read the wav form range from [-32767, 32768] or [-1, 1]
        if max(audio_temp) < 1:
            audio_np = audio_temp * 32768

        else:
            audio_np = audio_temp
        length = len(audio_np)

        audios.append(audio_np)
        lengths.append(length)

    max_length = max(lengths)

    # combine the audios into one array
    audios_np = np.zeros([batch_size, max_length])

    for i in range(batch_size):
        audios_np[i, : lengths[i]] = audios[i]

    audios_np = audios_np.astype(float)
    trans = np.expand_dims(data[-1], 0)

    return audios_np, sample_rate_np, trans


def main(argv):
    data = np.loadtxt(FLAGS.input, skiprows=1, delimiter=",", dtype=str)

    # calculate the number of loops to run the test
    num = len(data)
    batch_size = FLAGS.batch_size
    num_loops = num / batch_size
    assert num % batch_size == 0

    field_names = ["wav_filename", "transcript"]
    csvfile = open(os.path.join(FLAGS.output), "w")
    writer = csv.DictWriter(csvfile, fieldnames=field_names)
    writer.writeheader()

    with tf.device("/gpu:0"):
        tf.set_random_seed(1234)
        tfconf = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(config=tfconf) as sess:
            params = model_registry.GetParams(
                "asr.librispeech.Librispeech960Wpm", "Test"
            )
            params.is_eval = True
            params.cluster.worker.gpus_per_replica = 1
            cluster = cluster_factory.Cluster(params.cluster)
            with cluster, tf.device(cluster.GetPlacer()):
                params.vn.global_vn = False
                params.random_seed = 1234
                params.is_eval = True
                model = params.cls(params)
                task = model.GetTask()

                # saver = tf.train.Saver()
                saver = tf.train.Saver(
                    [
                        x
                        for x in tf.global_variables()
                        if x.name.startswith("librispeech")
                    ]
                )
                saver.restore(sess, FLAGS.checkpoint)

                # define the placeholders
                input_tf = tf.placeholder(tf.float32, shape=[batch_size, None])
                tgt_tf = tf.placeholder(tf.string)
                sample_rate_tf = tf.placeholder(tf.int32)

                # generate the features and inputs
                features = create_features(input_tf, sample_rate_tf)
                shape = tf.shape(features)
                inputs = create_inputs(model, features, tgt_tf, batch_size)

                # loss
                # metrics = task.FPropDefaultTheta(inputs)
                # loss = tf.get_collection("per_loss")[0]

                # prediction
                decoded_outputs = task.Decode(inputs)
                dec_metrics_dict = task.CreateDecoderMetrics()

                correct = 0
                for l in range(num_loops):
                    data_sub = data
                    audios_np, sample_rate, tgt_np = Read_input(data_sub[l], batch_size)
                    feed_dict = {
                        input_tf: audios_np,
                        sample_rate_tf: sample_rate,
                        tgt_tf: tgt_np,
                    }

                    # losses = sess.run(loss, feed_dict)
                    predictions = sess.run(decoded_outputs, feed_dict)

                    # task.PostProcessDecodeOut(predictions, dec_metrics_dict)
                    # wer_value = dec_metrics_dict["wer"].value * 100.0

                    for i in range(batch_size):
                        # print(
                        #     "{},{}".format(
                        #         data_sub[l, 0], predictions["topk_decoded"][i, 0]
                        #     )
                        # )
                        writer.writerow(
                            {
                                "wav_filename": data_sub[l, 0],
                                "transcript": predictions["topk_decoded"][i, 0],
                            }
                        )
    print("model's predictions have been written to ", FLAGS.output)

if __name__ == "__main__":
    app.run(main)
