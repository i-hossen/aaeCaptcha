import os
import tempfile
import shutil
import subprocess
import numpy as np
from scipy.io import wavfile as wav

ffmpeg_bin = "/usr/bin/ffmpeg"

def Audio_Compression(
    audios,
    lengths,
    in_format="wav",
    out_format=None,
    bit_rate=None,
    fs=16000,
):

    in_format = in_format
    out_format = out_format
    bit_rate = bit_rate
    pcm_type = "pcm_s16le"
    fs = fs

    if out_format not in ["mp3", "opus", "aac", "spx"]:
        raise ValueError(
            "invalid output file format. valid formats are mp3, opus, aac and spx"
        )

    if out_format == "mp3":
        codec = "mp3"
    elif out_format == "opus":
        codec = "libopus"
    elif out_format == "spx":
        codec = "libspeex"
    elif out_format == "aac":
        # codec = 'libfdk_aac'
        codec = "aac"

    tmp_dir = tempfile.mkdtemp()
    audios_new = np.zeros(audios.shape)

    for i in range(audios.shape[0]):
        orig_audio_path = os.path.join(
            tmp_dir, next(tempfile._get_candidate_names()) + ".wav"
        )
        wav.write(orig_audio_path, fs, audios[i][: lengths[i]].astype(np.int16))

        out_path = os.path.join(tmp_dir, next(tempfile._get_candidate_names()))
        out_path = "{}.{}".format(out_path, out_format)

        if not bit_rate is None:
            params = "-ac 1 -ar {} -b:a {} -c:a {}".format(fs, bit_rate, codec)
        else:
            params = "-ac 1 -ar {} -c:a {}".format(fs, codec)

        cmd = "{} -y -i {} {} {}".format(ffmpeg_bin, orig_audio_path, params, out_path)
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

        cmd = "{} -y -i {} -ac 1 -ar {} -c:a {} {}".format(
            ffmpeg_bin, out_path, fs, pcm_type, target_audio_path
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

        sr, audio_temp = wav.read(target_audio_path, fs)

        # Workaround: the length of the converted audio may be different than the original
        audios_new[i][: lengths[i]] = audio_temp[: lengths[i]]
        # audios_new[i][: len(audio_temp)] = audio_temp

    shutil.rmtree(tmp_dir)
    return audios_new
