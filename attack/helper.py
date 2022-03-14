import numpy as np
from scipy.io import wavfile as wav

MAX_AUDIO_LEN = 104240
MAX_T_LEN = 77

def LoadAudioData(data, batch_size):
    audios = np.zeros([batch_size, MAX_AUDIO_LEN], dtype=np.float32)
    lengths = np.zeros(batch_size, dtype=np.int32)

    for i in range(batch_size):
        _, audio_temp = wav.read(data[i, 0])

        if max(audio_temp) < 1:
            audio_np = audio_temp * 32768
        else:
            audio_np = audio_temp

        length = len(audio_np)
        audios[i, :length] = audio_np.astype(float)
        lengths[i] = length

    return audios, lengths