import numpy as np


def stack_audio_segments(audio_segments, sample_rate):
    stacked = []
    for segment in audio_segments:
        stacked.append(segment)
        stacked.append(np.zeros(int((0.1 + np.random.rand()) * sample_rate)))
    return np.concatenate(stacked)
