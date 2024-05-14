import soundfile as sf
import matplotlib.pyplot as plt
import sounddevice as sd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import find_peaks
from code_exercise1 import my_windowing, get_index_from_time, get_input_split_size
from code_exercise2 import convert_to_samples

def load_audio(file_path: str) -> [np.ndarray, int]:
    """
    Load an audio file.

    :param file_path: path to the audio file
    :return: a tuple containing
        - the audio data
        - the sampling rate
    """
    return sf.read(file_path)

if __name__ == "__main__":
    x, fs = load_audio('Audio/female8khz.wav')
    N = 32
    R = 8 #segment shift
    L = N - R 
    m_frames, v_time_frame = my_windowing(x, fs, N, R)
    # Why do we segment the signal prior to analysis instead of processing the whole signal at once?
    # -> more detailed: we want to detect the different vowels and consonants in the signal?
    # Is a segment length of 32 ms appropriate? Why or why not?
    # -> convert_to_samples(32, fs) = 256 sampels (kann mir gerade nichts vorstellen)
