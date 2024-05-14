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
def compute_power(signal_segment: np.ndarray) -> float:
    """
    Compute the power of a signal segment.

    :param signal_segment: the signal segment
    :return: the power of the signal segment
    """
    return np.sum(signal_segment**2)/len(signal_segment)

def compute_power_for_all_frames(m_frames: np.ndarray) -> np.ndarray:
    """
    Compute the power for all frames.

    :param m_frames: matrix containing the frames
    :return: a vector containing the power for all frames
    """
    return np.apply_along_axis(compute_power, 1, m_frames)

if __name__ == "__main__":
    # 2.2.1
    x, fs = load_audio('Audio/female8khz.wav')
    N = 32
    R = 8 #segment shift
    L = N - R 
    m_frames, v_time_frame = my_windowing(x, fs, N, R)
    # Why do we segment the signal prior to analysis instead of processing the whole signal at once?
    # -> more detailed: we want to detect the different vowels and consonants in the signal?
    # Is a segment length of 32 ms appropriate? Why or why not?
    # -> convert_to_samples(32, fs) = 256 sampels (kann mir gerade nichts vorstellen)
    
    # 2.2.2
    # Compute the power of each frame
    v_power = compute_power_for_all_frames(m_frames)
    # standrad deviation
    v_std = np.sqrt(v_power)
    # plot waveform and v_std
    plt.plot(x)
    plt.plot(v_time_frame, v_std)
    plt.show()
    # Why is it so short compared to the waveform? irgendwas ist komisch... also irgendwie aligned die zeit nicht
