import soundfile as sf
import matplotlib.pyplot as plt

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def my_windowing(v_signal: np.ndarray, sampling_rate: int, frame_length: int, frame_shift: int):
    frame_length_to_seconds = frame_length / 1000
    frame_length_to_ticks = int(frame_length_to_seconds * sampling_rate)
    frame_shift_to_seconds = frame_shift / 1000
    frame_shift_to_ticks = int(frame_shift_to_seconds * sampling_rate)

    all_windows = sliding_window_view(v_signal, window_shape=frame_length_to_ticks)
    all_windows = all_windows[::frame_shift_to_ticks]
    v_time_frame = [i * frame_shift + 0.5 * frame_length for i in range(0, all_windows.shape[0])]

    return all_windows, np.array(v_time_frame)


def get_index_from_time(time, sampling_rate):
    return int(np.round(time * sampling_rate, 0))


def get_input_split_size(signal_length, sampling_rate, frame_length, frame_shift):
    frame_length_to_seconds = frame_length / 1000
    frame_length_to_ticks = get_index_from_time(frame_length_to_seconds, sampling_rate)
    frame_shift_to_seconds = frame_shift / 1000
    frame_shift_to_ticks = get_index_from_time(frame_shift_to_seconds, sampling_rate)
    # wir gucken, wie viele frame_shifts in das singal passsen, aber müssen aufpassen, dass es gross genug ist für die frame_length_to_ticks
    return (signal_length - frame_length_to_ticks) // frame_shift_to_ticks + 1


def auto_correlate(v_signal_frame: np.ndarray):
    return np.convolve(v_signal_frame, v_signal_frame[::-1], mode='full')


def auto_correlate_positive_values(v_signal_frame: np.ndarray):
    auto_correlated_signal = auto_correlate(v_signal_frame)
    return auto_correlated_signal[len(auto_correlated_signal) // 2:]


# returns in HZ
def get_freq_from_period(index, sampling_rate):
    return 1 / (get_time_from_index(index, sampling_rate))


def get_period_from_freq(frequ, sampling_rate):
    return int(1 / frequ * sampling_rate)


def get_time_from_index(index, sampling_rate):
    return index / sampling_rate


def get_fundamental_frequency(data, samplerate, frame_length, frame_shift):
    """
    This function computes the fundamental frequency of a signal.

    :param data: The input signal
    :param samplerate: The samplerate of the input signal
    :param frame_length: Frame length in milliseconds
    :param frame_shift: Frame shift in milliseconds
    :return: A tuple containing the fundamental frequencies and the time frame around which a frame is centered
    """
    frames, v_time_frame = my_windowing(data, samplerate, frame_length, frame_shift)

    fundamental_frequencies = []
    for frame in frames:
        lower_bound = get_period_from_freq(400, samplerate)
        upper_bound = get_period_from_freq(80, samplerate)

        vocal_sound = np.argmax(auto_correlate_positive_values(frame)[lower_bound:upper_bound]) + lower_bound
        frequency = get_freq_from_period(vocal_sound, samplerate)
        fundamental_frequencies.append(frequency)

    return fundamental_frequencies, v_time_frame


if __name__ == "__main__":
    data, samplerate = sf.read("audio/speech1.wav")
    vocal_sounds, v_time_frame = get_fundamental_frequency(data, samplerate, 32, 16)

    plt.plot(data * 16000)
    plt.plot(v_time_frame * 16, vocal_sounds)  # *16, because ticks/milliesconds = 16 000/1000=16

    plt.show()








