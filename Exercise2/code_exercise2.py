import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import get_window
from code_exercise1 import (
    get_index_from_time,
    my_windowing,
    get_fundamental_frequency,
)


def convert_to_samples(milliseconds: int, sampling_freq: int):
    """
    Convert a millisecond duration into the number of samples given the sampling frequency.

    :param milliseconds: duration to be converted to number of samples
    :param sampling_freq: the sampling frequency
    :return: number of samples
    """
    return int(milliseconds * (10 ** (-3)) * sampling_freq)


def compute_stft(
        v_signal: np.ndarray,
        fs: int,
        frame_length: int,
        frame_shift: int,
        v_analysis_window: np.ndarray
) -> [np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the short-time Fourier transform.

    :param v_signal: vector containing the time domain signal
    :param fs: the sampling frequency in Hz
    :param frame_length: frame length in milliseconds
    :param frame_shift: frame shift in milliseconds
    :param v_analysis_window: vector that contains the spectral analysis window
    :return: a tuple containing
        - a matrix which stores the complex short-time spectra in each row
        - a vector which contains the frequency axis in Hz corresponding to the rows of the matrix
        - time steps around which a frame is centered
    """
    v_windows, v_time = my_windowing(v_signal, fs, frame_length, frame_shift)

    v_windows = np.apply_along_axis(lambda x: x * v_analysis_window, 1, v_windows)
    m_stft = np.fft.fft(v_windows)
    v_freq = np.fft.fftfreq(frame_length, d=1 / fs)
    m_stft = m_stft[:, :m_stft.shape[1] // 2]
    v_freq = v_freq[:frame_length // 2]

    return m_stft, v_freq, v_time


def compute_istft(stft: np.ndarray, sampling_rate: int, frame_shift: int, synthesis_window: np.ndarray) -> [np.ndarray]:
    """
    Compute the inverse short-time Fourier transform.

    :param stft: STFT transformed signal
    :param sampling_rate: the sampling rate in Hz
    :param frame_shift: the frame shift used to compute the STFT in milliseconds
    :param synthesis_window: a numpy array containing a synthesis window function (length must match with time domain
    signal segments that were used to compute the STFT)
    :return: a numpy array containing the time domain signal
    """

    # compute inverse rFFT and apply synthesis window
    time_frames = np.fft.irfft(stft)
    num_frames, samples_per_frame = time_frames.shape
    assert samples_per_frame == len(synthesis_window), "Synthesis window must match the number of samples per frame."
    time_frames *= synthesis_window

    # compute output size
    samples_per_shift = convert_to_samples(frame_shift, sampling_rate)
    output_len = samples_per_frame + (num_frames - 1) * samples_per_shift
    time_signal = np.zeros((output_len))

    # reconstruct signal by adding overlapping windowed segments
    for i in range(num_frames):
        time_signal[i * samples_per_shift:i * samples_per_shift + samples_per_frame] += time_frames[i]

    return time_signal


def plot_spectorgram(data, samplerate, frame_length, frame_shift, window="hann", fig=None, ax=None):
    v_analysis_window = get_window(window, get_index_from_time(frame_length / 1000, samplerate))
    m_stft, v_freq, v_time = compute_stft(data, samplerate, frame_length, frame_shift, v_analysis_window)

    if fig is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    im = ax.imshow(
        10 * np.log10(np.maximum(np.square(np.abs(m_stft.T)), 1e-15)),
        cmap='viridis',
        extent=[v_time[0], v_time[-1], v_freq[0], v_freq[-1]],  # noqa
        aspect='auto',
        origin='lower',
    )
    fig.colorbar(im, orientation='vertical', pad=0.2)
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Frequency [Hz]")
    ax.set_title(f"Spectrogram (frame length: {frame_length}ms, frame shift: {frame_shift}ms)")
    return ax


def plot_with_different_parameters(data, samplerate):

    fig, axes = plt.subplots(3, 1, figsize=(10, 10))

    plot_spectorgram(
        data,
        samplerate,
        frame_length=32,
        frame_shift=8,
        window="hann",
        fig=fig,
        ax=axes[0],
    )

    plot_spectorgram(
        data,
        samplerate,
        frame_length=8,
        frame_shift=2,
        window="hann",
        fig=fig,
        ax=axes[1],
    )

    plot_spectorgram(
        data,
        samplerate,
        frame_length=128,
        frame_shift=32,
        window="hann",
        fig=fig,
        ax=axes[2],
    )

    plt.tight_layout()
    plt.show()


def plot_with_fundamental_frequency(data, samplerate, frame_length, frame_shift, window="hann", harmonies=8):
    """
    Plot the spectrogram of the input signal and overlay the fundamental frequency and its harmonics.
    The fundamental frequency is estimated using autocorrelation.

    :param data: Input signal
    :param samplerate: Sampling rate of the input signal
    :param frame_length: Frame length in milliseconds
    :param frame_shift: Frame shift in milliseconds
    :param window: Window function to be used for spectral analysis
    :param harmonies: Number of harmonics to be plotted
    :return: None
    """
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)

    plot_spectorgram(data, samplerate, frame_length, frame_shift, window, fig, ax)
    fundamental_frequencies, v_time_frame = get_fundamental_frequency(data, samplerate, frame_length, frame_shift)
    for i in range(harmonies):
        ax.plot(v_time_frame, np.multiply(fundamental_frequencies, i + 1), color='red', linestyle='--')
    plt.show()



if __name__ == "__main__":
    # data, samplerate = sf.read('audio/phone.wav')
    data, samplerate = sf.read('audio/speech1.wav')

    # plot_spectorgram(data, samplerate, frame_length=32, frame_shift=8)  # 2a
    # plot_with_different_parameters(data, samplerate)  # 2c
    # plot_with_fundamental_frequency(data, samplerate, frame_length=32, frame_shift=8, harmonies=16)  # 2d


