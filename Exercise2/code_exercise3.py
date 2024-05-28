import soundfile as sf
import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
from code_exercise1 import my_windowing, get_index_from_time, get_input_split_size
from scipy.signal import get_window
import scipy as sp

def compute_LP_coefficients(signal, M=12):
    R = np.correlate(signal, signal, mode='full')
    R = R[len(R)//2:]
    c = -R[:M]
    b = R[1:M+1]
    a = sp.linalg.solve_toeplitz(c, b)
    return a
    
    
def plot_frequency_response(a, fs, numPoints, whole=False):
    """
    Plot the frequency response of the LP filter.

    :param a: the LP coefficients
    :param fs: the sampling frequency
    """
    w, h = sp.signal.freqz(1, np.concatenate(([1], a)), numPoints, whole=whole, fs=fs)
    plt.plot(w, abs(h)) # for the size use numPoints, the segment length in sampels
    plt.ylabel('Amplitude ')
    plt.xlabel('Frequency ')
    plt.title("Frequency Response of the LP filter")
    plt.show()
    
def plot_dft_and_filter(dft, a, fs, numPoints, gain=1):
    w, h = sp.signal.freqz(gain, np.concatenate(([1], a)), numPoints, whole=True, fs=fs)
    h = h[:len(h)//2]
    plt.plot(20 * np.log10(np.abs(dft)), label='DFT')
    plt.plot(20 * np.log10(abs(h)), label='Filter')
    plt.title(f"DFT and Filter with gain {gain}")
    plt.ylabel('Amplitude ')
    plt.xlabel('Frequency ')
    plt.legend()
    
    plt.show()

def plot_residual_signal(a, signal):
    e = sp.signal.lfilter(np.concatenate(([1], a)), 1, signal)
    print(np.sum(np.power(e, 2)))
    plt.plot(signal, label='Original Signal')
    plt.plot(e, label='Residual Signal')
    plt.legend()
    plt.title('Residual Signal and Original Signal')
    plt.xlabel('Time [in sampels]')
    plt.ylabel('Amplitude')
    plt.show()

def plot_pre_emphasized_signal_and_normal(signal, pre_emphasized):
    plt.plot(signal, label='Original Signal')
    plt.plot(pre_emphasized, label='Pre-emphasized Signal')
    plt.legend()
    plt.title('Pre-emphasized Signal and Original Signal')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.show()
    
if __name__ == '__main__':
    # No. 1
    data, samplerate = sf.read('Audio/speech1.wav')
    
    # No. 2
    # we need to look at the signal and select one voiced and one unvoiced segment
    # voiced is at v-time of 176, meaning it starts at 176-16=160 and ends at 176+16=192, this is index 10
    # unvoiced is at v-time 544, meaning it starts at 544-16=528 and ends at 544+16=560, this is index 33
    windows, v_time = my_windowing(data, samplerate, 32, 16)
    voiced_segment = windows[13]
    unvoiced_segment = windows[33]
    
    hann_window = get_window('hann', get_index_from_time(32 / 1000, samplerate))
    
    # voiced_segment = voiced_segment * hann_window
    # unvoiced_segment = unvoiced_segment * hann_window
    
    # No. 3
    a_voiced = compute_LP_coefficients(voiced_segment)
    a_unvoiced = compute_LP_coefficients(unvoiced_segment)
    print(a_voiced )
    print(a_unvoiced)
    
    # No. 4
    # plot_frequency_response(a_voiced, samplerate, len(voiced_segment))
    # plot_frequency_response(a_unvoiced, samplerate, len(unvoiced_segment))
    # b) Why do you use np.concatenate(([1], a)) and not only a?
    # -> because the first element of the denominator is 1, so we need to add it to the array (a_0)
    
    # No. 5
    dft_voiced = np.fft.rfft(voiced_segment)
    dft_unvoiced = np.fft.rfft(unvoiced_segment)
    
    # plot_dft_and_filter(dft_voiced, a_voiced, samplerate, len(voiced_segment))
    # plot_dft_and_filter(dft_unvoiced, a_unvoiced, samplerate, len(unvoiced_segment))
    
    # # No. 6
    
    # plot_residual_signal(a_voiced, voiced_segment)
    # plot_residual_signal(a_unvoiced, unvoiced_segment) 
    
    # b) Explain differences in e between the voiced and unvoiced segment
    # -> for voiced: low error, because it is a periodic signal, so prediction is easier; for unvoiced: higher error, because "random" signal is difficult to predict
    # c) Explain why scipy.signal.lfilter(np.concatenate(([1], a)), 1, s) yields the residual signal
    # ->  inverse of filter applied to signal should yield a constant, if it is approximately correct. Should it be 0 or a constant?
    
    # No. 7
    # a) Why are the logarithmic amplitudes of H and S (plots of assignment 5) not on the same level?
    # -> because the gain is different, gain information is lost in the DFT
    
    # b) How can you modify H to achieve a better match? Hint: Experiment with the energy of the residual e
    # -> See no correlation between residual energy and the gain of the filter
    
    # maybe modify the gain?
    # plot_dft_and_filter(dft_voiced, a_voiced, samplerate, len(voiced_segment), gain=2.8312536210370037e-05)
    # plot_dft_and_filter(dft_unvoiced, a_unvoiced, samplerate, len(unvoiced_segment), gain=0.00043334175528043006)
    
    # # square root of the energy of the residual signal
    # plot_dft_and_filter(dft_voiced, a_voiced, samplerate, len(voiced_segment), gain=0.005320952)
    # plot_dft_and_filter(dft_unvoiced, a_unvoiced, samplerate, len(unvoiced_segment), gain=0.0208)
    
    # No. 8
    
    a_voiced_2 = compute_LP_coefficients(voiced_segment, M=2)
    a_voiced_16 = compute_LP_coefficients(voiced_segment, M=16)
    a_voiced_32 = compute_LP_coefficients(voiced_segment, M=32)
    
    # plot_dft_and_filter(dft_voiced, a_voiced_2, samplerate, len(voiced_segment))
    # plot_dft_and_filter(dft_voiced, a_voiced_16, samplerate, len(voiced_segment))
    # plot_dft_and_filter(dft_voiced, a_voiced_32, samplerate, len(voiced_segment))
    
    # Describe differences in H(z) and explain reasons for that
    # -> differences: more detailed representation of the signal, because we can model the signal better with more coefficients (coefficient == blocks in block diagram)
    
    # No. 9
    # b are the numerators
    # a are the denominators
    # alpha is 0.95
    
     
    pre_emphasized_voiced = sp.signal.lfilter([1, 0.95], 1, voiced_segment)
    pre_emphasized_unvoiced = sp.signal.lfilter([1, 0.95], 1, unvoiced_segment)

    # plot_pre_emphasized_signal_and_normal(voiced_segment, pre_emphasized_voiced)
    # plot_pre_emphasized_signal_and_normal(unvoiced_segment, pre_emphasized_unvoiced)
    
    dft_pre_emph = np.fft.rfft(pre_emphasized_voiced)
    
    plot_dft_and_filter(dft_pre_emph, compute_LP_coefficients(pre_emphasized_voiced), samplerate, len(pre_emphasized_voiced))
    
    plot_dft_and_filter(dft_voiced, compute_LP_coefficients(voiced_segment), samplerate, len(voiced_segment))
    # a) Compare the results with and without pre-emphasis.
    # -> pre-emphasized singal makes high voice segments proportionally higher, so it is easier to detect them

    # b) What is the advantage of pre-emphasizing the speech signal?
    # -> makes it easier to detect high voice segments, making "noise" count less

    
    
    
    
    
        
