import soundfile as sf
import matplotlib.pyplot as plt
import sounddevice as sd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import find_peaks

# Exercise 1.1
data, samplerate = sf.read('speech1.wav')

# plt.plot(data)
# plt.show()
# 2000-2400 haben wir uns angeguckt
# wir haben mit dem Auge bestimmt: 70 samples zwischen den peaks
# da die sampling rate 16000 ist, haben wir 70/16000 = 0.004375s
# 1/0.004375 = 228.57, weil 1/periode = frequenz
# IN Hz sind das 228.57Hz
# -> female speaker (around 200hz)

#data, samplerate = sf.read('speech2.wav')

# plt.plot(data)
# plt.show()
# beim zweiten haben wir das erste "vokale" angeguckt
# (150 sampels pro periode)
# und sind da auf 106.66HZ gekommen
# -> male speaker (around 100 Hz)
# sd.play(data, samplerate, blocking=True)

#exercise 1.2
def my_windowing( v_signal : np . ndarray , sampling_rate : int , frame_length : int , frame_shift : int ):
    frame_length_to_seconds = frame_length/1000
    frame_length_to_ticks = int(frame_length_to_seconds * sampling_rate)
    frame_shift_to_seconds = frame_shift/1000
    frame_shift_to_ticks = int(frame_shift_to_seconds * sampling_rate)
    
    all_windows = sliding_window_view(v_signal, window_shape=frame_length_to_ticks)
    all_windows = all_windows[::frame_shift_to_ticks]
    v_time_frame = [i*frame_shift + 0.5*frame_length for i in range(0, all_windows.shape[0])]
    
    return all_windows, np.array(v_time_frame)

windows = my_windowing(data, samplerate, 1000, 500)


# for audio in windows:
#     sd.play(audio, samplerate, blocking=True)
def get_index_from_time(time, sampling_rate):
    return np.round(time*sampling_rate, 0)
    
def get_input_split_size(signal_length, sampling_rate, frame_length, frame_shift):
    frame_length_to_seconds = frame_length/1000
    frame_length_to_ticks = get_index_from_time(frame_length_to_seconds, sampling_rate)
    frame_shift_to_seconds = frame_shift/1000
    frame_shift_to_ticks =  get_index_from_time(frame_shift_to_seconds, sampling_rate)
    # wir gucken, wie viele frame_shifts in das singal passsen, aber müssen aufpassen, dass es gross genug ist für die frame_length_to_ticks
    return (signal_length - frame_length_to_ticks)//frame_shift_to_ticks +1

print(get_input_split_size(len(data), samplerate, 32, 16))
# exercise 1.3
frames, v_time_frame = my_windowing(data, samplerate, 32, 16)

def auto_correlate(v_signal_frame: np.ndarray):
    return np.convolve(v_signal_frame, v_signal_frame[::-1], mode='full')

def auto_correlate_positive_values(v_signal_frame: np.ndarray):
    auto_correlated_signal = auto_correlate(v_signal_frame)
    return auto_correlated_signal[len(auto_correlated_signal)//2:]

# returns in HZ
def get_fundamental_frequency(index, sampling_rate):
    return 1/(get_time_from_index(index, sampling_rate))

def get_time_from_frequ(frequ, sampling_rate):
    return int(1/frequ*sampling_rate)

def get_time_from_index(index, sampling_rate):
    return index/sampling_rate


vocal_sounds = []
for frame in frames:
    # print(get_time_from_frequ(400, samplerate))
    # print(get_time_from_frequ(80, samplerate))
    vocal_sound = np.argmax(auto_correlate_positive_values(frame)[get_time_from_frequ(400, samplerate):get_time_from_frequ(80, samplerate)])+get_time_from_frequ(400, samplerate)
    frequency = get_fundamental_frequency(vocal_sound, samplerate)
    vocal_sounds.append(frequency)
    


# # plot both in one plot
# print(f"length of vocal sounds: {len(vocal_sounds)}")
# print(f"length of time frame: {len(v_time_frame)}")
# print(v_time_frame)
#plt.scatter(v_time_frame, vocal_sounds)


plt.plot(data*16000)
plt.plot(v_time_frame*16, vocal_sounds) # *16, because ticks/milliesconds = 16 000/1000=16

plt.show()

    





    
