import IPython
from scipy.io import wavfile
import scipy.signal
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf

# wav_loc = "c:/nmb/nmb_data/M2.wav"
# rate, data = wavfile.read(wav_loc)
# data = data / 32768

filepath = 'c:/nmb/nmb_data/M5.wav'
data, rate = librosa.load(filepath)

# from https://stackoverflow.com/questions/33933842/how-to-generate-noise-in-frequency-range-with-numpy

def fftnoise(f):
    f = np.array(f, dtype="complex")
    Np = (len(f) - 1) // 2
    # print('Np : ', Np)
    phases = np.random.rand(Np) * 2 * np.pi
    phases = np.cos(phases) + 1j * np.sin(phases)
    f[1 : Np + 1] *= phases
    f[-1 : -1 - Np : -1] = np.conj(f[1 : Np + 1])
    return np.fft.ifft(f).real

# def fftnoise(f):
#     f = np.fft.fft(f)
#     return np.fft.ifft(f).real

def band_limited_noise(min_freq, max_freq, samples=1024, samplerate=1):
    freqs = np.abs(np.fft.fftfreq(samples, 1 / samplerate))
    f = np.zeros(samples)
    f[np.logical_and(freqs >= min_freq, freqs <= max_freq)] = 1
    return fftnoise(f)

noise_len = 5 # seconds
noise = band_limited_noise(min_freq=500, max_freq = 10000, samples=len(data), samplerate=rate) * 10
noise_clip = noise[:rate*noise_len]
audio_clip_band_limited = data+noise

import time
from datetime import timedelta as td


def _stft(y, n_fft, hop_length, win_length):
    return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)

def _amp_to_db(x):
    return librosa.core.amplitude_to_db(x, ref=1.0, amin=1e-20, top_db=80.0)


def _istft(y, hop_length, win_length):
    return librosa.istft(y, hop_length, win_length)

def _db_to_amp(x,):
    return librosa.core.db_to_amplitude(x, ref=1.0)
    
def removeNoise(
    audio_clip,
    noise_clip,
    n_grad_freq=2,
    n_grad_time=4,
    n_fft=512,
    win_length=512,
    hop_length=128,
    n_std_thresh=1.5,
    prop_decrease=1.0
):

    # STFT over noise
    noise_stft = _stft(noise_clip, n_fft, hop_length, win_length)
    # print('noise_stft : ', noise_stft)
    # print('noise_stft : ', noise_stft.shape) # noise_stft :  (257, 690)
    noise_stft_db = _amp_to_db(np.abs(noise_stft))  # convert to dB
    # print('noise_stft_db : ', noise_stft_db)
    # print('noise_stft_db : ', noise_stft_db.shape) # noise_stft_db :  (257, 690)
    # Calculate statistics over noise
    mean_freq_noise = np.mean(noise_stft_db, axis=1)
    # print('mean_freq_noise : ', mean_freq_noise)
    # print('mean_freq_noise : ', mean_freq_noise.shape) # mean_freq_noise :  (257,)
    std_freq_noise = np.std(noise_stft_db, axis=1)
    # print('std_freq_noise : ', std_freq_noise)
    # print('std_freq_noise : ', std_freq_noise.shape) # std_freq_noise :  (257,)
    noise_thresh = mean_freq_noise + std_freq_noise * n_std_thresh
    # print('noise_thresh : ', noise_thresh)
    # print('noise_thresh : ', noise_thresh.shape) # noise_thresh :  (257,)
    # STFT over signal
    sig_stft = _stft(audio_clip, n_fft, hop_length, win_length)
    # print('sig_stft : ', sig_stft)
    # print('sig_stft : ', sig_stft.shape) # sig_stft :  (257, 862)
    sig_stft_db = _amp_to_db(np.abs(sig_stft))
    # print('sig_stft_db : ', sig_stft_db)
    # print('sig_stft_db : ', sig_stft_db.shape) # sig_stft_db :  (257, 862)
    # Calculate value to mask dB to
    mask_gain_dB = np.min(_amp_to_db(np.abs(sig_stft)))
    # print('mask_gain_db : ', mask_gain_dB) # mask_gina_db :  -31.554999244459893
    # print('mask_gain_db : ', mask_gain_dB.shape) # ()
    # Create a smoothing filter for the mask in time and frequency
    smoothing_filter = np.outer(
        np.concatenate(
            [
                np.linspace(0, 1, n_grad_freq + 1, endpoint=False),
                np.linspace(1, 0, n_grad_freq + 2),
            ]
        )[1:-1],
        np.concatenate(
            [
                np.linspace(0, 1, n_grad_time + 1, endpoint=False),
                np.linspace(1, 0, n_grad_time + 2),
            ]
        )[1:-1],
    )
    smoothing_filter = smoothing_filter / np.sum(smoothing_filter)
    # print('smoothing_filter : ', smoothing_filter)
    # print('smoothing_filter : ', smoothing_filter.shape) # smoothing_filter :  (5, 9)
    # calculate the threshold for each frequency/time bin
    db_thresh = np.repeat(
        np.reshape(noise_thresh, [1, len(mean_freq_noise)]),
        np.shape(sig_stft_db)[1],
        axis=0,
    ).T
    # print('db_thresh : ', db_thresh)
    # print('db_thresh : ', db_thresh.shape) # db_thresh :  (257, 862)
    # mask if the signal is above the threshold
    sig_mask = sig_stft_db < db_thresh
    # print('sig_mask : ', sig_mask)
    # print('sig_mask : ', sig_mask.shape) # sig_mask :  (257, 862)
    # convolve the mask with a smoothing filter
    sig_mask = scipy.signal.fftconvolve(sig_mask, smoothing_filter, mode="same")
    # print('sig_mask : ', sig_mask)
    # print('sig_mask : ', sig_mask.shape) # sig_mask :  (257, 862)
    sig_mask = sig_mask * prop_decrease
    # print('sig_mask : ', sig_mask)
    # print('sig_mask : ', sig_mask.shape) # sig_mask :  (257, 862)
    # mask the signal
    sig_stft_db_masked = (
        sig_stft_db * (1 - sig_mask)
        + np.ones(np.shape(mask_gain_dB)) * mask_gain_dB * sig_mask
    )  # mask real
    # print('sig_stft_db_masked : ', sig_stft_db_masked)
    # print('sig_stft_db_masked : ', sig_stft_db_masked.shape) # sig_stft_db_masked :  (257, 862)
    sig_imag_masked = np.imag(sig_stft) * (1 - sig_mask)
    # print('sig_imag_masked : ', sig_imag_masked)
    # print('sig_imag_masked : ', sig_imag_masked.shape) # sig_imag_masked :  (257, 862)
    sig_stft_amp = (_db_to_amp(sig_stft_db_masked) * np.sign(sig_stft)) + (
        1j * sig_imag_masked
    )
    # print('sig_stft_amp : ', sig_stft_amp)
    # print('sig_stft_amp : ', sig_stft_amp.shape) # sig_stft_amp :  (257, 862)
    # recover the signal
    recovered_signal = _istft(sig_stft_amp, hop_length, win_length)
    # print('recovered_signal : ', recovered_signal)
    # print('recovered_signal : ', recovered_signal.shape) # recovered_signal :  (110208,)
    recovered_spec = _amp_to_db(
        np.abs(_stft(recovered_signal, n_fft, hop_length, win_length))
    )
    # print('recovered_spec : ', recovered_spec)
    # print('recovered_spec : ', recovered_spec.shape) # recovered_spec :  (257, 862)
    return recovered_signal

output = removeNoise(audio_clip=audio_clip_band_limited, noise_clip=noise_clip)
# print('output : ', output)
# print('output : ', output.shape) # output :  (110208,)

if __name__ == '__main__':
    sf.write(
        'c:/nmb/nmb_data/output2.wav', output, samplerate=rate
    )
    sf.write(
        'c:/nmb/nmb_data/original.wav', data, samplerate=rate
    )
    sf.write(
        'c:/nmb/nmb_Data/noise.wav', audio_clip_band_limited, samplerate=rate
    )

    fig = plt.figure(figsize = (16, 6))
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)

    librosa.display.waveplot(data, ax = ax1)
    librosa.display.waveplot(output, ax = ax2)
    librosa.display.waveplot(audio_clip_band_limited, ax = ax3)
    librosa.display.waveplot(noise, ax = ax4)
    ax1.set(title = 'original')
    ax2.set(title = 'denoise')
    ax3.set(title = 'data with noise')
    ax4.set(title = 'noise')

    fig.tight_layout()
    plt.show()