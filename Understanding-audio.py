import librosa
import numpy as np

# y: 진폭을 시간 순서대로 나열
# sample_rate: 1 초당 샘플의 개수, HZ

y, sample_rate = librosa.load("./example.wav")

print("y shape:", np.shape(y))
print("sampling rate: ", sample_rate)
print("Audio length: ", len(y)/sample_rate)

"""Waveform 으로 plot"""
import matplotlib.pyplot as plt
import librosa.display

plt.figure(figsize=(16,6))
librosa.display.waveshow(y=y, sr = sample_rate, color="blue")
plt.savefig('waveform.png')

"""Fourier Transform"""
"""
stft
window: STFT 에 사용할 window 종류
n_fft: "length of the windowed signal after padding with zeros", 가 frame 을 window_length 만큼 자르고
    n_fft 만큼 zero-padding 된다.
hop_length: 각 window center 간의 거리
Output 은 (1+ n_fft/2, n_frames)
n_frames = 1 + int( len(y) - frame_length / hop_length )
"""
D = np.abs(librosa.stft(y, n_fft = 2048, hop_length=512))
print("D shpae: ",D.shape)

plt.figure(figsize=(16,6))
plt.plot(D)
plt.savefig('FT.png')

"""Spectrogram"""
"""
amplitude_to_db: amplitude spectrogram 을 dB-scaled spectrogram 으로 바꿔준다.
"""
DB = librosa.amplitude_to_db(D, ref=np.max)

plt.figure(figsize=(16,6))
librosa.display.specshow(DB, sr = sample_rate, hop_length=512, x_axis='time', y_axis='log')
plt.colorbar()
plt.savefig("Spectrogram.png")

"""Mel-Spectrogram"""
S = librosa.feature.melspectrogram(y, sr = sample_rate, n_fft=512, hop_length=128)

plt.figure(figsize=(16,6))
S_DB = librosa.amplitude_to_db(S, ref=np.max)
librosa.display.specshow(S_DB, sr = sample_rate, hop_length=512, x_axis='time', y_axis='mel')
plt.savefig('Mel-Spectrogram.png')

"""Mel-Frequency Cepstral Coefficients (MFCCs)"""
"""
Mel-Spectrogram 에서 Cepstral 분석을 통해 추출된 값이다.
Cepstral: 스펙트럼 신호의 로그값에 역푸리에 변환, 배음 분석 (음색, 악기)
"""
import sklearn
mfccs = librosa.feature.mfcc(y, sr=sample_rate)
print("mfccs shape: ",mfccs.shape)

mfccs = sklearn.preprocessing.minmax_scale(mfccs, axis=1)
print("mean: ", mfccs.mean())
print("var: ", mfccs.var())

plt.figure(figsize=(16,6))
librosa.display.specshow(mfccs, sr=sample_rate, x_axis = 'time')
plt.colorbar()
plt.savefig('MFCCs.png')
