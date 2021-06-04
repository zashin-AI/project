from pydub import AudioSegment, effects
import os,librosa, librosa.display
import sys
sys.path.append('c:/nmb/nada/python_import/')
from volume_handling import volume_normal
import matplotlib.pyplot as plt

origin_dir = 'C:\\nmb\\nmb_data\\STT\\5s_last_test\\'
out_dir = 'C:\\nmb\\nmb_data\\STT\\5s_last_normal\\'
volume_normal(origin_dir=origin_dir,out_dir=out_dir)


# # 시각화
y1, sr1 = librosa.load('C:\\nmb\\nmb_data\\STT\\5s_last_test\\korea_multi_t12_0.wav')
y2, sr2 = librosa.load('C:\\nmb\\nmb_data\\STT\\5s_last_normal\\korea_multi_t12_0_volume_normal.wav')
# y3, sr3 = librosa.load('C:\\nmb\\nmb_data\\STT\\STT voice denoise\\test_01_denoise.wav')
# y4, sr4 = librosa.load('C:\\nmb\\nmb_data\\volume\\normal\\test_01_denoise_volume_normal.wav')

plt.figure(figsize=(20,20))
plt.subplot(5,2,1)
librosa.display.waveplot(y=y1, sr=sr1)
plt.title('origin')

plt.subplot(5,2,5)
librosa.display.waveplot(y=y2, sr=sr2)
plt.title('origin_normal')

# plt.subplot(4,2,5)
# librosa.display.waveplot(y=y3, sr=sr3)
# plt.title('denoise')

# plt.subplot(4,2,6)
# librosa.display.waveplot(y=y4, sr=sr4)
# plt.title('denoise_normal')

plt.show()
