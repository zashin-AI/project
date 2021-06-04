from pydub import AudioSegment, effects
import os,librosa, librosa.display
import sys
sys.path.append('c:/nmb/nada/python_import/')
from volume_handling import volume_updown
import matplotlib.pyplot as plt

path = 'C:\\nmb\\nmb_data\\STT\\PPT\\112_003_0107.wav'
outpath = 'C:\\nmb\\nmb_data\\STT\\PPT_VOL\\'
volume = 20

volume_updown(path=path, volume=volume, outpath=outpath)


y1, sr1 = librosa.load('C:\\nmb\\nmb_data\\STT\\PPT_VOL\\112_003_0107+999db_up.wav')
# y2, sr2 = librosa.load('C:\\nmb\\nmb_data\\STT\\5s_last_normal\\korea_multi_t12_0+15db_up.wav')
# y3, sr3 = librosa.load('C:\\nmb\\nmb_data\\STT\\5s_last_normal\\korea_multi_t12_0-15db_down.wav')

plt.figure(figsize=(20, 20))
plt.subplot(6,1,1)
librosa.display.waveplot(y=y1, sr=sr1)
plt.title('origin')

# plt.subplot(6,1,3)
# librosa.display.waveplot(y=y2, sr=sr2)
# plt.title('origin + 10db')

# plt.subplot(6,1,5)
# librosa.display.waveplot(y=y3, sr=sr3)
# plt.title('origin - 10db')

plt.show()