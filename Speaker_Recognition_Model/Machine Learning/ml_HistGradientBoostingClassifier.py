import numpy as np
import datetime 
import librosa
import sklearn
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier , HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, hamming_loss, hinge_loss, log_loss, mean_squared_error
# from sklearn.utils import all_estimators 
import pickle  
import warnings
warnings.filterwarnings('ignore')
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)

start_now = datetime.datetime.now()

# 데이터 불러오기
x = np.load('C:\\nmb\\nmb_data\\5s_last_0510\\total_data.npy')
x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])

y = np.load('C:\\nmb\\nmb_data\\5s_last_0510\\total_label.npy')
print(x.shape)  # (4536, 110336)
print(y.shape)  # (4536,)

# 전처리
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, test_size=0.2, random_state=42)
print(x_train.shape)    # (3628, 110336)
print(x_test.shape)     # (908, 110336)
print(y_train.shape)    # (3628,)
print(y_test.shape)     # (908,)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 모델 구성
model = HistGradientBoostingClassifier(verbose=1)
# model.fit(x_train, y_train)

# model & weight save
# pickle.dump(model, open('C:\\nmb\\nmb_data\\cp\\5s_last_0510_ml\\HBC_2_minmax.data', 'wb')) # wb : write
# print("== save complete ==")

# model load
model = pickle.load(open('C:\\nmb\\nmb_data\\cp\\5s_last_0510_ml\\HBC_2_minmax.data', 'rb'))  # rb : read
# time >>  

# evaluate
y_pred = model.predict(x_test)
# print(y_pred[:100])
# print(y_pred[100:])

accuracy = accuracy_score(y_test, y_pred)
log_loss = log_loss(y_test, y_pred)

print("log_loss : \t", log_loss)                        # Cross-entropy loss와 유사한 개념
print("accuracy : \t", accuracy)

pred = ['C:\\nmb\\nmb_data\\5s_last_0510\\predict_04_26\\F', 'C:\\nmb\\nmb_data\\5s_last_0510\\predict_04_26\\M']

count_f = 0
count_m = 0

for pred_pathAudio in pred:
    files = librosa.util.find_files(pred_pathAudio, ext=['wav'])
    files = np.asarray(files)
    for file in files:
        name = os.path.basename(file)
        length = len(name)
        name = name[0]

        y, sr = librosa.load(file, sr=22050)
        mels = librosa.feature.melspectrogram(y, sr=sr, hop_length=128, n_fft=512)
        pred_mels = librosa.amplitude_to_db(mels, ref=np.max)
        pred_mels = pred_mels.reshape(1, pred_mels.shape[0] * pred_mels.shape[1])
        pred_mels = scaler.transform(pred_mels)
        y_pred = model.predict(pred_mels)
        # print(y_pred)
        if y_pred == 0:   # 여성이라고 예측
            print(file, '여자입니다.')
            if name == 'F' :
                count_f = count_f + 1
        else:                   # 남성이라고 예측
            print(file, '남자입니다.')
            if name == 'M' :
                count_m = count_m + 1

print("43개 여성 목소리 중 "+str(count_f)+"개 정답")
print("43개 남성 목소리 중 "+str(count_m)+"개 정답")

end_now = datetime.datetime.now()
time = end_now - start_now
print("time >> " , time)    # time >

import winsound as sd
def beepsound():
    fr = 1000    # range : 37 ~ 32767
    du = 500     # 1000 ms ==1second
    sd.Beep(fr, du) # winsound.Beep(frequency, duration)

beepsound()
