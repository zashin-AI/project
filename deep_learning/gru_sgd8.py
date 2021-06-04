from itertools import count
import numpy as np
import os
import librosa
import sklearn
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, GRU, Conv1D, MaxPool1D, AveragePooling1D, Dropout, Activation, Flatten, Add, Input, Concatenate
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adadelta, Adam, Nadam, RMSprop, SGD
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.python.keras.layers.wrappers import Bidirectional
start = datetime.now()

x = np.load('E:\\nmb\\nmb_data\\5s_last_0510\\total_data.npy')
y = np.load('E:\\nmb\\nmb_data\\5s_last_0510\\total_label.npy')

print(x.shape, y.shape) # (4536, 128, 862) (4536,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=42
)

x_train_shape_1 = x_train.shape[1]
x_train_shape_2 = x_train.shape[2]

x_test_shape_1 = x_test.shape[1]
x_test_shape_2 = x_test.shape[2]

x_train = x_train.reshape(x_train.shape[0], x_train_shape_1 * x_train_shape_2)
x_test = x_test.reshape(x_test.shape[0], x_test_shape_1 * x_test_shape_2)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0], x_train_shape_1 , x_train_shape_2)
x_test = x_test.reshape(x_test.shape[0], x_test_shape_1 , x_test_shape_2)

# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

# print(x_train.shape, y_train.shape) # (3628, 128, 862) (3628, 2)
# print(x_test.shape, y_test.shape)   # (908, 128, 862) (908, 2)

# 모델 구성

model = Sequential()

def residual_block(x, units, conv_num=3, activation='tanh'):  # ( input, output node, for 문 반복 횟수, activation )
    # Shortcut
    s = GRU(units, return_sequences=True)(x) 
    for i in range(conv_num - 1):
        x = GRU(units, return_sequences=True)(x) # return_sequences=True 이거 사용해서 lstm shape 부분 3차원으로 맞춰줌 -> 자세한 내용 찾아봐야함
        x = Activation(activation)(x)
    x = GRU(units)(x)
    x = Add()([x,s])
    return Activation(activation)(x)

def build_model(input_shape, num_classes):
    inputs = Input(shape=input_shape, name='input')

    x = residual_block(inputs, 1024, 2)
    x = residual_block(x, 512, 2)
    x = residual_block(x, 512, 3)
    x = residual_block(x, 256, 3)
    x = residual_block(x, 256, 3)

    x = Bidirectional(GRU(16))(x)  #  LSTM 레이어 부분에 Bidirectional() 함수 -> many to one 유형
    x = Dense(256, activation="tanh")(x)
    x = Dense(128, activation="tanh")(x)

    outputs = Dense(num_classes, activation='softmax', name="output")(x)
    
    return Model(inputs=inputs, outputs=outputs)

model = build_model(x_train.shape[1:], 2)

model.summary()

# 컴파일, 훈련
path = 'E:\\nmb\\nmb_data\\cp\\5s_deeplearning\\gru_1_sgd8.h5'
op = SGD(lr=1e-3)
batch_size = 8

es = EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True, verbose=1)
lr = ReduceLROnPlateau(monitor='val_loss', vactor=0.4, patience=20, verbose=1)
mc = ModelCheckpoint(path, monitor='val_loss', verbose=1, save_best_only=True)
model.compile(optimizer=op, loss="sparse_categorical_crossentropy", metrics=['acc'])
history = model.fit(x_train, y_train, epochs=5000, batch_size=batch_size, validation_split=0.2, callbacks=[es, lr, mc])

# 평가, 예측
model = load_model(path)
# model.load_weights('C:/nmb/nmb_data/h5/5s/Conv2D_1.h5')
result = model.evaluate(x_test, y_test, batch_size=batch_size)
print("loss : {:.5f}".format(result[0]))
print("acc : {:.5f}".format(result[1]))

############################################ PREDICT ####################################

pred = ['E:\\nmb\\nmb_data\\5s_last_0510\\predict_04_26\\F', 'E:\\nmb\\nmb_data\\5s_last_0510\\predict_04_26\\M']

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
        pred_mels = pred_mels.reshape(1, pred_mels.shape[0]* pred_mels.shape[1])

        pred_mels = scaler.transform(pred_mels)
        pred_mels = pred_mels.reshape(1, 128, 862)

        y_pred = model.predict(pred_mels)
        y_pred_label = np.argmax(y_pred)
        if y_pred_label == 0:   # 여성이라고 예측
            print(file, '{:.4f} 의 확률로 여자입니다.'.format((y_pred[0][0])*100))
            if name == 'F' :
                count_f += 1
        else:                   # 남성이라고 예측
            print(file, '{:.4f} 의 확률로 남자입니다.'.format((y_pred[0][1])*100))
            if name == 'M' :
                count_m += 1
print("43개 여성 목소리 중 "+str(count_f)+"개 정답")
print("42개 남성 목소리 중 "+str(count_m)+"개 정답")

end = datetime.now()
time = end - start
print("작업 시간 : ", time)

import winsound as sd
def beepsound():
    fr = 440    # range : 37 ~ 32767
    du = 1000     # 1000 ms ==1second
    sd.Beep(fr, du) # winsound.Beep(frequency, duration)

beepsound()

"""
Epoch 00117: early stopping
114/114 [==============================] - 12s 80ms/step - loss: 0.3531 - acc: 0.8667
loss : 0.35313
acc : 0.86674
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F1.wav 98.4943 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F10.wav 97.1050 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F11.wav 98.6491 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F12.wav 98.2813 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F13.wav 98.8753 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F14.wav 94.5383 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F15.wav 93.6808 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F16.wav 89.2812 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F17.wav 89.4898 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F18.wav 98.6277 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F19.wav 97.2699 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F2.wav 77.8294 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F20.wav 93.2163 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F21.wav 94.3156 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F22.wav 96.9247 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F23.wav 88.8587 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F24.wav 95.8355 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F25.wav 98.9080 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F26.wav 96.0408 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F27.wav 97.7206 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F28.wav 86.5517 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F29.wav 97.2547 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F3.wav 68.1292 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F30.wav 86.4760 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F31.wav 83.7939 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F32.wav 83.9619 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F33.wav 99.6605 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F34.wav 99.6606 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F35.wav 96.2851 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F36.wav 97.1051 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F37.wav 96.8212 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F38.wav 68.6069 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F39.wav 97.8253 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F4.wav 91.2754 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F40.wav 70.0319 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F41.wav 95.6983 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F42.wav 75.0607 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F43.wav 72.5289 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F5.wav 69.9949 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F6.wav 63.2959 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F7.wav 94.2030 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F8.wav 98.9902 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F9.wav 89.6913 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M1.wav 92.5649 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M10.wav 91.6584 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M11.wav 92.9063 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M12.wav 88.9736 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M13.wav 98.5131 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M14.wav 89.4376 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M15.wav 96.5289 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M16.wav 96.0017 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M17.wav 87.0388 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M18.wav 87.0689 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M19.wav 95.1752 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M2.wav 79.0102 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M20.wav 61.6526 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M21.wav 92.1793 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M22.wav 82.4662 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M23.wav 67.9328 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M24.wav 66.7057 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M25.wav 93.6363 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M26.wav 52.7761 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M27.wav 94.0265 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M28.wav 78.1334 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M29.wav 78.1319 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M3.wav 80.9127 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M30.wav 89.3336 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M31.wav 96.2304 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M32.wav 95.2521 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M33.wav 98.7764 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M34.wav 95.6629 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M35.wav 52.7245 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M36.wav 71.9837 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M37.wav 82.6027 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M38.wav 89.2920 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M39.wav 93.2548 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M4.wav 88.6004 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M40.wav 94.5406 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M41.wav 77.6132 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M42.wav 76.5241 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M43.wav 98.5548 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M5.wav 72.1910 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M6.wav 63.2059 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M7.wav 77.3195 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M8.wav 54.5158 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M9.wav 94.9953 의 확률로 남자입니다.
43개 여성 목소리 중 38개 정답
42개 남성 목소리 중 31개 정답
작업 시간 :  2:28:12.866416
"""