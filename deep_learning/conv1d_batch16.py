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
from tensorflow.keras.layers import Dense, Conv1D, MaxPool1D, AveragePooling1D, Dropout, Activation, Flatten, Add, Input, Concatenate
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adadelta, Adam, Nadam, RMSprop
from sklearn.preprocessing import StandardScaler, MinMaxScaler

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

print(x_train.shape, y_train.shape) # (3628, 128, 862) (3628,)
print(x_test.shape, y_test.shape)   # (908, 128, 862) (908,)

# 모델 구성

model = Sequential()

def residual_block(x, filters, conv_num=3, activation='relu'):  # ( input, output node, for 문 반복 횟수, activation )
    # Shortcut
    s = Conv1D(filters, 1, padding='same')(x)
    for i in range(conv_num - 1):
        x = Conv1D(filters, 3, padding='same')(x)
        x = Activation(activation)(x)
    x = Conv1D(filters, 3, padding='same')(x)
    x = Add()([x,s])
    x = Activation(activation)(x)
    return MaxPool1D(pool_size=2, strides=1)(x)

def build_model(input_shape, num_classes):
    inputs = Input(shape=input_shape, name='input')

    x = residual_block(inputs, 16, 2)
    x = residual_block(x, 32, 2)
    x = residual_block(x, 64, 3)
    x = residual_block(x, 128, 3)
    x = residual_block(x, 128, 23)

    x = AveragePooling1D(pool_size=3, strides=3)(x)
    x = Flatten()(x)
    x = Dense(256, activation="relu")(x)
    x = Dense(128, activation="relu")(x)

    outputs = Dense(num_classes, activation='softmax', name="output")(x)
    
    return Model(inputs=inputs, outputs=outputs)

model = build_model(x_train.shape[1:], 2)

model.summary()

# 컴파일, 훈련
path = 'E:\\nmb\\nmb_data\\cp\\5s_deeplearning\\conv1d_5_batch16.h5'
op = Nadam(lr=0.0001)
batch_size = 16

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
Epoch 00043: early stopping
57/57 [==============================] - 2s 24ms/step - loss: 0.0553 - acc: 0.9780
loss : 0.05533
acc : 0.97797
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F1.wav 99.9990 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F10.wav 99.9821 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F11.wav 99.9992 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F12.wav 100.0000 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F13.wav 99.9999 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F14.wav 99.9999 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F15.wav 100.0000 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F16.wav 99.9976 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F17.wav 99.9879 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F18.wav 99.9993 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F19.wav 99.9641 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F2.wav 99.9996 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F20.wav 100.0000 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F21.wav 100.0000 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F22.wav 99.6602 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F23.wav 100.0000 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F24.wav 99.9989 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F25.wav 99.9993 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F26.wav 99.9998 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F27.wav 99.9998 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F28.wav 100.0000 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F29.wav 98.8995 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F3.wav 99.9992 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F30.wav 99.9995 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F31.wav 100.0000 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F32.wav 73.8256 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F33.wav 100.0000 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F34.wav 100.0000 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F35.wav 71.8609 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F36.wav 99.9972 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F37.wav 99.9922 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F38.wav 99.9969 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F39.wav 99.9940 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F4.wav 99.9999 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F40.wav 99.9999 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F41.wav 99.9722 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F42.wav 99.9996 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F43.wav 99.8200 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F5.wav 99.9992 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F6.wav 99.9949 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F7.wav 91.9831 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F8.wav 99.9994 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F9.wav 99.9997 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M1.wav 99.9999 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M10.wav 99.9998 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M11.wav 99.9982 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M12.wav 99.9730 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M13.wav 93.3033 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M14.wav 99.9646 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M15.wav 100.0000 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M16.wav 99.9910 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M17.wav 99.9745 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M18.wav 100.0000 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M19.wav 100.0000 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M2.wav 99.9960 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M20.wav 100.0000 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M21.wav 99.9989 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M22.wav 99.7402 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M23.wav 100.0000 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M24.wav 99.9370 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M25.wav 71.5554 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M26.wav 99.9387 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M27.wav 99.9693 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M28.wav 99.7417 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M29.wav 99.7417 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M3.wav 99.9995 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M30.wav 95.9182 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M31.wav 99.9998 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M32.wav 99.9991 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M33.wav 99.9792 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M34.wav 99.9997 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M35.wav 99.9938 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M36.wav 97.1352 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M37.wav 99.9997 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M38.wav 99.9999 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M39.wav 99.9957 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M4.wav 99.9957 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M40.wav 100.0000 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M41.wav 97.6094 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M42.wav 99.9978 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M43.wav 94.9367 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M5.wav 99.9996 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M6.wav 100.0000 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M7.wav 92.4211 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M8.wav 99.9952 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M9.wav 99.9985 의 확률로 남자입니다.
43개 여성 목소리 중 42개 정답
42개 남성 목소리 중 41개 정답
작업 시간 :  0:21:06.349508
"""