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
from tensorflow.keras.layers import Dense, GRU, Conv1D, MaxPool1D, AveragePooling1D, Dropout, Activation, Flatten, Add, Input, Concatenate, LSTM, SimpleRNN
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

# 모델 구성

model = Sequential()

def residual_block(x, units, conv_num=3, activation='tanh'):  # ( input, output node, for 문 반복 횟수, activation )
    # Shortcut
    s = SimpleRNN(units, return_sequences=True)(x) 
    for i in range(conv_num - 1):
        x = SimpleRNN(units, return_sequences=True)(x) # return_sequences=True 이거 사용해서 lstm shape 부분 3차원으로 맞춰줌 -> 자세한 내용 찾아봐야함
        x = Activation(activation)(x)
    x = SimpleRNN(units)(x)
    x = Add()([x,s])
    return Activation(activation)(x)

def build_model(input_shape, num_classes):
    inputs = Input(shape=input_shape, name='input')

    x = residual_block(inputs, 16, 2)
    x = residual_block(x, 32, 2)
    x = residual_block(x, 64, 3)
    x = residual_block(x, 128, 3)
    x = residual_block(x, 128, 3)

    x = Bidirectional(SimpleRNN(16))(x)  #  LSTM 레이어 부분에 Bidirectional() 함수 -> many to one 유형
    x = Dense(256, activation="tanh")(x)
    x = Dense(128, activation="tanh")(x)

    outputs = Dense(num_classes, activation='softmax', name="output")(x)
    
    return Model(inputs=inputs, outputs=outputs)

model = build_model(x_train.shape[1:], 2)

model.summary()

# 컴파일, 훈련
path = 'E:\\nmb\\nmb_data\\cp\\5s_deeplearning\\rnn_1_batch16.h5'
op = RMSprop(lr=1e-3)
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
Epoch 00057: early stopping
57/57 [==============================] - 21s 341ms/step - loss: 0.3460 - acc: 0.8645
loss : 0.34597
acc : 0.86454
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F1.wav 92.9962 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F10.wav 97.2704 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F11.wav 97.0618 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F12.wav 97.9353 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F13.wav 96.0746 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F14.wav 95.7867 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F15.wav 96.0350 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F16.wav 97.2522 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F17.wav 91.6835 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F18.wav 96.9945 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F19.wav 97.5160 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F2.wav 86.6277 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F20.wav 95.6271 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F21.wav 86.0325 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F22.wav 96.8570 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F23.wav 90.5170 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F24.wav 96.6625 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F25.wav 96.3950 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F26.wav 96.7419 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F27.wav 97.3686 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F28.wav 88.3399 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F29.wav 95.2202 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F3.wav 91.5327 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F30.wav 95.3587 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F31.wav 90.6111 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F32.wav 53.2643 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F33.wav 95.4215 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F34.wav 95.4215 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F35.wav 95.2388 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F36.wav 97.6715 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F37.wav 98.0153 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F38.wav 80.0743 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F39.wav 97.5963 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F4.wav 97.9355 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F40.wav 79.5600 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F41.wav 98.3357 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F42.wav 95.8885 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F43.wav 95.3047 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F5.wav 97.1963 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F6.wav 97.4937 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F7.wav 90.4720 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F8.wav 97.7940 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\F\F9.wav 96.7003 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M1.wav 98.4715 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M10.wav 98.1158 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M11.wav 98.0778 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M12.wav 97.0116 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M13.wav 97.3985 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M14.wav 98.5584 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M15.wav 94.3353 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M16.wav 95.5498 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M17.wav 97.3357 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M18.wav 93.2862 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M19.wav 94.5206 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M2.wav 81.7980 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M20.wav 73.2144 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M21.wav 96.6519 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M22.wav 94.4182 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M23.wav 90.2008 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M24.wav 72.8826 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M25.wav 95.7111 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M26.wav 91.4482 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M27.wav 96.6088 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M28.wav 65.8460 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M29.wav 65.8460 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M3.wav 96.9562 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M30.wav 95.3557 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M31.wav 82.3203 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M32.wav 92.3744 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M33.wav 93.3840 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M34.wav 75.2944 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M35.wav 75.3624 의 확률로 여자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M36.wav 96.4766 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M37.wav 55.6053 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M38.wav 95.7213 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M39.wav 98.0130 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M4.wav 96.9299 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M40.wav 94.2039 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M41.wav 50.5643 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M42.wav 96.3184 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M43.wav 94.0417 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M5.wav 96.7142 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M6.wav 98.4087 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M7.wav 94.6625 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M8.wav 97.3834 의 확률로 남자입니다.
E:\nmb\nmb_data\5s_last_0510\predict_04_26\M\M9.wav 74.5487 의 확률로 남자입니다.
43개 여성 목소리 중 39개 정답
42개 남성 목소리 중 37개 정답
작업 시간 :  6:37:20.556160
"""