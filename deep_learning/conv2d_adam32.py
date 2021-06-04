import numpy as np
import os
import librosa
import sklearn
from datetime import datetime
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D,MaxPooling2D, AveragePooling2D, Dropout, Activation, Flatten, Add, Input, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import Adadelta, Adam, Nadam, RMSprop
from tensorflow.keras.layers import LeakyReLU,ReLU


# 데이터 불러오기
f_ds = np.load('C:\\nmb\\nmb_data\\npy\\female_denoise_mel_data.npy')
f_lb = np.load('C:\\nmb\\nmb_data\\npy\\female_denoise_mel_label.npy')
m_ds = np.load('C:\\nmb\\nmb_data\\npy\\male_denoise_mel_data.npy')
m_lb = np.load('C:\\nmb\\nmb_data\\npy\\male_denoise_mel_label.npy')

x = np.concatenate([f_ds, m_ds], 0)
y = np.concatenate([f_lb, m_lb], 0)
print(x.shape, y.shape) 
# (3840, 128, 862) (3840,)

# 전처리
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.8, random_state=42
)
aaa = 1
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], aaa)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], aaa)
print(x_train.shape, y_train.shape) # (3072, 128, 862, 1) (3072,)
print(x_test.shape, y_test.shape)   # (768, 128, 862, 1) (768,) 

# 모델 구성

model = Sequential()
def residual_block(x, filters, conv_num=3, activation='relu'): 
    s = Conv2D(filters, 3, padding='same')(x)
    for i in range(conv_num - 1):
        x = Conv2D(filters, 3, padding='same')(x)
        x = Activation(activation)(x)
    x = Conv2D(filters, 3, padding='same')(x)
    x = Add()([x, s])
    x = Activation(activation)(x)
    return MaxPool2D(pool_size=2, strides=2)(x)

def build_model(input_shape, num_classes):
    inputs = Input(shape=input_shape, name='input')
    x = residual_block(inputs, 16, 2)
    x = residual_block(x, 32, 2)
    x = residual_block(x, 64, 3)
    x = residual_block(x, 128, 4)
    x = residual_block(x, 256, 5)
    x = AveragePooling2D(pool_size=3, strides=3)(x)
    x = Flatten()(x)
    x = Dense(256, activation="relu")(x)
    x = Dense(128, activation="relu")(x)
    x = Dense(64, activation="relu")(x)
    outputs = Dense(num_classes, activation='softmax', name="output")(x)
    return Model(inputs=inputs, outputs=outputs)
model = build_model(x_train.shape[1:], 2)

print(x_train.shape[1:])    # (128, 862, 1)
model.summary()

model.save('C:/nmb/nmb_data/h5/Conv2D_model_t11.h5')

start = datetime.now()
# 컴파일, 훈련
op = Adadelta(lr=1e-2)
batch_size =32

es = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
lr = ReduceLROnPlateau(monitor='val_loss', vactor=0.5, patience=10, verbose=1)
path = 'C:/nmb/nmb_data/h5/Conv2D_weight_t11.h5'
mc = ModelCheckpoint(path, monitor='val_loss', verbose=1, save_best_only=True)
tb = TensorBoard(log_dir='C:/nmb/nmb_data/graph/'+ start.strftime("%Y%m%d-%H%M%S") + "/",histogram_freq=0, write_graph=True, write_images=True)
model.compile(optimizer=op, loss="sparse_categorical_crossentropy", metrics=['acc'])
# history = model.fit(x_train, y_train, epochs=5000, batch_size=batch_size, validation_split=0.2, callbacks=[es, lr, mc, tb])


# 평가, 예측
model.load_weights('C:/nmb/nmb_data/h5/Conv2D_weight_t11.h5')
result = model.evaluate(x_test, y_test, batch_size=batch_size)
print("loss : {:.5f}".format(result[0]))
print("acc : {:.5f}".format(result[1]))

############################################ PREDICT ####################################

pathAudio_F = 'C:\\nmb\\nmb_data\\0422test'


files_F = librosa.util.find_files(pathAudio_F, ext=['flac','wav'])

aaa=1
'''
for file in files_F:
    y, sr = librosa.load(file, sr=22050, duration=5.0)
    mels = librosa.feature.melspectrogram(y, sr=sr, n_fft=512, hop_length=128, n_mels=128)
    mels = librosa.amplitude_to_db(mels, ref=np.max)
    mels = mels.reshape(1, mels.shape[0], int(mels.shape[1]/aaa), aaa)
    y_pred = model.predict(mels)
    y_pred_label = np.argmax(y_pred)
    if y_pred_label == 0 :  # 여성이라고 예측
        print(file,'{:.4f} %의 확률로 여자입니다.'.format((y_pred[0][0])*100))
    else:                   # 남성이라고 예측              
        print(file,'{:.4f} %의 확률로 남자입니다.'.format((y_pred[0][1])*100))
'''
# loss : 0.05219
# acc : 0.98568
# C:\nmb\nmb_data\predict\gan\F10000.wav 91.0954 %의 확률로 남자입니다.
# C:\nmb\nmb_data\predict\gan\F20000.wav 55.5631 %의 확률로 남자입니다.
# C:\nmb\nmb_data\predict\gan\F30000.wav 96.7462 %의 확률로 남자입니다.
# C:\nmb\nmb_data\predict\gan\F5000.wav 79.6266 %의 확률로 남자입니다.
# 작업 시간 :  0:00:04.071813


for file in files_F:
    y, sr = librosa.load(file, sr=22050)
    mels = librosa.feature.melspectrogram(y, sr=sr, n_fft=512, hop_length=128)
    print(mels.shape) #(128, 5168)
    mels = librosa.amplitude_to_db(mels, ref=np.max)
    print(mels.shape) #(128, 5168)
    mels = mels.reshape(1, mels.shape[0], int(mels.shape[1]/aaa), aaa)
    print(mels.shape) #(1, 128, 5168, 1)
    y_pred = model.predict(mels)
    y_pred_label = np.argmax(y_pred)
    if y_pred_label == 0 :  # 여성이라고 예측
        print(file,'{:.4f} %의 확률로 여자입니다.'.format((y_pred[0][0])*100))
    else:                   # 남성이라고 예측              
        print(file,'{:.4f} %의 확률로 남자입니다.'.format((y_pred[0][1])*100))