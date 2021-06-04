import numpy as np
import librosa
import sklearn
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, AveragePooling2D, Dropout, Activation, Flatten, Add, Input, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adadelta, Adam, Nadam, RMSprop

def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)

start_now = datetime.now()

# 데이터 불러오기
x = np.load('C:/nmb/nmb_data/npy/project_total_npy/total_data.npy')
y = np.load('C:/nmb/nmb_data/npy/project_total_npy/total_label.npy')

print(x.shape, y.shape) # (4536, 128, 862) (4536,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=42
)

# aaa = 1
# x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], aaa)
# x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], aaa)

# print(x_train.shape, y_train.shape) # (858, 128, 862, 1) (858,)
# print(x_test.shape, y_test.shape)   # (215, 128, 862, 1) (215,)

# 모델 구성
model = Sequential()

def residual_block(x, units, conv_num=3, activation='relu'):  # ( input, output node, for 문 반복 횟수, activation )
    # Shortcut
    s = Dense(units)(x)
    for i in range(conv_num - 1):
        x = Dense(units)(x)
        x = Activation(activation)(x)
    x = Dense(units)(x)
    # x = Add()([x,s])
    x = Concatenate(axis=-1)([x, s])
    return Activation(activation)(x)
    
def build_model(input_shape, num_classes):
    inputs = Input(shape=input_shape, name='input')

    # x = residual_block(inputs, 16, 2)
    # x = residual_block(x, 32, 2)
    # x = residual_block(x, 64, 3)
    # x = residual_block(x, 128, 3)
    # x = residual_block(x, 128, 23)

    # Total params: 8,964,818
    # Trainable params: 8,964,818
    # Non-trainable params: 0

    x = residual_block(inputs, 1024, 2)
    x = residual_block(x, 512, 2)
    x = residual_block(x, 512, 3)
    x = residual_block(x, 256, 3)
    x = residual_block(x, 256, 3)

    # Total params: 24,614,018
    # Trainable params: 24,614,018
    # Non-trainable params: 0

    x = Flatten()(x)
    x = Dense(256, activation="relu")(x)
    x = Dense(128, activation="relu")(x)

    outputs = Dense(num_classes, activation='softmax', name="output")(x)
    
    return Model(inputs=inputs, outputs=outputs)

model = build_model(x_train.shape[1:], 2)
print(x_train.shape[1:])    # (128, 862, 1)

model.summary()

op = Adam(lr=1e-3)
batch_size = 32

# 컴파일, 훈련
model.compile(optimizer=op, loss="sparse_categorical_crossentropy", metrics=['acc'])
es = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
lr = ReduceLROnPlateau(monitor='val_loss', vactor=0.5, patience=10, verbose=1)
path = 'C:/nmb/nmb_data/h5/5s/DNN_adam_1.h5'
mc = ModelCheckpoint(path, monitor='val_loss', verbose=1, save_best_only=True)
# tb = TensorBoard(log_dir='C:/nmb/nmb_data/graph',histogram_freq=0, write_graph=True, write_images=True)
history = model.fit(x_train, y_train, epochs=300, batch_size=16, validation_split=0.2, callbacks=[es, lr, mc])

# 평가, 예측
model.load_weights('C:/nmb/nmb_data/h5/5s/DNN_adam_1.h5')

# 평가, 예측
# model = load_model('C:/nmb/nmb_data/h5/5s/DNN_adam_1.h5')
model.load_weights('C:/nmb/nmb_data/h5/5s/DNN_adam_1.h5')
result = model.evaluate(x_test, y_test, batch_size=8)
print("loss : {:.5f}".format(result[0]))
print("acc : {:.5f}".format(result[1]))

############################################ PREDICT ####################################

pred = ['C:/nmb/nmb_data/predict_04_26/F', 'C:/nmb/nmb_data/predict_04_26/M']

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
        pred_mels = pred_mels.reshape(1, pred_mels.shape[0], pred_mels.shape[1])
        y_pred = model.predict(pred_mels)
        y_pred_label = np.argmax(y_pred)
        if y_pred_label == 0:   # 여성이라고 예측
            print(file, '{:.4f} 의 확률로 여자입니다.', format((y_pred[0][0])*100))
            if name == 'F' :
                count_f = count_f + 1
        else:                   # 남성이라고 예측
            print(file, '{:.4f} 의 확률로 남자입니다.', format((y_pred[0][1])*100))
            if name == 'M' :
                count_m = count_m + 1
print("43개 여성 목소리 중 "+str(count_f)+"개 정답")
print("43개 남성 목소리 중 "+str(count_m)+"개 정답")

end = datetime.now()
time = end - start_now
print("작업 시간 : ", time)

# 시각화
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.suptitle('Densenet')

plt.subplot(2, 1, 1)    # 2행 1열중 첫번째
plt.plot(history.history['loss'], marker='.', c='red', label='loss')
plt.plot(history.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()

plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')

plt.subplot(2, 1, 2)    # 2행 1열중 두번째
plt.plot(history.history['acc'], marker='.', c='red', label='acc')
plt.plot(history.history['val_acc'], marker='.', c='blue', label='val_acc')
plt.grid()

plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(loc='upper right')
plt.show()

# loss : 0.16942
# acc : 0.92731
# 43개 여성 목소리 중 40개 정답
# 43개 남성 목소리 중 41개 정답
# 작업 시간 :  0:18:03.439263