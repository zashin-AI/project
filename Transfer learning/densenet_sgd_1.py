from itertools import count
import numpy as np
import os
import librosa
import sklearn
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import MobileNet, VGG16, DenseNet121
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, AveragePooling2D, Dropout, Activation, Flatten, Add, Input, Concatenate, LeakyReLU, ReLU
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adadelta, Adam, Nadam, RMSprop, SGD

start_now = datetime.now()

# 데이터 불러오기
x = np.load('C:/nmb/nmb_data/npy/project_total_npy/total_data.npy')
y = np.load('C:/nmb/nmb_data/npy/project_total_npy/total_label.npy')

print(x.shape, y.shape) # (1073, 128, 862) (1073,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=42
)

aaa = 1 
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], aaa)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], aaa)
print(x_train.shape, y_train.shape) # (3628, 128, 862, 1) (3628,)
print(x_test.shape, y_test.shape)   # (908, 128, 862, 1) (908,)

model = DenseNet121(
    include_top=True,
    input_shape=(128,862,1),
    classes=2,
    pooling=None,
    weights=None,
)

model.summary()
# model.trainable = False

model.save('C:/nmb/nmb_data/h5/5s/densenet/densenet_sgd_1.h5')

# 컴파일, 훈련
op = SGD(lr=1e-2)
batch_size = 4

es = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
lr = ReduceLROnPlateau(monitor='val_loss', vactor=0.5, patience=10, verbose=1)
path = 'C:/nmb/nmb_data/h5/5s/densenet/densenet_sgd_1.h5'
mc = ModelCheckpoint(path, monitor='val_loss', verbose=1, save_best_only=True)

model.compile(optimizer=op, loss="sparse_categorical_crossentropy", metrics=['acc'])
history = model.fit(x_train, y_train, epochs=1000, batch_size=batch_size, validation_split=0.2, callbacks=[es, lr, mc])

# 평가, 예측
model = load_model('C:/nmb/nmb_data/h5/5s/densenet/densenet_sgd_1.h5')
# model.load_weights('C:/nmb/nmb_data/h5/5s/densenet/densenet_sgd_1.h5')
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

# loss : 0.00311
# acc : 0.99890
# 43개 여성 목소리 중 42개 정답
# 43개 남성 목소리 중 42개 정답
# 작업 시간 :  1:15:48.632067
