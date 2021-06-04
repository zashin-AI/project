import numpy as np
from tensorflow.keras.layers import Dense, LeakyReLU, Dropout, Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# --------------------------------------------------------------------------
# 데이터 불러오기
fm_ds = np.load('C:/nmb/nmb_data/npy/1s_2m_total_fm_data.npy')
f_ds = fm_ds[:9600]

f_ds = f_ds.reshape(f_ds.shape[0], f_ds.shape[1]*f_ds.shape[2])
print(f_ds.shape)
# (9600, 22144)
# ------------------------------------------------------------------------
# Normalize
# generator 마지막에 activation이 tanh.
# tanh을 거친 output 값이 -1~1 사이로 나오기 때문에 최대 1 최소 -1 로 맞춰줘야 한다.

print(np.max(f_ds), np.min(f_ds))
# 3.8146973e-06 -80.0

from sklearn.preprocessing import MaxAbsScaler, StandardScaler
scaler1 = StandardScaler()
scaler1.fit(f_ds)
f_ds = scaler1.transform(f_ds)

scaler2 = MaxAbsScaler()
scaler2.fit(f_ds)
f_ds = scaler2.transform(f_ds)

# 이 값이 -1 ~ 1 사이에 있는지 확인
print(np.max(f_ds), np.min(f_ds))
# 1.0 -0.9439434
# 최대한 비슷하게 맞춰줌

# ------------------------------------------------------------------------
# Hyperparameters 설정값 지정

# gan에 입력되는 noise에 대한 dimension
NOISE_DIM = 100

# adam optimizer 정의, learning_rate = 0.0002, beta_1로 줍니다.
# Vanilla Gan과 DCGAN에서 이렇게 셋팅을 해주는데
# 이렇게 해줘야 훨씬 학습을 잘합니다.
adam = Adam(lr=0.0002, beta_1=0.5)

# ------------------------------------------------------------------------
# Generator 생성자 함수 정의

generator = Sequential([
    Dense(256, input_dim=NOISE_DIM), 
    LeakyReLU(0.2), 
    Dense(512), 
    LeakyReLU(0.2), 
    Dense(1024), 
    LeakyReLU(0.2), 
    Dense(22144, activation='tanh'),    # 데이터 쉐잎과 맞춰줌(22144)
])

generator.summary()
# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# dense (Dense)                (None, 256)               25856     
# _________________________________________________________________
# leaky_re_lu (LeakyReLU)      (None, 256)               0
# _________________________________________________________________
# dense_1 (Dense)              (None, 512)               131584    
# _________________________________________________________________
# leaky_re_lu_1 (LeakyReLU)    (None, 512)               0
# _________________________________________________________________
# dense_2 (Dense)              (None, 1024)              525312    
# _________________________________________________________________
# leaky_re_lu_2 (LeakyReLU)    (None, 1024)              0
# _________________________________________________________________
# dense_3 (Dense)              (None, 22144)             22697600  
# =================================================================
# Total params: 23,380,352
# Trainable params: 23,380,352
# Non-trainable params: 0
# _________________________________________________________________

# ------------------------------------------------------------------------
# Discriminator 판별자 함수 정의
discriminator = Sequential([
    Dense(1024, input_shape=(22144,),
    kernel_initializer=RandomNormal(stddev=0.02)),
    LeakyReLU(0.2), 
    Dropout(0.3), 
    Dense(512),
    LeakyReLU(0.2), 
    Dropout(0.3), 
    Dense(256),
    LeakyReLU(0.2), 
    Dropout(0.3), 
    Dense(1, activation='sigmoid')
])

discriminator.summary()
# Model: "sequential_1"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# dense_4 (Dense)              (None, 1024)              22676480
# _________________________________________________________________
# leaky_re_lu_3 (LeakyReLU)    (None, 1024)              0
# _________________________________________________________________
# dropout (Dropout)            (None, 1024)              0
# _________________________________________________________________
# dense_5 (Dense)              (None, 512)               524800
# _________________________________________________________________
# leaky_re_lu_4 (LeakyReLU)    (None, 512)               0
# _________________________________________________________________
# dropout_1 (Dropout)          (None, 512)               0
# _________________________________________________________________
# dense_6 (Dense)              (None, 256)               131328
# _________________________________________________________________
# leaky_re_lu_5 (LeakyReLU)    (None, 256)               0
# _________________________________________________________________
# dropout_2 (Dropout)          (None, 256)               0
# _________________________________________________________________
# dense_7 (Dense)              (None, 1)                 257
# =================================================================
# Total params: 23,332,865
# Trainable params: 23,332,865
# Non-trainable params: 0
# _________________________________________________________________

# compile
# 반드시 컴파일 해주어야 함
discriminator.compile(loss='binary_crossentropy', optimizer=adam)



# ------------------------------------------------------------------------
# Gan 모델을 만들자
# generator와 discriminator를 연결

discriminator.trainable = False         # discriminator는 학습을 하지 않음
gan_input = Input(shape=(NOISE_DIM,))
x = generator(inputs=gan_input)         # generator만 학습을 진행
output = discriminator(x)

# gan 모델을 정의
gan = Model(gan_input, output)

gan.summary()
# Model: "functional_1"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# input_1 (InputLayer)         [(None, 500)]             0             # 10은 Noise_dim
# _________________________________________________________________
# sequential (Sequential)      (None, 22144)             23482752      # 원하는 데이터 쉐이프
# _________________________________________________________________
# sequential_1 (Sequential)    (None, 1)                 23332865      # 0 진짜 or 1 가짜
# =================================================================     
# Total params: 46,815,617
# Trainable params: 23,482,752
# Non-trainable params: 23,332,865
# _________________________________________________________________

# Model: "functional_1"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# input_1 (InputLayer)         [(None, 100)]             0
# _________________________________________________________________
# sequential (Sequential)      (None, 22144)             23380352
# _________________________________________________________________
# sequential_1 (Sequential)    (None, 1)                 23332865
# =================================================================
# Total params: 46,713,217
# Trainable params: 23,380,352
# Non-trainable params: 23,332,865
# _________________________________________________________________
# compile
gan.compile(loss='binary_crossentropy', optimizer=adam)

# ------------------------------------------------------------------------
# batch 생성

def get_batches(data, batch_size):
    batches = []
    for i in range(int(data.shape[0] // batch_size)):
        batch = data[i * batch_size: (i + 1) * batch_size]
        batches.append(batch)
    return np.asarray(batches)
    
# ------------------------------------------------------------------------
# 시각화, 저장

def visualize_training(epoch, d_losses, g_losses):
    # 오차에 대한 시각화
    plt.figure(figsize=(8, 4))
    plt.plot(d_losses, label='Discriminator Loss')
    plt.plot(g_losses, label='Generatror Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    # plt.show()
    plt.savefig('C:/nmb/gan_0504/hist/'+ filename + '._{}.png'.format(str(epoch).zfill(5)), bbox_inches='tight')

    print('epoch: {}, Discriminator Loss: {}, Generator Loss: {}'.format(epoch, np.asarray(d_losses).mean(), np.asarray(g_losses).mean()))
    
    #샘플 데이터 생성 후 시각화
    noise = np.random.normal(0, 1, size=(24, NOISE_DIM))
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(-1, 128, 173)

    # generator로 생성된 npy 저장
    np.save('C:/nmb/gan_0504/npy/'+ filename + '_total{}.npy'.format(str(epoch).zfill(5)), arr=generated_images)
    
    # 이미지 시각화하여 저장
    plt.figure(figsize=(8, 4))
    for i in range(generated_images.shape[0]):
        plt.subplot(4, 6, i+1)
        plt.imshow(generated_images[i], interpolation='nearest', cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    # plt.show()
    plt.savefig('C:/nmb/gan_0504/visualize/'+ filename + '._{}.png'.format(str(epoch).zfill(5)), bbox_inches='tight')

# ------------------------------------------------------------------------
# 학습

# 배치사이즈와 에폭 지정
BATCH_SIZE = 100
EPOCHS= 50

filename = 'b100_e50_n500'

# discriminator와 gan 모델의 loss 측정을 위한 list
d_losses = []
g_losses = []

# 시간 측정
start = datetime.now()

for epoch in range(1, EPOCHS + 1):
    # 각 배치별 학습
    for real_images in get_batches(f_ds, BATCH_SIZE):
        # 랜덤 노이즈 생성
        input_noise = np.random.uniform(-1, 1, size=[BATCH_SIZE, NOISE_DIM])
        
        # 가짜 이미지 데이터 생성
        generated_images = generator.predict(input_noise)
        
        # Gan에 학습할 X 데이터 정의
        x_dis = np.concatenate([real_images, generated_images])
        
        # Gan에 학습할 Y 데이터 정의
        y_dis = np.zeros(2 * BATCH_SIZE)
        y_dis[:BATCH_SIZE] = 0.9                                                # 반(진짜)에는 0.9, 나머지 반(위조)에는 0     
                                                                                # 왜 0.9인가? https://kakalabblog.wordpress.com/2017/07/27/gan-tutorial-2016/
                                                                                # One-sided label smoothing : 진짜(1.0)에 가까운 0.9를 줌으로 D의 극심한 추정을 막는다.
        
        # Discriminator 훈련                                                    # discriminator 먼저 학습
        discriminator.trainable = True
        d_loss = discriminator.train_on_batch(x_dis, y_dis)                     # 이때 돌아가는 역전파는 D에만 영향
        
        # Gan 훈련                                                              # 다음 gan 학습
        noise = np.random.uniform(-1, 1, size=[BATCH_SIZE, NOISE_DIM])
        y_gan = np.ones(BATCH_SIZE)                                             # 일부러 라벨링 1(진짜)로 들어감, D가 잘 판단하는지 보려고
        
        # Discriminator의 판별 학습을 방지
        discriminator.trainable = False                                         # D에 역전파 돌아감을 방지
        g_loss = gan.train_on_batch(noise, y_gan)                               # 위에서 갱신 된 D와 G를 이어준 gan 모델을 학습
        
    d_losses.append(d_loss)
    g_losses.append(g_loss)
    
    if epoch == 1 or epoch % 10 == 0:
        visualize_training(epoch, d_losses, g_losses)

# 시간 측정
end = datetime.now()
time = end - start
print("작업 시간 : " , time)