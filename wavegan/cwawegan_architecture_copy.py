from tensorflow.keras.layers import Input, Conv1D, Flatten, Dense, Conv2DTranspose, Reshape, Lambda, LeakyReLU, ReLU, Embedding, Concatenate, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow import pad, maximum, random, int32

#Original WaveGAN: https://github.com/chrisdonahue/wavegan
#Label embeding using the method in https://machinelearningmastery.com/how-to-develop-a-conditional-generative-adversarial-network-from-scratch/

#TODO: clean/redo this
# TODO : 정리/다시 실행 -> generator 부분에서 실행

def Conv1DTranspose(input_tensor, filters, kernel_size, strides=4, padding='same'
                    , name = '1DTConv', activation = 'relu'):
    x = Conv2DTranspose(filters=filters, kernel_size=(1, kernel_size), strides=(1, strides), padding=padding, 
                        name = name, activation = activation)(K.expand_dims(input_tensor, axis=1)) 
                        # input_tensor의 쉐이프 차원에서 두번째 차원을 추가하여 확장한다.
    x = K.squeeze(x, axis=1) 
    # x의 쉐이프의 차원 중 사이즈 1인 것을 찾아서 제거한다.
    return x

def generator(z_dim = 100,
              architecture_size = 'audio_size',
              n_classes = 2):
        
    generator_filters = [1024, 512, 256, 128, 64]

    label_input = Input(shape=(1,), dtype='int32', name='generator_label_input')
    label_em = Embedding(n_classes, n_classes * 20, name = 'label_embedding')(label_input) # 양의 정수 (인덱스)를 고정 된 크기의 조밀 한 벡터로 변환합니다.
    label_em = Dense(16, name = 'label_dense')(label_em)
    label_em = Reshape((16, 1), name = 'label_respahe')(label_em)
    
    generator_input = Input(shape=(z_dim,), name='generator_input')
    x = generator_input

    if architecture_size == 'audio_size':
        x = Dense(32768, name='generator_input_dense')(x)
        x = Reshape((16, 2048), name='generator_input_reshape')(x)
        
    x = ReLU()(x)
    
    x = Concatenate()([x, label_em]) 
    
    if architecture_size == 'audio_size':
        #layer 0 to 4
        for i in range(5):
            x = Conv1DTranspose(
                input_tensor = x
                , filters = generator_filters[i]
                , kernel_size = 25
                , strides = 4
                , padding='same'
                , name = f'generator_Tconv_{i}'
                , activation = 'relu'
                )
        
        #layer 5
        x = Conv1DTranspose(
            input_tensor = x
            , filters = 1
            , kernel_size = 25
            , strides = 7
            , padding='same'
            , name = 'generator_Tconv_5'
            , activation = 'tanh'
            ) 
    
    generator_output = x 
    generator = Model([generator_input, label_input], generator_output, name = 'Generator')
    return generator

model = generator()
# model.summary()

def discriminator(architecture_size='audio_size',
                  n_classes = 2):
    
    # discriminator_filters = [64, 128, 256, 512, 1024, 2048]
    discriminator_filters = [4, 16, 64, 256, 1024, 4096]
    
    if architecture_size == 'audio_size':
        audio_input_dim = 114688
        
    label_input = Input(shape=(1,), dtype='int32', name='discriminator_label_input')
    label_em = Embedding(n_classes, n_classes * 20)(label_input) # 양의 정수 (인덱스)를 고정 된 크기의 조밀 한 벡터로 변환합니다.
    label_em = Dense(audio_input_dim)(label_em)
    label_em = Reshape((audio_input_dim, 1))(label_em)

    discriminator_input = Input(shape=(audio_input_dim, 1), name='discriminator_input')
    x = Concatenate()([discriminator_input, label_em]) 

    if architecture_size == 'audio_size':
        
        # layers
        for i in range(4):
            x = Conv1D(
                filters = discriminator_filters[i]
                , kernel_size = 25
                , strides = 4
                , padding = 'same'
                , name = f'discriminator_conv_{i}'
                )(x)
            x = LeakyReLU(alpha = 0.2)(x)

        #last 2 layers without phase shuffle
        x = Conv1D(
            filters = discriminator_filters[4]
            , kernel_size = 25
            , strides = 4
            , padding = 'same'
            , name = 'discriminator_conv_4'
            )(x)
        x = LeakyReLU(alpha = 0.2)(x)
        
        x = Conv1D(
            filters = discriminator_filters[5]
            , kernel_size = 25
            , strides = 4
            , padding = 'same'
            , name = 'discriminator_conv_5'
            )(x)
        x = LeakyReLU(alpha = 0.2)(x)
        x = Flatten()(x)
        
    discriminator_output = Dense(1)(x)
    discriminator = Model([discriminator_input, label_input], discriminator_output, name = 'Discriminator')
    return discriminator

model = discriminator()
model.summary()