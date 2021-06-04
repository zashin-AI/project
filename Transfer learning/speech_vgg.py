'''
Module containing speechVGG code for network.
Based on the Keras VGG-16 implementation.
'''
# speechVGG는 심층 음성 기능 추출기로, 표현에서 응용 프로그램 및 음성 처리 문제에서 전이 학습을 위해 특별히 맞춤화됩니다. 
# 추출기는 고전적인 VGG-16 아키텍처를 채택하고 단어 인식 작업을 통해 학습됩니다.
# 일반화된 음성 표현은 사전 훈련된 모델에 의해 다른 데이터 세트를 사용하여 고유한 음성 처리 작업을 통해 전송할 수 있다

from keras.models import Model
from keras import layers

# include_top : 네트워크 상단에 3 개의 완전히 연결된 레이어를 포함할지 여부. 
#               연결하고 더 추가안해서 그냥 기본적인 speech 모델 사용
# weights : None(무작위 초기화), 'imagenet'(ImageNet 사전 학습) 중 하나
#           이미지넷의 사전 학습된 가중치이기 때문에 사용안함
# pooling :include_top이 False일 때 형상 추출을 위한 선택적 풀링 모드입니다. 
#         - none은 모델의 출력이 마지막 컨볼루션 블록의 4D 텐서 출력이 된다는 것을 의미합니다. 
#         - avg는 마지막 컨볼루션 블록의 출력에 global average pooling이 적용된다는 것을 의미하므로 모델의 출력이 2D 텐서가 됩니다.
#         - max는 global max pooling이 적용됩니다.
# classes : 이미지를 분류 할 선택적 클래스 수, include_top=True 인 경우에만 지정 되고 weights인수가 지정 되지 않은 경우에만 지정됩니다.

# model.trainable = False (전이학습 모델의 특징 추출 부분의 가중치가 고정, 
#                          고정하면 fit하는 중 지정된 층의 가중치가 업데이트 되지 않는다.)
# True(기본값)로 하면 전이학습 모델의 특징 추출 부분의 가중치가 흐트러지고 새로 훈련시키는 것



def speechVGG(include_top=True,
            weights=None, 
            input_shape=(128,128,1),
            classes=8,
            pooling=None,
            transfer_learning=False):

    img_input = layers.Input(shape=input_shape)

    # Block 1
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv1')(img_input)
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv1')(x)
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv1')(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv2')(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv1')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv2')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv1')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv2')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    if include_top: # = fc 계층
        # allows to load all layers except these ones that will finetune to task.
        if transfer_learning:
            add_string = '_new'
        else:
            add_string = ''
        # Classification block
        x = layers.Flatten(name='flatten' + add_string)(x)
        x = layers.Dense(256, activation='relu', name='fc1' + add_string)(x)
        x = layers.Dense(256, activation='relu', name='fc2' + add_string)(x)
        x = layers.Dense(classes, activation='softmax', name='predictions' + add_string)(x)
    else:
        x = x

    inputs = img_input

    # Create model.
    model = Model(inputs, x, name='speech_vgg')

    if weights:
        model.load_weights(weights, by_name=True)

    return model
