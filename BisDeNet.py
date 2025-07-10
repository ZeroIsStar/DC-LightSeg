# BisDeNet: A New Lightweight Deep Learning-Based Framework for Efficient Landslide Detection
# 2024
from keras._tf_keras.keras.layers import *
from keras._tf_keras.keras.models import Model
import keras as K
from keras._tf_keras.keras.optimizers import SGD
from context_model import VGG16,Xception,Xception1,ResNet50_1,ResNet18,DenseNet
import numpy as np

def AttentionRefinementModule(x, channel, name):

    ARM = GlobalAveragePooling2D(data_format='channels_last', name=name+'_globalpooling')(x)
    ARM = Reshape(target_shape=(1,1,channel), name=name+'_reshape')(ARM)
    ARM = Conv2D(channel, 1, strides=1, padding='same', activation='relu', name=name+'_conv')(ARM)
    ARM = BatchNormalization(name=name+'_bn')(ARM)
    ARM = Activation('sigmoid', name=name+'_sigmoid')(ARM)
    ARM = multiply([x,ARM], name=name+'_multiply')

    return ARM


def FeatureFusionModule(Spatial_Path, Context_Path, n_classes):

    y3 = concatenate([Spatial_Path,Context_Path])
    y3 = Conv2D(n_classes, 3, strides=1, padding='same', activation=None, name='FFM_conv1')(y3)

    y3 = BatchNormalization(name='y3_bn')(y3)
    y3 = Activation('relu', name='FFM_relu')(y3)# 32 32 2

    y4 = GlobalAveragePooling2D(data_format='channels_last')(y3)
    y4 = Reshape(target_shape=(1,1,-1))(y4)
    y4 = Conv2D(n_classes, 1, strides=1, padding='same', activation='relu', name='FFM_conv2')(y4)
    y4 = Conv2D(n_classes, 1, strides=1, padding='same', activation='sigmoid', name='FFM_conv3')(y4)


    y5 = multiply([y3,y4])
    y6 = add([y5,y3])

    return y6


def BiSeNet_ResNet18(n_classes):
    input_shape=[128,128,12]
    x = Input(shape=input_shape)

    #Spatial Path
    y1 = Conv2D(64, 3, strides=2, padding='same', activation=None, name='Spatial_Conv1')(x)
    y1 = BatchNormalization(name='Spatial_Conv1_bn')(y1)
    y1 = Activation('relu', name='Spatial_Conv1_relu')(y1)
    y1 = Conv2D(128, 3, strides=2, padding='same', activation=None, name='Spatial_Conv2')(y1)
    y1 = BatchNormalization(name='Spatial_Conv2_bn')(y1)
    y1 = Activation('relu', name='Spatial_Conv2_relu')(y1)# 32 32
    y1 = Conv2D(256, 3, strides=2, padding='same', activation=None, name='Spatial_Conv3')(y1)
    y1 = BatchNormalization(name='Spatial_Conv3_bn')(y1)  # 16 16 256
    _Spatial_Path = Activation('relu', name='Spatial_Conv3_relu')(y1)  # 16 16 256

    # Context Path
    y2_16, y2_32, y2_Global = DenseNet()(x) # 8 8 1024; 4 4 2048; ? 2048
    # print(y2_16.shape)
    # print(y2_32.shape)
    # print(y2_Global.shape)

    #ARM
    ARM1 = AttentionRefinementModule(y2_16, channel=1024, name='ARM1')
    ARM2 = AttentionRefinementModule(y2_32, channel=2048, name='ARM2')
    ARM2 = multiply([ARM2,y2_Global])
    ARM1 = UpSampling2D(size=(2, 2), data_format='channels_last', interpolation='bilinear')(ARM1)
    ARM2 = UpSampling2D(size=(4, 4), data_format='channels_last', interpolation='bilinear')(ARM2)
    _Context_Path = concatenate([ARM1,ARM2])



    #FFM
    FFM = FeatureFusionModule(_Spatial_Path, _Context_Path, n_classes)

    #output
    out = UpSampling2D(size=(8, 8), data_format='channels_last', interpolation='bilinear')(FFM)
    output = Activation('softmax', name='output_softmax')(out)

    model = Model(inputs=x, outputs=output)
    adam=K.optimizers.Adam(lr=0.01)
    model.compile(optimizer=adam,loss='categorical_crossentropy', metrics=['acc'])

    return model


def BiSeNet_ResNet50(input_shape, n_classes, training):


    x = Input(shape=input_shape)

    #Spatial Path
    y1 = Conv2D(64, 3, strides=2, padding='same', activation=None, name='Spatial_Conv1')(x)
    y1 = BatchNormalization(name='Spatial_Conv1_bn')(y1)
    y1 = Activation('relu', name='Spatial_Conv1_relu')(y1)
    y1 = Conv2D(128, 3, strides=2, padding='same', activation=None, name='Spatial_Conv2')(y1)
    y1 = BatchNormalization(name='Spatial_Conv2_bn')(y1)
    y1 = Activation('relu', name='Spatial_Conv2_relu')(y1)
    y1 = Conv2D(256, 3, strides=2, padding='same', activation=None, name='Spatial_Conv3')(y1)
    y1 = BatchNormalization(name='Spatial_Conv3_bn')(y1)
    _Spatial_Path = Activation('relu', name='Spatial_Conv3_relu')(y1)

    # Context Path
    y2_16, y2_32, y2_Global = ResNet50_1()(x)

    #ARM
    ARM1 = AttentionRefinementModule(y2_16, channel=1024, name='ARM1')
    ARM2 = AttentionRefinementModule(y2_32, channel=2048, name='ARM2')
    ARM2 = multiply([ARM2,y2_Global])
    ARM1 = UpSampling2D(size=(2, 2), data_format='channels_last', interpolation='bilinear')(ARM1)
    ARM2 = UpSampling2D(size=(4, 4), data_format='channels_last', interpolation='bilinear')(ARM2)
    _Context_Path = concatenate([ARM1,ARM2])

    #FFM
    FFM = FeatureFusionModule(_Spatial_Path, _Context_Path, n_classes)

    #output
    out = UpSampling2D(size=(8, 8), data_format='channels_last', interpolation='bilinear')(FFM)
    output = Activation('softmax', name='output_softmax')(out)

    if training == True:
        aux_1 = Conv2D(n_classes, 1, strides=1, padding='same', name='Aux1_Conv')(ARM1)
        aux_2 = Conv2D(n_classes, 1, strides=1, padding='same', name='Aux2_Conv')(ARM2)
        aux_1 = UpSampling2D(size=(8, 8), data_format='channels_last', interpolation='bilinear', name='Aux1_Upsampling1')(aux_1)
        aux_2 = UpSampling2D(size=(8, 8), data_format='channels_last', interpolation='bilinear', name='Aux1_Upsampling2')(aux_2)
        aux_1 = Activation('softmax', name='output_softmax1')(aux_1)
        aux_2 = Activation('softmax', name='output_softmax2')(aux_2)
        model = Model(inputs=x, outputs=[output, aux_2, aux_1])
    else:
        model = Model(inputs=x, outputs=output)

    optim = SGD(lr=0.01, momentum=0.9, decay=0.0005)
    model.compile(optim, 'categorical_crossentropy', loss_weights=[1.0, 1.0, 1.0], metrics=['categorical_accuracy'])

    return model


def BiSeNet_Xception(input_shape, n_classes, training):


    x = Input(shape=input_shape)

    #Spatial Path
    y1 = Conv2D(64, 3, strides=2, padding='same', activation=None, name='Spatial_Conv1')(x)
    y1 = BatchNormalization(name='Spatial_Conv1_bn')(y1)
    y1 = Activation('relu', name='Spatial_Conv1_relu')(y1)
    y1 = Conv2D(128, 3, strides=2, padding='same', activation=None, name='Spatial_Conv2')(y1)
    y1 = BatchNormalization(name='Spatial_Conv2_bn')(y1)
    y1 = Activation('relu', name='Spatial_Conv2_relu')(y1)
    y1 = Conv2D(256, 3, strides=2, padding='same', activation=None, name='Spatial_Conv3')(y1)
    y1 = BatchNormalization(name='Spatial_Conv3_bn')(y1)
    _Spatial_Path = Activation('relu', name='Spatial_Conv3_relu')(y1)

    # Context Path
    y2_16, y2_32, y2_Global = Xception()(x)
    # print(y2_16.shape)
    # print(y2_32.shape)
    # print(y2_Global.shape)

    #ARM
    ARM = GlobalAveragePooling2D(data_format='channels_last', name= '1_globalpooling')(y2_16)
    ARM = Reshape(target_shape=(1,1,1024), name='1_reshape')(ARM)
    ARM = Conv2D(1024, 1, strides=1, padding='same', activation='relu', name='1_conv')(ARM)
    ARM = BatchNormalization(name='1_bn')(ARM)
    ARM = Activation('sigmoid', name='1_sigmoid')(ARM)
    ARM1 = multiply([y2_16,ARM], name='1_multiply')

    ARM = GlobalAveragePooling2D(data_format='channels_last', name= '2_globalpooling')(y2_32)
    ARM = Reshape(target_shape=(1,1,2048), name='2_reshape')(ARM)
    ARM = Conv2D(2048, 1, strides=1, padding='same', activation='relu', name='2_conv')(ARM)
    ARM = BatchNormalization(name='2_bn')(ARM)
    ARM = Activation('sigmoid', name='2_sigmoid')(ARM)
    ARM2 = multiply([y2_32,ARM], name='2_multiply')



    # ARM1 = AttentionRefinementModule(y2_16, channel=1024, name='ARM1')
    # ARM2 = AttentionRefinementModule(y2_32, channel=2048, name='ARM2')

    ARM2 = multiply([ARM2,y2_Global])
    ARM1 = UpSampling2D(size=(2, 2), data_format='channels_last', interpolation='bilinear')(ARM1)
    ARM2 = UpSampling2D(size=(4, 4), data_format='channels_last', interpolation='bilinear')(ARM2)
    _Context_Path = concatenate([ARM1,ARM2])

    #FFM
    FFM = FeatureFusionModule(_Spatial_Path, _Context_Path, n_classes)

    #output-modified改动了三个地方（FFM的三个卷积层以及这里的上采样部分）
    out = UpSampling2D(size=(8, 8), data_format='channels_last', interpolation='bilinear')(FFM)
    output = Activation('softmax', name='output_softmax')(out)

    # out = UpSampling2D(size=(2, 2), data_format='channels_last', name='upsampling1', interpolation='bilinear')(FFM)
    # out = Conv2D(256, 3, padding='SAME', name='conv_modified1_3x3')(out)
    # out = Conv2D(256, 3, padding='SAME', name='conv_modified2_3x3')(out)
    # out = Conv2D(n_classes, 1, padding='SAME', name='conv_modified3_1x1')(out)
    # out = UpSampling2D(size=(4, 4), data_format='channels_last', name='upsampling2', interpolation='bilinear')(out)
    # output = Activation('softmax', name='output_softmax')(out)

    if training == True:
        aux_1 = Conv2D(n_classes, 1, strides=1, padding='same', name='Aux1_Conv')(ARM1)
        aux_2 = Conv2D(n_classes, 1, strides=1, padding='same', name='Aux2_Conv')(ARM2)
        aux_1 = UpSampling2D(size=(8, 8), data_format='channels_last', interpolation='bilinear', name='Aux1_Upsampling1')(aux_1)
        aux_2 = UpSampling2D(size=(8, 8), data_format='channels_last', interpolation='bilinear', name='Aux1_Upsampling2')(aux_2)
        aux_1 = Activation('softmax', name='output_softmax1')(aux_1)
        aux_2 = Activation('softmax', name='output_softmax2')(aux_2)
        model = Model(inputs=x, outputs=[output, aux_2, aux_1])
    else:
        model = Model(inputs=x, outputs=output)

    return model

def BiSeNet_DenseNet6(n_classes):

    input_shape=[128,128,12]
    x = Input(shape=input_shape)

    #Spatial Path
    y1 = Conv2D(64, 3, strides=2, padding='same', activation=None, name='Spatial_Conv1')(x)
    y1 = BatchNormalization(name='Spatial_Conv1_bn')(y1)
    y1 = Activation('relu', name='Spatial_Conv1_relu')(y1)
    y1 = Conv2D(128, 3, strides=2, padding='same', activation=None, name='Spatial_Conv2')(y1)
    y1 = BatchNormalization(name='Spatial_Conv2_bn')(y1)
    y1 = Activation('relu', name='Spatial_Conv2_relu')(y1)# 32 32
    y1 = Conv2D(256, 3, strides=2, padding='same', activation=None, name='Spatial_Conv3')(y1)
    y1 = BatchNormalization(name='Spatial_Conv3_bn')(y1)  # 16 16 256
    _Spatial_Path = Activation('relu', name='Spatial_Conv3_relu')(y1)  # 16 16 256

    # Context Path
    y2_16, y2_32, y2_Global = DenseNet()(x) # 8 8 35; 4 4 53; ? 53
    # print(y2_16.shape)
    # print(y2_32.shape)
    # print(y2_Global.shape)

    #ARM
    ARM1 = AttentionRefinementModule(y2_16, channel=35, name='ARM1') # 8 8 69
    ARM2 = AttentionRefinementModule(y2_32, channel=53, name='ARM2') # 4 4 70
    ARM2 = multiply([ARM2,y2_Global])# 4 4 70
    ARM1 = UpSampling2D(size=(2, 2), data_format='channels_last', interpolation='bilinear')(ARM1) # 16 16 69
    ARM2 = UpSampling2D(size=(4, 4), data_format='channels_last', interpolation='bilinear')(ARM2) # 16 16 70
    _Context_Path = concatenate([ARM1,ARM2])# 16 16



    #FFM
    FFM = FeatureFusionModule(_Spatial_Path, _Context_Path, n_classes)

    #output
    out = UpSampling2D(size=(8, 8), data_format='channels_last', interpolation='bilinear')(FFM)
    output = Activation('softmax', name='output_softmax')(out)

    model = Model(inputs=x, outputs=output)
    adam=K.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['acc'])

    return model


if __name__=='__main__':
    import os
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    model=BiSeNet_DenseNet6(2)
    # batch_size = 32
    # epochs = 10
    # x = np.random.rand(1, 128, 128, 12).astype('float32')
    # y_train = np.random.randint(0, 2, size=(1, 128, 128, 1)).astype('int32')
    # y_train = np.eye(2)[y_train[..., 0]]
    #
    # history = model.fit(
    #     x, y_train,
    #     batch_size=batch_size,
    #     epochs=epochs,
    # )
    model.summary()# 723685
    print('done')


