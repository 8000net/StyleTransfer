from keras.layers import Conv2D, Conv2DTranspose, Input, Lambda
from keras.models import Model
import keras.layers

from keras_contrib.layers.normalization import InstanceNormalization

def Conv2DInstanceNorm(inputs, filters, kernel_size,
                    strides=1, activation='relu'):
    return InstanceNormalization()(
            Conv2D(
                filters,
                (kernel_size, kernel_size),
                strides=strides,
                activation=activation,
                padding='same'
            )(inputs))

def Conv2DTransposeInstanceNorm(inputs, filters, kernel_size,
                              strides=1, activation=None):
    return InstanceNormalization()(
            Conv2DTranspose(
                filters,
                (kernel_size, kernel_size),
                strides=strides,
                activation=activation,
                padding='same'
            )(inputs))

def Conv2DResidualBlock(inputs):
    tmp     = Conv2DInstanceNorm(inputs, 128, 3)
    tmp2    = Conv2DInstanceNorm(tmp, 128, 3, activation=None)
    return keras.layers.add([tmp, tmp2]) 

# TODO: init weights?
def TransformNet(inputs):
    conv1   = Conv2DInstanceNorm(inputs, 32, 9)
    conv2   = Conv2DInstanceNorm(conv1, 64, 3, strides=2)
    conv3   = Conv2DInstanceNorm(conv2, 128, 3, strides=2)
    resid1  = Conv2DResidualBlock(conv3)
    resid2  = Conv2DResidualBlock(resid1)
    resid3  = Conv2DResidualBlock(resid2)
    resid4  = Conv2DResidualBlock(resid3)
    resid5  = Conv2DResidualBlock(resid4)
    conv_t1 = Conv2DTransposeInstanceNorm(conv3, 64, 3, strides=2)
    conv_t2 = Conv2DTransposeInstanceNorm(conv_t1, 32, 3, strides=2)
    conv_t3 = Conv2DInstanceNorm(conv_t2, 3, 9, activation='tanh')
    preds = Lambda(lambda x : x * 150 + 255./2)(conv_t3)
    return preds
