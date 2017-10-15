from functools import reduce

from keras.models import Model, Sequential
from keras.layers import Input
from keras import backend as K
import numpy as np
import tensorflow as tf

from vgg import VGG19, preprocess_input

STYLE_LAYERS = ('block1_conv1', 'block2_conv1',
                'block3_conv1', 'block4_conv1',
                'block5_conv1')

CONTENT_LAYER = 'block4_conv2'

CONTENT_TRAINING_SIZE = (256, 256, 3)

def tensor_size(x):
    return np.nanprod(np.array(K.int_shape(x), dtype=np.float))

def l2_loss(x):
    return K.sum(K.square(x)) / 2

def get_vgg_features(input, layers, input_shape):
    if len(K.int_shape(input)) == 3:
        input = K.expand_dims(input, axis=0)
    input = preprocess_input(input)
    vgg = VGG19(input, input_shape)
    outputs = [layer.output for layer in vgg.layers if layer.name in layers]
    return outputs
  

def calculate_content_loss(content_image, reconstructed_image,
                           content_weight, image_shape):
    content_features = get_vgg_features(
            content_image, CONTENT_LAYER, image_shape)[0]
    reconstructed_content_features = get_vgg_features(
            reconstructed_image, CONTENT_LAYER, image_shape)[0]
   
    content_size = tensor_size(content_features)
    content_loss = content_weight * (2 * l2_loss(
        reconstructed_content_features - content_features) / content_size)
    
    return content_loss
    
def calculate_style_loss(style_image, reconstructed_image,
                         style_weight, style_image_shape, content_image_shape,
                         batch_size):
     # Get outputs of style and content images at VGG layers
    style_vgg_features = get_vgg_features(
            style_image, STYLE_LAYERS, style_image_shape)
    reconstructed_style_vgg_features = get_vgg_features(
            reconstructed_image, STYLE_LAYERS, content_image_shape)
    
    # Calculate the style features and content features
    # Style features are the gram matrices of the VGG feature maps
    style_grams = []
    style_rec_grams = []
    for features in style_vgg_features:
        _, h, w, filters = K.int_shape(features)

        # shape in K.reshape needs to be np.array to convert Dimension to int
        # (should be fixed in newer versions of Tensorflow)
        features = K.reshape(features, np.array((1, h * w, filters)))

        features_size = tensor_size(features)
        features_T = tf.transpose(features, perm=[0,2,1])
        gram = tf.matmul(features_T, features) / features_size
        style_grams.append(gram)
        
    for features in reconstructed_style_vgg_features:
        _, h, w, filters = K.int_shape(features)

        # Need to know batch_size ahead of time
        features = K.reshape(features, np.array((batch_size, h * w, filters)))

        features_size = tensor_size(features)
        features_T = tf.transpose(features, perm=[0,2,1])
        gram = tf.matmul(features_T, features) / features_size
        style_rec_grams.append(gram)       
        
    # Style loss
    style_losses = []
    for style_gram, style_rec_gram in zip(style_grams, style_rec_grams):
        style_gram_size = tensor_size(style_gram)
        l2 = l2_loss(style_gram - style_rec_gram)
        style_losses.append(2 * l2 / style_gram_size)
    
    style_loss = style_weight * reduce(tf.add, style_losses) / batch_size
    
    return style_loss
    
    
def calculate_tv_loss(x, tv_weight, batch_size):
    tv_y_size = tensor_size(x[:,1:,:,:])
    tv_x_size = tensor_size(x[:,:,1:,:])
    y_tv = l2_loss(x[:,1:,:,:] - x[:,:CONTENT_TRAINING_SIZE[0]-1,:,:])
    x_tv = l2_loss(x[:,:,1:,:] - x[:,:,:CONTENT_TRAINING_SIZE[1]-1,:])
    tv_loss = tv_weight*2*(x_tv/tv_x_size + y_tv/tv_y_size)/batch_size
    return tv_loss


def create_loss_fn(style_image, content_weight,
                   style_weight, tv_weight, batch_size):
    style_image = tf.convert_to_tensor(style_image)

    def style_transfer_loss(y_true, y_pred):
        """
        y_true - content_image
        y_pred - reconstructed image
        """

        content_image = y_true
        reconstructed_image = y_pred
        
        content_loss = calculate_content_loss(content_image,
                reconstructed_image, content_weight, CONTENT_TRAINING_SIZE)
        style_loss = calculate_style_loss(style_image,
                reconstructed_image, style_weight, K.int_shape(style_image),
                CONTENT_TRAINING_SIZE, batch_size)
        tv_loss = calculate_tv_loss(reconstructed_image, tv_weight, batch_size)

        loss = content_loss + style_loss + tv_loss
        return loss

    return style_transfer_loss
