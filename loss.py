# TODO: write these functions with symbolic backend functions

from keras.applications.vgg19 import VGG19
from keras.applications import vgg19
from keras.models import Model
from keras.layers import Input
from keras import backend as K
import numpy as np

STYLE_LAYERS = ('block1_conv1', 'block2_conv1',
                'block3_conv1', 'block4_conv1',
                'block5_conv1')

CONTENT_LAYER = 'block4_conv2'

# TODO: remove need for this
BATCH_SIZE = 1
BATCH_SHAPE = (BATCH_SIZE, 256, 256, 3)


def l2_loss(x):
    return np.sum(x**2) / 2


def get_vgg_features(input, layers):
    # TODO: this needs to handle symbolic tensors

    # When model is compiled, input will be a symbolic tensor
    # from the last layer in the model.
    #
    # We can't pass input straight to VGG with input_tensor,
    # because input_tensor expects an Input layer only,
    # other load_weights() fails, because the weights are
    # for 16 layers, but the model now has 16 + n layers
    #
    # Shape is not known at compile time either, so
    # we can't pass input_shape=input.shape
    #
    # Possible solutions:
    #   - Use VGG default input size and reshape tensor?
    #   - Recreate VGG from scratch, or pass input tensor through layers?
    #
    print(input)
    vgg = VGG19(include_top=False, input_shape=input.shape[1:])
    outputs = [layer.output for layer in vgg.layers if layer.name in layers]
    return outputs

    # This works on np arrays:
    #
    #vgg = VGG19(include_top=False, input_shape=input.shape)
    #
    #input = np.expand_dims(input, axis=0)
    #input = vgg19.preprocess_input(input)   

    #inp = vgg.input
    #outputs = [layer.output for layer in vgg.layers if layer.name in layers]
    #functor = K.function([inp]+ [K.learning_phase()], outputs )
    #return functor([input, 1.])
   

def calculate_content_loss(content_image, reconstructed_image, content_weight):
    content_features = get_vgg_features(content_image, CONTENT_LAYER)[0]
    reconstructed_content_features = get_vgg_features(
            reconstructed_image, CONTENT_LAYER)[0]
    
    content_loss = content_weight * (2 * l2_loss(
        reconstructed_content_vgg_features - content_features) / content_size)
    
    return content_loss
    
def calculate_style_loss(style_image, reconstructed_image, style_weight):
     # Get outputs of style and content images at VGG layers
    style_vgg_features = get_vgg_features(style_image, STYLE_LAYERS)
    reconstructed_style_vgg_features = get_vgg_features(
            reconstructed_image, STYLE_LAYERS)
    
    # Calculate the style features and content features
    # Style features are the gram matrices of the VGG feature maps
    style_grams = []
    style_rec_grams = []
    for features in style_vgg_features:
        features = np.reshape(features, (-1, features.shape[3]))
        gram = np.matmul(features.T, features) / features.size
        style_grams.append(gram)
        
    for features in reconstructed_style_vgg_features:
        batch_size, h, w, filters = features.shape
        features = np.reshape(features, (batch_size, h * w, filters))
        gram = np.matmul(features.T, features) / features.size
        style_rec_grams.append(gram)       
        
    # Style loss
    style_loss = 0
    for style_gram, style_rec_grams in zip(style_grams, style_rec_grams):
        style_loss += 2 * l2_loss(style_gram - style_rec_gram) / style_gram.size
    
    style_loss *= style_weight
    
    return style_loss
    
    
def calculate_tv_loss(x):
    tv_y_size = x[:,1:,:,:].size
    tv_x_size = x[:,:,1:,:].size
    y_tv = l2_loss(x[:,1:,:,:] - x[:,:BATCH_SHAPE[1]-1,:,:])
    x_tv = l2_loss(x[:,:,1:,:] - x[:,:,:BATCH_SHAPE[2]-1,:])
    tv_loss = tv_weight*2*(x_tv/tv_x_size + y_tv/tv_y_size)/BATCH_SIZE
    return tv_loss


def create_loss_fn(style_image, content_weight, style_weight, tv_weight):
    def style_transfer_loss(y_true, y_pred):
        """
        y_true - content_image
        y_pred - reconstructed image
        """
        content_image = y_true
        reconstructed_image = y_pred
        
        content_loss = calculate_content_loss(content_image,
                reconstructed_image, content_weight)
        style_loss = calculate_style_loss(style_image,
                reconstructed_image, style_weight)
        tv_loss = calculate_tv_loss(reconstructed_image, tv_weight)

        return content_loss + style_loss + tv_loss

    return style_transfer_loss
