from keras.models import Model
from keras.layers import Input
from keras.preprocessing import image
from scipy.misc import imresize

from transform import TransformNet
from loss import create_loss_fn

# For testing:
from loss import get_vgg_features, CONTENT_LAYER


CONTENT_WEIGHT = 0.5
STYLE_WEIGHT = 100
TV_WEIGHT = 200

style_img_path = 'wave.jpg'
style_img = image.load_img(style_img_path)
style_target = image.img_to_array(style_img)

content_img_path = 'content.jpg'
content_img = image.load_img(content_img_path)

# resize to 256x256 for training
content_img = imresize(content_img, (256, 256, 3))
content_target = image.img_to_array(content_img)

# Set 
# K.set_learning_phase(1)

inputs = Input(shape=(256, 256, 3))
transform_net = TransformNet(inputs)
model = Model(inputs=inputs, outputs=transform_net)
loss_fn = create_loss_fn(style_target, CONTENT_WEIGHT, STYLE_WEIGHT, TV_WEIGHT)
#model.compile(optimizer='adam', loss=loss_fn)

#print(get_vgg_features(content_target, CONTENT_LAYER))
