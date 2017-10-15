from keras.models import Model
import keras.backend as K
from keras.layers import Input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from scipy.misc import imresize
import numpy as np

from transform import TransformNet
from loss import create_loss_fn

CONTENT_WEIGHT = 0.5
STYLE_WEIGHT = 100
TV_WEIGHT = 200

def create_gen(img_dir, target_size, batch_size):
    datagen = ImageDataGenerator()
    gen = datagen.flow_from_directory(img_dir, target_size=target_size,
                                      batch_size=batch_size, class_mode=None)

    def tuple_gen():
        for img in gen:
            yield (img, img)

    return tuple_gen

style_img_path = 'wave.jpg'
style_img = image.load_img(style_img_path)
style_target = image.img_to_array(style_img)

content_img_path = 'content.jpg'
content_img = image.load_img(content_img_path)

# resize to 256x256 for training
content_img = imresize(content_img, (256, 256, 3))
content_target = image.img_to_array(content_img)

# Needed so that certain layers function in training mode (batch norm)
K.set_learning_phase(1)

inputs = Input(shape=(256, 256, 3))
transform_net = TransformNet(inputs)
model = Model(inputs=inputs, outputs=transform_net)
loss_fn = create_loss_fn(style_target, CONTENT_WEIGHT, STYLE_WEIGHT, TV_WEIGHT)
model.compile(optimizer='adam', loss=loss_fn)

#content_target = np.expand_dims(content_target, axis=0)
#X = [content_target]
#y = [content_target]
#model.fit(X, y)

gen = create_gen('data', target_size=(256, 256), batch_size=1)
model.fit_generator(gen, steps_per_epoch=82783)
