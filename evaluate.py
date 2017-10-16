from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from scipy.misc import imresize, imsave

from keras_contrib.layers import InstanceNormalization
from loss import create_loss_fn

# Load model
CONTENT_WEIGHT = 0.5
STYLE_WEIGHT = 100
TV_WEIGHT = 200

BATCH_SIZE = 4


style_img_path = 'wave.jpg'
style_img = image.load_img(style_img_path)
style_target = image.img_to_array(style_img)

loss_fn = create_loss_fn(style_target, CONTENT_WEIGHT,
                         STYLE_WEIGHT, TV_WEIGHT, BATCH_SIZE)

model = load_model('wave-bs4.h5', custom_objects={
    'InstanceNormalization': InstanceNormalization,
    'style_transfer_loss': loss_fn
})


# Get output
content_img_path = 'content.jpg'
content_img = image.load_img(content_img_path)

content_img = imresize(content_img, (256, 256, 3))
content_target = image.img_to_array(content_img)
content_target = np.expand_dims(content_target, axis=0)


output = model.predict([content_target])[0]
imsave('doge-bs4.jpg', output)
