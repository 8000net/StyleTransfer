from keras.models import Model
import keras.backend as K
from keras.callbacks import Callback
from keras.layers import Input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from scipy.misc import imresize, imsave
import numpy as np
import pandas as pd

from transform import TransformNet
from loss import create_loss_fn

def create_gen(img_dir, target_size, batch_size):
    datagen = ImageDataGenerator()
    gen = datagen.flow_from_directory(img_dir, target_size=target_size,
                                      batch_size=batch_size, class_mode=None)

    def tuple_gen():
        for img in gen:
            yield (img, img)

    return tuple_gen()


if __name__ == '__main__':
    content_weight = 7.5
    style_weight = 100
    tv_weight = 200
    batch_size = 4
    preview_increment = 50
    style_img_path = 'wave.jpg'
    test_img_path = 'doge.jpg'
    model_output_path = 'wave.h5'
    train_path = 'data'
    epochs = 2

    # Needed so that certain layers function in training mode (batch norm)
    K.set_learning_phase(1)

    # This needs to be in scope where model is defined
    class OutputPreview(Callback):
        def __init__(self, test_img_path, increment):
            test_img = image.load_img(test_img_path)
            test_img = imresize(test_img, (256, 256, 3))
            test_target = image.img_to_array(test_img)
            test_target = np.expand_dims(test_target, axis=0)
            self.test_img = test_target

            self.increment = increment
            self.batch_num = 0

        def on_batch_end(self, batch, logs={}):
            if (self.batch_num % self.increment == 0):
                output = model.predict(self.test_img)[0]
                imsave('preview/test-preview-%d.jpg' % self.batch_num, output)

            self.batch_num += 1




    style_img = image.load_img(style_img_path)
    style_target = image.img_to_array(style_img)

    inputs = Input(shape=(256, 256, 3))
    transform_net = TransformNet(inputs)
    model = Model(inputs=inputs, outputs=transform_net)
    loss_fn = create_loss_fn(style_target, content_weight,
                             style_weight, tv_weight, batch_size)
    model.compile(optimizer='adam', loss=loss_fn)

    gen = create_gen(train_path, target_size=(256, 256), batch_size=batch_size)
    output_preview = OutputPreview(test_img_path, increment=preview_increment)
    history = model.fit_generator(gen, steps_per_epoch=82780, epochs=2,
                                  callbacks=[output_preview])

    model.save(model_output_path)
    #pd.DataFrame(history.history).to_csv('wave.csv')

