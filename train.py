from argparse import ArgumentParser
import os

CONTENT_WEIGHT = 1.5e1
STYLE_WEIGHT = 1e2
TV_WEIGHT = 2e2
BATCH_SIZE = 4
NUM_EPOCHS = 2
TRAIN_PATH = 'data'
# num images in MSCOCO / 4 (default batch size)
STEPS_PER_EPOCH = 20695

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--style', type=str,
                        dest='style', help='style image path',
                        metavar='STYLE', required=True)

    parser.add_argument('--model-output', type=str,
                        dest='model_output', help='model output path',
                        metavar='MODEL_OUTPUT', required=True)

    parser.add_argument('--test', type=str,
                        dest='test', help='test image path',
                        metavar='TEST', default=False)

    parser.add_argument('--test-dir', type=str,
                        dest='test_dir', help='test image save dir',
                        metavar='TEST_DIR', default=False)

    parser.add_argument('--test-increment', type=int,
                        dest='test_increment', help='number of batches to test after',
                        metavar='TEST_INCREMENT', default=100)

    parser.add_argument('--train-path', type=str,
                        dest='train_path', help='path to training images folder',
                        metavar='TRAIN_PATH', default=TRAIN_PATH)

    parser.add_argument('--epochs', type=int,
                        dest='epochs', help='num epochs',
                        metavar='EPOCHS', default=NUM_EPOCHS)

    parser.add_argument('--batch-size', type=int,
                        dest='batch_size', help='batch size',
                        metavar='BATCH_SIZE', default=BATCH_SIZE)

    parser.add_argument('--steps-per-epoch', type=int,
                        dest='steps_per_epoch',
                        help='number of batches of samples per epoch, ' + \
                             '(should be # of samples / batch size)',
                        metavar='BATCH_SIZE', default=STEPS_PER_EPOCH)

    parser.add_argument('--content-weight', type=float,
                        dest='content_weight',
                        help='content weight (default %(default)s)',
                        metavar='CONTENT_WEIGHT', default=CONTENT_WEIGHT)

    parser.add_argument('--style-weight', type=float,
                        dest='style_weight',
                        help='style weight (default %(default)s)',
                        metavar='STYLE_WEIGHT', default=STYLE_WEIGHT)

    parser.add_argument('--tv-weight', type=float,
                        dest='tv_weight',
                        help='total variation regularization weight (default %(default)s)',
                        metavar='TV_WEIGHT', default=TV_WEIGHT)

    return parser


def check_opts(opts):
    assert os.path.exists(opts.style), "style image path not found!"
    assert os.path.exists(opts.train_path), "train path not found!"
    if opts.test or opts.test_dir:
        assert os.path.exists(opts.test), "test image not found!"
        assert os.path.exists(opts.test_dir), "test directory not found!"
    if opts.test:
        assert options.test_dir != False, "test output dir must be given with test"
    assert opts.epochs > 0
    assert opts.batch_size > 0
    assert opts.content_weight >= 0
    assert opts.style_weight >= 0
    assert opts.tv_weight >= 0


parser = build_parser()
options = parser.parse_args()
check_opts(options)


from keras.models import Model
import keras.backend as K
from keras.callbacks import Callback
from keras.layers import Input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from scipy.misc import imresize, imsave
import numpy as np
import pandas as pd
import tensorflow as tf

from transform import TransformNet
from loss import create_loss_fn


def create_gen(img_dir, target_size, batch_size):
    datagen = ImageDataGenerator()
    gen = datagen.flow_from_directory(img_dir, target_size=target_size,
                                      batch_size=batch_size, class_mode=None)

    def tuple_gen():
        for img in gen:
            # (X, y)
            # X will go through TransformNet,
            # y will go through VGG
            yield (img/255., img)

    return tuple_gen()


# Needed so that certain layers function in training mode
K.set_learning_phase(1)

# This needs to be in scope where model is defined
class OutputPreview(Callback):
    def __init__(self, test_img_path, increment, preview_dir_path):
        test_img = image.load_img(test_img_path)
        test_img = imresize(test_img, (256, 256, 3))
        test_target = image.img_to_array(test_img)
        test_target = np.expand_dims(test_target, axis=0)
        self.test_img = test_target

        self.preview_dir_path = preview_dir_path

        self.increment = increment
        self.iteration = 0

    def on_batch_end(self, batch, logs={}):
        if (self.iteration % self.increment == 0):
            output_img = model.predict(self.test_img)[0]
            fname = '%d.jpg' % self.iteration
            out_path = os.path.join(self.preview_dir_path, fname)
            imsave(out_path, output_img)

        self.iteration += 1


style_img = image.load_img(options.style)
style_target = image.img_to_array(style_img)

inputs = Input(shape=(256, 256, 3))
transform_net = TransformNet(inputs)
model = Model(inputs=inputs, outputs=transform_net)
loss_fn = create_loss_fn(style_target, options.content_weight,
                         options.style_weight, options.tv_weight,
                         options.batch_size)
model.compile(optimizer='adam', loss=loss_fn)

gen = create_gen(options.train_path, target_size=(256, 256),
                 batch_size=options.batch_size)
callbacks = None
if options.test:
    callbacks = [OutputPreview(options.test, options.test_increment,
                               options.test_dir)]
model.fit_generator(gen, steps_per_epoch=options.steps_per_epoch,
                    epochs=options.epochs, callbacks=callbacks)
model.save(options.model_output)
