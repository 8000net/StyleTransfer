from argparse import ArgumentParser
import os

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--model', type=str,
                        dest='model', help='model path',
                        metavar='MODEL', required=True)

    parser.add_argument('--input', type=str,
                        dest='input', help='input image path',
                        metavar='INPUT', required=True)

    parser.add_argument('--output', type=str,
                        dest='output', help='output image path',
                        metavar='OUTPUT', required=True)

    return parser


def check_opts(options):
    assert os.path.exists(options.model), "model path not found!"
    assert os.path.exists(options.input), "input path not found!"

parser = build_parser()
options = parser.parse_args()
check_opts(options)

from keras.layers import Input
from keras.models import Model
from keras.preprocessing import image
import numpy as np
from scipy.misc import imsave

from transform import TransformNet

# Get input image
input_img = image.load_img(options.input)
input_img = image.img_to_array(input_img)
h, w, c = input_img.shape
input_img = np.expand_dims(input_img, axis=0)


# Load model
inputs = Input(shape=(h, w, c))
transform_net = TransformNet(inputs)
model = Model(inputs, transform_net)
model.load_weights(options.model)

output = model.predict([input_img])[0]
imsave(options.output, output)
