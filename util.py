import multiprocessing.pool
from functools import partial
import os

from keras.preprocessing.image import _count_valid_files_in_directory

def count_num_samples(directory):
    """
    From Keras DirectoryIterator
    """
    classes = []
    for subdir in sorted(os.listdir(directory)):
        if os.path.isdir(os.path.join(directory, subdir)):
            classes.append(subdir)

    white_list_formats = {'png', 'jpg', 'jpeg', 'bmp', 'ppm'}
    pool = multiprocessing.pool.ThreadPool()
    function_partial = partial(_count_valid_files_in_directory,
                               white_list_formats=white_list_formats,
                               follow_links=False)
    num_samples = sum(pool.map(function_partial,
                               (os.path.join(directory, subdir)
                                for subdir in classes)))
    pool.close()
    pool.join()
    return num_samples
