# pip install -U PIL
# pip install -U numpy
# or, just use Anaconda3, which has everything installed already

import os
import multiprocessing
import datetime
import numpy as np
from PIL import Image as ImagePIL

start_dir = '/Users/kevindunn/Delft/DelftDemo/delftdemo/working-directory/'

class Image(object):
    """
    Base image class to read, and do basic image processing using the
    Python PIL to do the heavy work.
    """
    def __init__(self, filename='', imgobj=None):
        self.rows = 0
        self.cols = 0
        self.layers = 0
        self.filename = filename
        if filename:
            self.img = ImagePIL.open(filename)
            self.rows = self.img.size[1]
            self.cols = self.img.size[0]

        if isinstance(imgobj, ImagePIL.Image):
            # For the case when constructing the object from a PIL image object
            self.img = imgobj
            self.rows = self.img.size[1]
            self.cols = self.img.size[0]

    def subsample(self):
        """Returns a subsampled image, as an instance of this class."""
        new_size = (int(self.cols/2.0), int(self.rows/2.0))
        return Image(imgobj=self.img.resize(new_size))

    def to_gray(self):
        """Returns a grayscale image, as another instance of this class."""
        return Image(imgobj=self.img.convert(mode="L"))


def fft2_image(image_1D):
    """ Calculates FFT2 image, returns it as an instance of ``np.array``."""
    raw_data = np.asarray(image_1D.img)
    return np.fft.fft2(raw_data)

#@profile
def gauss_cwt(inFFT, scale, height, width):
    """ Performs the Gaussian Continuous Wavelet Transformation, given the FFT2
    transform of the image, ``inImg``. It does that at the required ``scale``,
    Some information about the size of the original image, ``height`` and
    ``width`` is also required to set up the iFFT2 at the end.
    """
    # Create the height and width pulses. TODO: vectorize this
    h_pulse = np.zeros(height)
    multiplier_h = 2 * np.pi / height
    split = np.floor(height/2)
    for k in np.arange(0, split):
        h_pulse[k] = np.power(scale * k * multiplier_h, 2)
        h_pulse[k+split] = np.power(- scale * multiplier_h * (split-k), 2)

    w_pulse = np.zeros(width)
    multiplier_w = 2 * np.pi / width
    split = width/2
    for k in np.arange(0, split):
        w_pulse[k] = np.power(scale * k * multiplier_w, 2)
        w_pulse[k+split] = np.power(-scale * multiplier_w * (split-k), 2)

    # Multiply with the incoming FFT image, and return the ``outFFT`` output
    neg_sigma_sq = -0.5*np.power(1, 2.0)
    h_matrix, v_matrix = np.meshgrid(w_pulse, h_pulse)
    multiplier_test = np.exp(neg_sigma_sq * (h_matrix + v_matrix))
    outFFT = inFFT * multiplier_test

    return outFFT


def ifft2_cImage_to_matrix(in_image, scale, height, width):
    """
    Recreates an image from the complex inputs by using the inverse fast
    Fourier transform.

    You must also specify the recreated image dimensions: height (rows) and
    width (columns).
    """
    n_elements = height * width;
    outIm = np.fft.ifft2(in_image)
    outIm = np.abs(outIm * scale / n_elements)
    return Image(imgobj = ImagePIL.fromarray(outIm))


def threshold(restored):
    """TODO: Thresholds the image"""

    return [restored.img.getpixel((0,0)), 0]


def project_onto_model(features):
    """ TODO: Projects the results on the PCA model."""
    return np.sum(features)


#@profile
def flotation_image_processing(filename):
    """A single function that processes the flotation image given in the
    filename.

    The image processing pipeline:
    """
    print('Image: {0} with process id {1}'.format(filename, os.getpid()))
    raw_image = Image(start_dir + filename)
    image_1D = raw_image.subsample().to_gray();
    image_complex = fft2_image(image_1D)

    features = []
    for scale in np.arange(1, 13, 2):
        wavelet_image = gauss_cwt(image_complex, scale,
                                    image_1D.rows, image_1D.cols)

        restored = ifft2_cImage_to_matrix(wavelet_image, scale,
                                          image_1D.rows,
                                          image_1D.cols)

        f1f2 = threshold(restored)
        features.append(f1f2[0])

    calc_outputs = project_onto_model(features)
    return calc_outputs


if __name__ == '__main__':

    # Currently 1.28 seconds per image (no thresholding step yet)
    #   * 98.2% of the time is in the GaussCWT function

    file_list = []
    for filename in os.listdir(path=start_dir):
        if filename.endswith('.bmp'):
            file_list.append(filename)
            #flotation_image_processing(filename)
            #print(filename)

    print(datetime.datetime.now())

    # Start as many workers as there are CPUs
    pool = multiprocessing.Pool(processes=4)
    result = pool.map(flotation_image_processing, file_list)
    pool.close() # No more tasks can be added to the pool
    pool.join()  # Wrap up all current tasks and terminate

    print(result)

    print(datetime.datetime.now())

