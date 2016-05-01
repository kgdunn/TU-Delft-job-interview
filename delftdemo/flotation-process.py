import os
import numpy as np

from PIL import Image as ImagePIL

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


    # Starting the computations for the Gaussian. Set up storage for the
    # FFTW structure. Note (again) that the column dimension is half the
    # image width, plus 1 column padding. So when we do the convolution below,
    # we are doing it with the knowledge of the symmetry.
    #halfwidth = (width / 2) + 1

    neg_sigma_sq = -0.5*np.power(1, 2.0)
    multiplier = 0.0
    idx = 0
    outFFT = np.zeros(inFFT.shape, dtype=np.complex128)
    for k in np.arange(0, height):
        for j in np.arange(0, width):
            idx = k*width + j
            multiplier = np.exp( neg_sigma_sq * (w_pulse[j] + h_pulse[k]) )
            outFFT[k, j] = inFFT[k, j] * multiplier;


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
    """Thresholds the image"""
    return [0, 0]

def project_onto_model(features):
    """ Projects the results on the PCA model."""
    pass

def flotation_pipeline(filename):
    """A single function that processes the flotation image given in the
    filename.

    The image processing pipeline:
    """
    raw_image = Image(filename)
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

if __name__ == '__main__':

    start_dir = '/Users/kevindunn/Delft/DelftDemo/delftdemo/working-directory/'
    for filename in os.listdir(path=start_dir):
        if filename.endswith('.bmp'):
            flotation_pipeline(start_dir + filename)