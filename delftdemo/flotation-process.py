import os
import numpy as np

from PIL import Image as ImagePIL
#j = Image.fromarray(b)
#j.save('img2.png')

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

def gauss_cwt(image_complex, scale, image_height, image_width):
    """ Calculates the Gauss CWT """
    pass

def ifft2_cImage_to_matrix(wavelet_image, scale, image_height, image_width):
    """ Inverse FFT2 """
    pass

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