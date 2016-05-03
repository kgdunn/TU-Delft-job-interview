# pip install -U PIL
# pip install -U numpy
# or, just use Anaconda3, which has everything installed already

import os
import multiprocessing
import datetime
import numpy as np
from PIL import Image as ImagePIL

start_dir = '/Users/kevindunn/Delft/DelftDemo/delftdemo/working-directory/'
#start_dir = '/Users/kevindunn/Delft/DelftDemo/delftdemo/delftdemo/'

class Param(object):
    """
    A structure that holds the model parameters that are used to process
    each image. The structure is considered "read-only", i.e. no results
    are stored in it.
    """
    def __init__(self):
        self.subsample_image = True
        self.start_level = 1         # start and ends levels for the
        self.end_level = 13          # wavelet resolution
        self.sigma_xy = 1.0          # Gaussian coefficient
        self.percent_retained = 0.85 # percentage energy retained
        self.tolerance = 0.000001    # thresholding tolerance for convergence
        self.n_components = 2        # number of PCA components
        self.display_results = False # writes intermediate files to disk
                                     # so images can be viewed later

        self.mean_vector = np.array([0.026289407767760, 0.016596246522800,
                                     0.010541172410390, 0.007098798177090,
                                     0.005045800751860, 0.003766555672320])
        self.scaling_vector = np.array([0.007311898849230, 0.002373383101510,
                                        0.002344658601980, 0.002067958707430,
                                        0.001679164709720, 0.001339596033420])
        # Each component stored in a row, with ``n_features`` per row: 2x6
        self.loadings = np.array([[-0.3731, 0.2012, 0.4418,
                                                     0.4612, 0.4583, 0.4499],
                                  [-0.5057, -0.8117, -0.2479,
                                                    -0.0237, 0.0789, 0.1310]]);


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
    # Create the height and width pulses.
    multiplier_h = 2 * np.pi / height
    split = np.floor(height/2)
    left = np.arange(0, split, step=1) * scale * multiplier_h
    right = np.arange(-split, 0, step=1) * scale * multiplier_h
    h_pulse = np.power(np.hstack((left, right)), 2)

    multiplier_w = 2 * np.pi / width
    split = width/2
    left = np.arange(0, split, step=1) * scale * multiplier_w
    right = np.arange(-split, 0, step=1) * scale * multiplier_w
    w_pulse = np.power(np.hstack((left, right)), 2)

    # Multiply with the incoming FFT image, and return the ``outFFT`` output
    neg_sigma_sq = -0.5 * np.power(1, 2.0)
    h_matrix, v_matrix = np.meshgrid(w_pulse, h_pulse)
    multiplier_test = np.exp(neg_sigma_sq * (h_matrix + v_matrix))
    outFFT = inFFT * multiplier_test

    return outFFT


def ifft2_cImage_to_matrix(in_image, scale):
    """
    Recreates an image from the complex inputs by using the inverse fast
    Fourier transform.

    You must also specify the recreated image dimensions: height (rows) and
    width (columns).
    """
    outIm = np.fft.ifft2(in_image)
    outIm = np.abs(outIm * scale)
    return Image(imgobj = ImagePIL.fromarray(outIm))


def norm_threshold(X, apply_thresh=True, thresh=0.0):
    """
    Calculates the sum-of-squares of the elements. Or, if ``apply_thresh``
    ``thresh`` are provided, it will only do this on elements in X (not X^2)
    that exceed that given ``thresh`` value.

    MATLAB: normvalue = sum(sum( ((A>=thresh).*A).^2 ));  % for matrix A
    """
    out = 0
    if apply_thresh:
        return ((X ** 2) * (X>=thresh)).sum()
    else:
        return (X * X).sum()

    return out

#@profile
def threshold(inImg, model):
    """
    Thresholds the wavelet coefficients based on retained energy. Calculates
    the percentage of pixels that exceeds this energy level.
    Returns: 2-element vector
       1/ The percentage retained coefficients.
       2/ The threshold value computed to retain an energy level.
    """
    # 1. Initialize parameters required to determine the thresholding value
    per_retained = model.percent_retained
    X = np.asarray(inImg.img)

    # 2. We need some basic statistics about the image.

    minX = X.min()
    maxX = X.max()
    sumX = X.sum()
    meanX = X.mean()
    stdX = X.std()

    base_energy = norm_threshold(X, False)

    # 3. Set up search algorithm to find energy level. Uses the Golden section
    #    search routine (https://en.wikipedia.org/wiki/Golden_section_search)
    R = (np.sqrt(5)-1)/2  # Golden ratio: 0.61803399
    C = 1 - R
    x0 = minX
    x3 = maxX
    x1 = x0 + C*(x3-x0)
    x2 = x3 - C*(x3-x0)

    f1 = np.abs(norm_threshold(X, True, x1)/base_energy - per_retained);
    f2 = np.abs(norm_threshold(X, True, x2)/base_energy - per_retained);

    # 4. Run the search algorithm in a while loop, protecting for the case of
    #    non-convergence. Allows a maximum of 100 iterations. Most images
    #    use around 30 to 38 iterations.
    n_iter = 0;
    max_iterations = 100;
    while ((np.abs(x3-x0)/stdX > model.tolerance) and (n_iter < max_iterations)):
        n_iter += 1
        if (f2 < f1):
            x0 = x1
            x1 = x2
            x2 = R*x1 + C*x3
            f1 = f2
            f2 = np.abs(norm_threshold(X, 1, x2)/base_energy - per_retained)

        else:
            x3 = x2
            x2 = x1
            x1 = R*x2 + C*x0
            f2 = f1
            f1 = np.abs(norm_threshold(X, 1, x1)/base_energy - per_retained)

    if (n_iter >= max_iterations):
        print("Maximum number of while loop iterations exceeded. ")
        assert(False)
    else:
        #print("Required {0} iterations.".format(n_iter))
        pass

    # Finished. Assign the outputs and return.
    if (f1 < f2):
        output1 = x1
    else:
        output1 = x2

    # MATLAB: = sum(sum(A >= ThrValue))
    # The percentage of the pixels that exceed the energy level
    output0 = (X > output1).sum() / X.size
    return (output0, output1)


def project_onto_model(features, model):
    """

    Projects (applies) the calculated features from the image onto a pre-
    existing PCA model. That PCA model was built from the features extracted
    on the training image data.

    In other words, this function is seeing how similar/dissimilar the
    current image is in comparison with the training data.


    First difference the features. (That explains why we count features
    starting at -1 in ``param load_model_parameters(...)``.
    """

    n_features = len(features)
    pca_features  = np.zeros((1, n_features))
    mcuv_features = np.zeros((1, n_features))

    # Then mean center and unit-variance (mcuv) scale after calculating
    # the features.
    pca_features = np.diff(features)
    mcuv_features = (pca_features - model.mean_vector) / model.scaling_vector

    # Calculate the PCA model outputs: the scores, and the SPE vector
    pca_scores = np.dot(mcuv_features,  model.loadings.T)
    spe_vector = np.dot(pca_scores, model.loadings)
    spe_value = norm_threshold(spe_vector, False)

    # Place the PCA scores and the SPE value in the return vector.
    # And we are finished!
    return (pca_scores.tolist(), spe_value)



def flotation_image_processing(filename):
    """A single function that processes the flotation image given in the
    filename.

    The image processing pipeline:
    """
    print('Image: {0} with process id {1}'.format(filename, os.getpid()))

    model = Param()
    raw_image = Image(start_dir + filename)
    image_1D = raw_image.subsample().to_gray();
    image_complex = fft2_image(image_1D)

    features = []
    for scale in np.linspace(model.start_level, model.end_level,
                             (model.end_level-model.start_level)/2+1):
        wavelet_image = gauss_cwt(image_complex, scale,
                                  image_1D.rows, image_1D.cols)

        restored = ifft2_cImage_to_matrix(wavelet_image, scale)

        f1f2 = threshold(restored, model)
        features.append(f1f2[0])

    calc_outputs = project_onto_model(features, model)
    return calc_outputs


if __name__ == '__main__':

    # Currently 1.28 seconds per image (no thresholding step yet)
    #   * 98.2% of the time is in the GaussCWT function

    # Now down to 0.4 seconds per images with Numpy vectorization.

    print(datetime.datetime.now())
    file_list = []
    for filename in os.listdir(path=start_dir):
        if filename.endswith('.bmp'):
            file_list.append(filename)
            #print(flotation_image_processing(filename))
            #print(filename)



    print(datetime.datetime.now())

    # Start as many workers as there are CPUs
    pool = multiprocessing.Pool(processes=1)
    result = pool.map(flotation_image_processing, file_list)
    pool.close()    # No more tasks can be added to the pool
    pool.join()     # Wrap up all current tasks and terminate

    print(result)

    print(datetime.datetime.now())

