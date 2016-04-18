
#include <cstdio>
#include <iostream>
#include <memory>
#include <algorithm>
#include <vector>
#include <cmath>

// 3rd party libraries. Please see README.txt to see how to install and set up.
#include <fftw3.h>
#include "bitmap_image.hpp"
#include "Eigen/Core"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core_c.h"
#include "opencv2/core/persistence.hpp"

using namespace cv;

// Our libraries
#include "flotation.h"

param load_model_parameters(std::string directory, std::string filename){
    // Loads the model parameters from an XML file. The commented code here will
    // create a basic barebones XML file for you that you can tweak by hand,
    // or, of course, uncomment the code to let the software do it for you.
    //
    if(true){
        // Once off: if you don't have an YML file, and want to create a basic
        // one then run this code. After that, you can hand-edit the YML.
        cv::FileStorage fs("model-parameters.yml", cv::FileStorage::WRITE);
        fs << "subsample_image" << 1;       // cuts image size by 4
        fs << "start_level" << 1;           // start and ends levels for the
        fs << "end_level" << 13;            // wavelet resolution
        fs << "sigma_xy" << 1.0;            // Gaussian coefficient
        fs << "percent_retained" << 0.85;   // percentage retained
        fs << "n_components" << 2;          // number of PCA components
        fs << "display_results" << true;    // writes intermediate files to disk
                                            // so images can be viewed later

        cv::Mat mean_vector = (cv::Mat_<double>(1,6) << 0.026289407767760,
                      0.016596246522800, 0.010541172410390, 0.007098798177090,
                      0.005045800751860, 0.003766555672320);
        fs << "mean_vector" << mean_vector;
        
        cv::Mat scaling_vector = (cv::Mat_<double>(1,6) <<  0.007311898849230,
                      0.002373383101510, 0.002344658601980, 0.002067958707430,
                      0.001679164709720, 0.001339596033420);
        fs << "scaling_vector" << scaling_vector;
        
        // Each component stored in a row, with ``n_features`` per row: 2x6
        cv::Mat loadings = (cv::Mat_<double>(2,6) << -0.3731, 0.2012, 0.4418,
                            0.4612, 0.4583, 0.4499, -0.5057, -0.8117, -0.2479,
                            -0.0237, 0.0789, 0.1310);
        fs << "loadings" << loadings;
    }
    
    // Loads the settings from the YML file. These will be applied to all images.
    param model;
    cv::FileStorage fs((directory+filename).c_str(), cv::FileStorage::READ);
    fs["subsample_image"] >> model.subsample_image;
    fs["start_level"] >> model.start_level;
    fs["end_level"] >> model.end_level;
    fs["sigma_xy"] >> model.sigma_xy;
    fs["percent_retained"] >> model.percent_retained;
    fs["n_components"] >> model.n_components;
    fs["display_results"] >> model.display_results;
    int n_features = -1; // this is intentional!
    for (int i=model.end_level; i>=model.start_level; i-=2)
        n_features += 1;
    model.n_features = n_features;
    
    cv::Mat mean_vector, scaling_vector, loadings; // temporary matrices
    
    // Parameters used in the principal component analysis (PCA) model.
    fs["mean_vector"] >> mean_vector;
    model.mean_vector.resize(1, n_features);
    for (int k=0; k < n_features; k++)
        model.mean_vector(k) = mean_vector.at<double>(k);
    
    fs["scaling_vector"] >> scaling_vector;
    model.scaling_vector.resize(1, n_features);
    for (int k=0; k < n_features; k++)
        model.scaling_vector(k) = scaling_vector.at<double>(k);
    
    fs["loadings"] >> loadings;
    model.loadings.resize(model.n_components, n_features);
    for (int k=0; k < model.n_components; k++) //row
        for (int j=0; j < n_features; j++) //col
            model.loadings(k, j) = loadings.at<double>(k, j);
    return model;
};


// Constructor: default type
Image::Image(){
    rows_ = cols_ = layers_ = length_ = 0;
    is_complex = false;
    filename = "";
}

Image::Image(const Image &incoming){
    // Copy constructor: creates a deepcopy
    rows_ = incoming.rows_;
    cols_ = incoming.cols_;
    layers_ = incoming.layers_;
    length_ = incoming.length_;
    is_complex = incoming.is_complex;
    filename = incoming.filename;
    src_ = new unsigned char[length_];
    for (std::size_t i=0; i<length_; i++)
        src_[i] = incoming.src_[i];
};

Image::Image(const string & image_filename){
    // Constructor: when creating an image from a filename
    // Reads a bitmap BMP image (using the bitmap_image.hpp library).
    // Reads it into our Image class.
    
    is_complex = false;
    
    bitmap_image image(image_filename);
    if (!image){
        std::cout << "Could not open " << filename.c_str() << endl;
        std::cout << "Returning a null (zero-size) image!";
        rows_ = cols_ = layers_ = length_ = 0;
        filename = "";
    }
    else{
        cols_ = image.width();
        rows_ = image.height();
        layers_ = 3;
        length_ = cols_ * rows_ * layers_;
        filename = image_filename;
        src_ = new unsigned char[length_];
        // The image data is aligned as BGR, BGR triplets
        // Starting at (0,0) top left, and going across the first row. Once
        // that is done, it moves to the next row. From top to bottom.
        // In other words: it is row-major aligned.
        unsigned char* start = image.data();
        for (std::size_t i=0; i< length_; i++){
            src_[i] = start[i];
            //cout << static_cast<int>(src_[i]) << endl;
        }
    } 
}

Image::Image(int rows, int cols, int layers, bool is_complex_image){
    // Constructor: when creating a zero'd image of a specific size
    rows_ = rows;
    cols_ = cols;
    layers_ = layers;
    is_complex = is_complex_image;
    if(is_complex){
        length_ = rows_ * cols_ * layers_ * 2;
        src_ = new unsigned char [length_];
        std::fill(src_, src_ + length_, 0x00);
    }
    else{
        length_ = rows_ * cols_ * layers_ * 1;
        src_ = new unsigned char [length_];
        std::fill(src_, src_ + length_, 0x00);
    }
}
// Destructor
Image::~Image(){
    //cout << this->filename << endl;
    delete[] src_;
}

Image read_image(std::string filename){
    // Use the Image constructor to do the work for us.
    return Image(filename);
};

Image subsample_image(Image inImg){
    // Takes every second pixel, starting at (0,0) top left, and going to
    // the edges. Images with uneven columns or rows will omit that last
    // column or row.
    int inW = inImg.width();
    int inH = inImg.height();
    int width = static_cast<int>(floor(inW / 2.0));
    int height = static_cast<int>(floor(inH / 2.0));
    int layers = inImg.layers();
    Image output(height, width, layers, false);
    output.filename = inImg.filename + "--subsampled";
    
    unsigned char * inI = inImg.start();
    unsigned char * outI = output.start();
    std::size_t idx = 0;
    
    for (std::size_t j=0; j<inH; j+=2){
        for (std::size_t i=0; i<inW; i+=2){
            for (std::size_t k=0; k < layers; k++){
                outI[idx++] = inI[j*inW*layers + i*layers + k];
                //cout << static_cast<int>(inI[j*inW*layers+i*layers+k]) << endl;
                //cout << static_cast<int>(outI[idx-1]) << endl;
            }
        }
    }
    return output;
};

Image colour2gray(Image inImg){
    // Convert a 3-channel BRG image to a grayscale image.
    int width = inImg.width();
    int height = inImg.height();
    int layers = 1;
    Image output(height, width, layers, false);
    output.filename = inImg.filename + "--grayscaled";
    unsigned char * outI = output.start();
    unsigned char * inI = inImg.start();
    
    if (inImg.layers()==3){
        // This is a common weighted sum for colour -> gray, since the human
        // eye is more attuned to green colour (higher prevalence of green cones).
        double bmul = 0.114020;
        double gmul = 0.587043;
        double rmul = 0.298936;
        std::size_t m_src = 0;
        std::size_t m_dst = 0;
        for (std::size_t j=0; j<height; j++){
            for (std::size_t i=0; i<width; i++){
                outI[m_dst++] = inI[m_src+0]*bmul +
                                inI[m_src+1]*gmul +
                                inI[m_src+2]*rmul;
                m_src += 3;
                //cout << static_cast<int>(outI[m_dst-1]) << endl;
            }
        }
    }
    return output;
};

fftw_complex* fft2_image(Image inImg){
    // Performs the forward fast Fourier transform of a 2D matrix on the input
    // image, inImg.
    // 1. Convert input image to floats, allocating the right size array
    // 2. Create the output storage to receive the FFT result. Use the FFTW
    //    storage types.
    // 3. Clean up the temporary input floating array, and the FFT plan.
    
    // 1. Note that FFTW stores the image data in row major order
    double *inFFT = new double[inImg.height()*inImg.width()];
    unsigned char *start_pixel = inImg.start();
    std::size_t idx = 0;
    for (std::size_t i = 0; i < inImg.height(); i++){ // rows
        for (std::size_t j = 0; j < inImg.width(); j++){ //cols
            inFFT[i*inImg.width()+j] = static_cast<double>(start_pixel[idx++]);
        }
    }
    
    // 2. Set up the outputs
    std::size_t halfwidth = (inImg.width() / 2) + 1;
    std::size_t fft_size = sizeof(fftw_complex)*inImg.height()*halfwidth;
    fftw_complex *outFFT = (fftw_complex*)fftw_malloc (fft_size);
    
    fftw_plan plan_forward = fftw_plan_dft_r2c_2d (inImg.height(),
                                                   inImg.width(),
                                                   inFFT, outFFT, FFTW_ESTIMATE);
    fftw_execute (plan_forward);
    //for (std::size_t i = 0; i < inImg.height(); i++){
    //    for (std::size_t j = 0; j < halfwidth; j++){
    //        cout << "\t" << i << "\t" << j << "\t" << outFFT[i*halfwidth+j][0]
    //             << "\t" << outFFT[i*halfwidth+j][1] << endl;
    //    }
    //}
    
    // 3. Clean up temporary storage
    delete[] inFFT;
    fftw_destroy_plan(plan_forward);
    
    return outFFT;
};


MatrixRM ifft2_cImage_to_matrix(fftw_complex* inImg, double scale,
                                int height, int width, param model){
    // Recreates an image from the complex inputs by using the inverse fast
    // Fourier transform.
    //
    // You must also specify the recreated image dimensions: height (rows) and
    // width (columns). Note that the inImg stores in row-major order an FFT
    // that is as many rows as the original, but roughly half the number of
    // columns. But no consistency checking is done (yet) to ensure you have
    // specified a sane ``height`` and ``width``.
    //
    // Note 1: the stored result is passed through the absolute value function,
    // since that step is required next. It can be done here efficiently.
    //
    // Note 2: this function destroys the input ``inImg``, since it is not
    //         used again after this.


    double * outIm = new double[height * width];
    double n_elements = height*width;
    fftw_plan plan_backward = fftw_plan_dft_c2r_2d(height, width,
                                                inImg, outIm, FFTW_ESTIMATE);
    fftw_execute(plan_backward);
    
    // Copy the result from FFTW to our image storage array
    MatrixRM outputI;
    outputI.resize(height, width);
    for (std::size_t i = 0; i < height; i++ ){
        for (std::size_t j  = 0; j < width; j++ ){
            *(outputI.data() + i*width+j) = std::abs(scale * outIm[i*width+j]
                                                                  / n_elements);
            //cout << "\t" << i << "\t" << j << "\t"
            //     << *(outputI.data() + i*width+j) << endl;
        }
    }

    // Clean up temporary storage.
    fftw_destroy_plan(plan_backward);
    delete[] outIm;
    
    // Clean up the input image here. It will leak memory if not freed.
    fftw_free(inImg);
    
    if (model.display_results){
        // Write the image result to a JPG file, to visualize outside this
        // function. Using ``display_results=true`` will slow processing.
        string fname = model.working_dir + "temp/texture-" + to_string((int)scale) + ".bmp";
        cv::Mat outputI_cv(height, width, CV_8UC1);
        unsigned char* outputI_ptr = outputI_cv.ptr<unsigned char>(0);
        float subtractor = outputI.minCoeff();
        float max = outputI.maxCoeff();
        // This scales all pixels in the image between 0 (black) and 255 (white)
        float scaling_factor = 255.0 / (max - subtractor);
        for (std::size_t i = 0; i < height; i++ ){
            for (std::size_t j  = 0; j < width; j++ ){
                outputI_ptr[i*width+j] = (*(outputI.data() + i*width+j) - subtractor)
                                                        / scaling_factor;
            }
        }
        // This image is colour mapped: each pixel between 0 and 255 is mapped
        // to a different color through a lookup table.
        cv::Mat color_mapped_image;
        cv::applyColorMap(outputI_cv, color_mapped_image, cv::COLORMAP_JET);
        cv::imwrite(fname, color_mapped_image);
    }
    return outputI;
};

fftw_complex* gauss_cwt(fftw_complex* inFFT, double scale, param model,
                int height, int width){
    // Performs the Gaussian Continuous Wavelet Transformation, given the FFT2
    // transform of the image, ``inImg``. It does that at the required ``scale``,
    // and at the model parameter ``sigma``.
    // Some information about the size of the original image, ``height`` and
    // ``width`` is also required to set up the iFFT2 at the end.
    
    // Create the height and width pulses
    double *h_pulse = new double[height];
    double multiplier_h = 2 * M_PI / height;
    int split = floor(height/2);
    for(int k=0; k<split; k++){
        h_pulse[k] = pow(scale * k * multiplier_h, 2);
        h_pulse[k+split] = pow(- scale * multiplier_h * (split-k), 2);
        //cout << k << "\t" << h_pulse[k] << "\t" << h_pulse[k+split] << endl;
    }
    
    double *w_pulse = new double[width];
    double multiplier_w = 2 * M_PI / width;
    split = width/2;
    for(int k=0; k<split; k++){
        w_pulse[k] = pow(scale * k * multiplier_w, 2);
        w_pulse[k+split] = pow(-scale * multiplier_w * (split-k), 2);
        //cout << k << "\t" << w_pulse[k] << "\t" << w_pulse[k+split] << endl;
    }
    
    // Starting the computations for the Gaussian. Set up storage for the
    // FFTW structure. Note (again) that the column dimension is half the
    // image width, plus 1 column padding. So when we do the convolution below,
    // we are doing it with the knowledge of the symmetry.
    std::size_t halfwidth = (width / 2) + 1;
    std::size_t fft_size = sizeof(fftw_complex)* height * halfwidth;
    fftw_complex *outFFT = (fftw_complex*)fftw_malloc (fft_size);

    double neg_sigma_sq = -0.5*pow(model.sigma_xy, 2.0);
    double multiplier = 0.0;
    std::size_t idx = 0;
    for(std::size_t k=0; k < height; k++){
        for(std::size_t j=0; j < halfwidth; j++){
            idx = k*halfwidth + j;
            multiplier = exp( neg_sigma_sq * (w_pulse[j] + h_pulse[k]) );
            outFFT[idx][0] = inFFT[idx][0] * multiplier; // real
            outFFT[idx][1] = inFFT[idx][1] * multiplier; // complex
            //cout << k << "\t" << idx  << "\t" << outFFT[idx][0] << "\t"
            //     << outFFT[idx][1] << endl;
        }
    }

    // Clean up and return
    delete[] h_pulse;
    delete[] w_pulse;
    return outFFT;
};

double norm_threshold(const float *X, long n_elements, int apply_thresh, double thresh=0.0){
    // Using the ``n_elements`` in contiguous matrix (or vector) ``X``, it
    // calculates the sum-of-squares of the elements. Or, if ``apply_thresh``
    // ``thresh`` are provided, it will only do this on elements in X (not X^2)
    // that exceed that given ``thresh`` value.
    //
    // MATLAB: normvalue = sum(sum( ((A>=thresh).*A).^2 ));  % for matrix A

    double temp = 0.0;
    if (apply_thresh==1){
       for (std::size_t k=0; k < n_elements; k++)
        temp += X[k]*X[k]*(X[k]>=thresh);
    }
    else{
        for (std::size_t k=0; k < n_elements; k++)
        temp += X[k]*X[k];
    }
    return(temp);
}

Eigen::VectorXf threshold(const MatrixRM inImg, param model){
    // Thresholds the wavelet coefficients based on retained energ. Calculates
    // the percentage of pixels that exceeds this energy level.
    //
    // Returns: 2-element vector
    //   1/ The percentage retained coefficients.
    //   2/ The threshold value computed to retain an energy level.

    
    // 1. Initialize parameters required to determine the thresholding value
    double per_retained = model.percent_retained;
    long n_elements = inImg.rows() * inImg.cols();
    const float* X = inImg.data();
    float stdX = 0.0;
    float sumX = 0.0;
    float minX = X[0];
    float maxX = 0.0;
    float meanX = 0.0;
    
    // 2. We need some basic statistics about the image. It is no more efficient
    //    to calculate them outside the matrix library, than to calculate them
    //    manually. Rather use the library (cleaner code).
    minX = inImg.minCoeff();
    maxX = inImg.maxCoeff();
    sumX = inImg.sum();
    meanX = sumX / n_elements;
    for (std::size_t k=0; k < n_elements; k++){
        stdX += (X[k] - meanX) * (X[k] - meanX);
    }
    stdX = sqrt(stdX / (n_elements - 1));
    double base_energy = norm_threshold(inImg.data(), n_elements, 0);
    
    // 3. Set up search algorithm to find energy level. Uses the Golden section
    //    search routine (https://en.wikipedia.org/wiki/Golden_section_search)
    double R = (sqrt(5)-1)/2;  // Golden ratio: 0.61803399
    double C = 1 - R;
    double x0, x1, x2, x3, f1, f2;
    x0 = minX;
    x3 = maxX;
    x1 = x0 + C*(x3-x0);
    x2 = x3 - C*(x3-x0);
    
    f1 = std::abs(norm_threshold(X, n_elements,1, x1)/base_energy - per_retained);
    f2 = std::abs(norm_threshold(X, n_elements,1, x2)/base_energy - per_retained);
    
    // 4. Run the search algorithm in a while loop, protecting for the case of
    //    non-convergence. Allows a maximum of 100 iterations. Most images
    //    use around 30 to 38 iterations.
    int n_iter = 0;
    int max_iterations = 100;
    double tolerance = 0.000001;
    while ((std::abs(x3-x0)/stdX > tolerance) && (n_iter < max_iterations)){
        n_iter++;
        if (f2 < f1){
            x0 = x1;
            x1 = x2;
            x2 = R*x1 + C*x3;
            f1 = f2;
            f2 = std::abs(norm_threshold(X, n_elements, 1, x2)/base_energy - per_retained);
        }
        else{
            x3 = x2;
            x2 = x1;
            x1 = R*x2 + C*x0;
            f2 = f1;
            f1 = std::abs(norm_threshold(X ,n_elements, 1, x1)/base_energy - per_retained);
        }
    }
    if (n_iter >= max_iterations){
        cout << "Maximum number of while loop iterations exceeded. " << endl;
        abort();
    }else{
        //cout << "Required " << n_iter << " iterations." << endl;
    }
    
    // Finished. Assign the outputs and return.
    Eigen::VectorXf output(2);
    if (f1 < f2)
        output(1) = x1;
    else
        output(1) = x2;
    
    // MATLAB: = sum(sum(A >= ThrValue))
    output[0] = 0.0;
    for (std::size_t k=0; k < n_elements; k++)
        output(0) += static_cast<double>(X[k] > output(1));
    output(0) = output(0) / static_cast<double>(n_elements);
    return output;
};

Eigen::VectorXf project_onto_model(const Eigen::VectorXf& features, param model){
    // Projects (applies) the calculated features from the image onto a pre-
    // existing PCA model. That PCA model was built from the features extracted
    // on the training image data.
    
    // In other words, this function is seeing how similar/dissimilar the
    // current image is in comparison with the training data.
    
    // For details on the function signature:
    // http://eigen.tuxfamily.org/dox/group__TopicPassingByValue.html
    
    // First difference the features. (That explains why we count features
    // starting at -1 in ``param load_model_parameters(...)``.
    VectorRM pca_features, mcuv_features, pca_scores, spe_vector;
    pca_features.resize(1, model.n_features);
    mcuv_features.resize(1, model.n_features);
    
    // Then mean center and unit-variance (mcuv) scale after calculating
    // the features.
    for (int k=0; k < model.n_features; k++){
        pca_features(k) = features(k+1) - features(k);
        mcuv_features(k) = (pca_features(k) - model.mean_vector(k)) /
                                                     model.scaling_vector(k);
    }
    
    // Calculate the PCA model outputs: the scores, and the SPE vector
    pca_scores.resize(1, model.n_components);
    pca_scores = mcuv_features * model.loadings.transpose();
    
    spe_vector.resize(1, model.n_features);
    spe_vector = pca_scores * model.loadings;
    double spe_value = norm_threshold(spe_vector.data(), model.n_features, 0);

    // Place the PCA scores and the SPE value in the return vector.
    // And we are finished!
    Eigen::VectorXf output(model.n_components+1);
    for (int k=0; k < model.n_components; k++)
        output(k) = pca_scores(k);
    output(model.n_components) = static_cast<float>(spe_value);
    return output;
};


// Code to display the image results in an OpenCV window
// Code is directly from:
// http://stackoverflow.com/questions/5089927/show-multiple-2-3-4-images-in-the-same-window-in-opencv
/**
 * @brief makeCanvas Makes composite image from the given images
 * @param vecMat Vector of Images.
 * @param windowHeight The height of the new composite image to be formed.
 * @param nRows Number of rows of images. (Number of columns will be calculated
 *              depending on the value of total number of images).
 * @return new composite image.
 */
cv::Mat makeCanvas(std::vector<cv::Mat>& vecMat, int windowHeight, int nRows) {
    int N = static_cast<int>(vecMat.size());
    nRows  = nRows > N ? N : nRows;
    int edgeThickness = 10;
    int imagesPerRow = ceil(double(N) / nRows);
    int resizeHeight = floor(2.0 * ((floor(double(windowHeight - edgeThickness) / nRows)) / 2.0)) - edgeThickness;
    int maxRowLength = 0;
    
    std::vector<int> resizeWidth;
    for (int i = 0; i < N;) {
        int thisRowLen = 0;
        for (int k = 0; k < imagesPerRow; k++) {
            double aspectRatio = double(vecMat[i].cols) / vecMat[i].rows;
            int temp = int( ceil(resizeHeight * aspectRatio));
            resizeWidth.push_back(temp);
            thisRowLen += temp;
            if (++i == N) break;
        }
        if ((thisRowLen + edgeThickness * (imagesPerRow + 1)) > maxRowLength) {
            maxRowLength = thisRowLen + edgeThickness * (imagesPerRow + 1);
        }
    }
    int windowWidth = maxRowLength;
    cv::Mat canvasImage(windowHeight, windowWidth, CV_8UC3, cv::Scalar(0, 0, 0));
    
    for (int k = 0, i = 0; i < nRows; i++) {
        int y = i * resizeHeight + (i + 1) * edgeThickness;
        int x_end = edgeThickness;
        for (int j = 0; j < imagesPerRow && k < N; k++, j++) {
            int x = x_end;
            cv::Rect roi(x, y, resizeWidth[k], resizeHeight);
            cv::Size s = canvasImage(roi).size();
            // change the number of channels to three
            cv::Mat target_ROI(s, CV_8UC3);
            if (vecMat[k].channels() != canvasImage.channels()) {
                if (vecMat[k].channels() == 1) {
                    cv::cvtColor(vecMat[k], target_ROI, CV_GRAY2BGR);
                }
            }
            cv::resize(target_ROI, target_ROI, s);
            if (target_ROI.type() != canvasImage.type()) {
                target_ROI.convertTo(target_ROI, canvasImage.type());
            }
            target_ROI.copyTo(canvasImage(roi));
            x_end += resizeWidth[k] + edgeThickness;
        }
    }
    return canvasImage;
}