#ifndef flotation_h
#define flotation_h

#include <iostream>
#include <string>
using namespace std;

// 3rd party libraries. Please see README.txt to see how to install and set up.
#include <fftw3.h>
// Use Eigen for matrix computations, and typedef aliased shorter names.
#include "Eigen/Core"
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixRM;
typedef Eigen::Matrix<float, 1, Eigen::Dynamic> VectorRM;


struct param{
    // A structure that holds the model parameters that are used to process
    // each image. The structure is considered "read-only", i.e. no results
    // are stored in it.
    
    int subsample_image;    // cuts image size by a factor of 2(rows)*2(cols)=4
    float start_level;      // start and end levels for the
    float end_level;        // wavelet resolution
    float sigma_xy;         // Gaussian coefficient
    float percent_retained; // percentage energy retained
    bool display_results;   // writes results to file, for later display
    string working_dir;     // where images should be written to
    
    // Parameters used in the principal component analysis (PCA) model.
    int n_features;         // Number of features used in the PCA projection
    VectorRM mean_vector;   // 1 x n_features vector
    VectorRM scaling_vector;// 1 x n_features vector
    int n_components;
    MatrixRM loadings;      // n_features x n_components matrix
};


class Image{
private:
    int rows_, cols_, layers_, length_;
    unsigned char *src_;
public:
    string filename;
    bool is_complex;
    
    // Three different constructor options
    Image();
    Image(int rows, int cols, int layers, bool is_complex = false);
    Image(const Image &);
    Image(const string & filename);  // Copy constructor
    //Image& operator=(const Image& input);   // Assignment operator
    ~Image();
  
    // Public member functions
    inline int width() { return cols_; }
    inline int height() { return rows_; }
    inline int layers() { return layers_; }
    inline int length() { return length_; }
    inline unsigned char* start() {return src_;}
};

// Function prototypes, roughly in the order they are used:
param load_model_parameters(string directory, string filename);
Image read_image(string filename);
Image subsample_image(Image inImg);
Image colour2gray(Image inImg);
fftw_complex* fft2_image(Image inImg);
MatrixRM ifft2_cImage_to_matrix(fftw_complex* inImg, double scale,
                                int height, int width, param model);
fftw_complex* gauss_cwt(fftw_complex* inFFT, double scale, double sigma,
                        int height, int width);
Eigen::VectorXf threshold(MatrixRM inImg, param model);
Eigen::VectorXf project_onto_model(const Eigen::VectorXf& features, param model);

// Note: this function is not my own work. It is documented in flotation.cpp.
cv::Mat makeCanvas(vector<cv::Mat>& vecMat, int windowHeight, int nRows);
#endif /* flotation_h */
