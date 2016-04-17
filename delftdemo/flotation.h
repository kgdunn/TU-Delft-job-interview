#ifndef flotation_h
#define flotation_h

#include <iostream>
#include <string>
using namespace std;

#include <fftw3.h>

// Use Eigen for matrix computations, and typedef alias a shorter name.
#include "Eigen/Core"
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixRM;

struct param{};

class Image
{
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
    Image(const string & filename);         // Copy constructor
    //Image& operator=(const Image& input);   // Assignment operator
    ~Image();
  
    // Public member functions
    inline int width() { return cols_; }
    inline int height() { return rows_; }
    inline int layers() { return layers_; }
    inline int length() { return length_; }
    inline unsigned char* start() {return src_;}
};

// Function prototypes, roughly in the order they are used
param load_model_parameters(std::string directory, std::string filename);
Image read_image(std::string filename);
Image subsample_image(Image inImg);
Image colour2gray(Image inImg);
fftw_complex* fft2_image(Image inImg);
MatrixRM ifft2_cImage_to_matrix(fftw_complex* inImg, int height, int width);
fftw_complex* gauss_cwt(fftw_complex* inFFT, double scale, double sigma, int height, int width);
Image multiply_scalar(Image inImg, double scalar);
std::vector<double> threshold(Image inImg, param model);
std::vector<double> project_onto_model(std::vector<double> features, param model);

#endif /* flotation_h */
