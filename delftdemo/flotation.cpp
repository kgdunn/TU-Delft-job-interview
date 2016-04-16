
#include <cstdio>
#include <iostream>
#include <memory>
#include <algorithm>
#include <vector>
#include <cmath>

// 3rd party libraries. You     make install (installs the library to /usr/local/include)
#include <fftw3.h> // make install this, and add the appropriate locations to
                   // to both the Header and the Library search paths.
                   // Linker flags required: "-lfftw3 -lm"
#include "bitmap_image.hpp"

// Our libraries
#include "flotation.h"

param load_model_parameters(std::string directory,
                            std::string filename){
    
    param output;
    //FILE *f = fopen(filename.c_str(), "r");
    //    std::vector<BeadPos> beads;
    //
    //    if (!f) return beads;
    //
    //    while (!feof(f)) {
    //        BeadPos bp;
    //        fscanf(f, "%d\t%d\n", &bp.x,&bp.y);
    //        beads.push_back(bp);
    //    }
    //
    //    fclose(f);
    //    return beads;
    return output;
};


// Constructor: default type
Image::Image(){
    rows_ = cols_ = layers_ = length_ = 0;
    is_complex = false;
    filename = "";
}

Image::Image(const Image & incoming){
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

//Image::Image& operator=(const Image& input){
//    // Assignment operator
//    if (this != &input)
//    {
//        file_name_       = input.filename_;
//        width_           = input.width_;
//        height_          = input.height_;
//        std::copy(input.src_, input.src_ + input.length_, data_);
//    }
//
//    return *this;
//}


Image::Image(const string & image_filename){
    // Constructor: when creating it from a filename
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
        // that is down, it moves to the next row. From top to bottom.
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
    cout << this->width() << endl;
    cout << this->height() << endl;
    cout << this->filename << endl;
    delete[] src_;
}

Image read_image(std::string filename){
    // Use the Image constructor to do the work for us.
    return Image(filename);
};

Image subsample_image(Image inImg){
    // Takes every second pixel, starting at (0,0) top left, and going to
    // the edges. Images with uneven columns or rows will omit that last
    // column or row
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
                outI[idx++] = inI[j*inW*layers+i*layers+k];
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
        std::size_t m_src = 0x00;
        std::size_t m_dst = 0x00;
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
    // N. Clean up the temporary input floating array, and the FFT plan.
    
    // 1. Note that FFTW stores the image data in row major order
    double *inFFT = new double[inImg.height()*inImg.width()];
    unsigned char * start_pixel = inImg.start();
    std::size_t idx = 0x00;
    for (std::size_t i = 0; i < inImg.height(); i++){ // rows
        for (std::size_t    j = 0; j < inImg.width(); j++){ //col
            inFFT[i*inImg.width()+j] = static_cast<double>(start_pixel[idx++]);
            //cout << inFFT[i*inImg.width()+j] << endl;
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
    
    // N. Clean up temporary storage
    delete[] inFFT;
    fftw_destroy_plan(plan_forward);
    
    //Image output;
    return outFFT;
};

Image ifft2_complex_image(fftw_complex* inImg, int height, int width, bool keep_input){
    // Recreates an image from the complex inputs by using the inverse fast
    // Fourier transform.
    //
    // You must also specify the recreated image dimensions: height (rows) and
    // width (columns). Note that the inImg stores in row-major order an FFT
    // that is as many rows as the original, but roughly half the number of
    // columns. But no consistency checking is done (yet) to ensure you have
    // specified a sane ``height`` and ``width``.
    
    // It has the side effect of destroying "inImg", unless you request not to
    // with the ``keep_input`` flag.

    double * outIm = new double[height * width];
    fftw_plan plan_backward = fftw_plan_dft_c2r_2d(height, width,
                                                inImg, outIm, FFTW_ESTIMATE);
    fftw_execute(plan_backward);
    
    // Copy the result from FFTW to our image storage array
    Image output(height, width, 1, false);
    //std::size_t idx = 0x00;
    
    // TODO: copy the data over, in float form to a different array structure.
    for (std::size_t i = 0; i < height; i++ ){
        for (std::size_t j = 0; j < width; j++ ){
            //cout << "\t" << i << "\t" << j << "\t"
            //     << outIm[i*width+j] / (double) (height*width) << endl;
        }
    }
    
    // Clean up temporary storage.
    fftw_destroy_plan(plan_backward);
    delete[] outIm;
    if (!keep_input){
        fftw_free(inImg);
    }
    return output;
};



Image gauss_cwt(fftw_complex* inFFT, double scale, double sigma,
                int height, int width){
    // Performs the Gaussian Continuous Wavelet Transformation, given the FFT2
    // transform of the image, ``inImg``
    
    // 1. Get pointers to the required parts of the complex input image. R x C
    //Areal = mxGetPr(prhs[1]);
    //Aimag = mxGetPi(prhs[1]);
    
    // 3. Set up variables required for the calculations
    //int n_elements = height * width;

    // 2. Set up the output image
    double *B = new double [height * width];
    
    // 3. Intermediate storage required
    //double *temp_real = new double[height * width];
    //double *temp_imag = new double[height * width];
    //tempOutPtr  = mxCreateDoubleMatrix(nRows,nCols,mxCOMPLEX);                 // used for intermediate calculations
    
    // Create the height and width pulses
    double *h_pulse = new double[height];
    double multiplier_h = 2*3.141592653589/height;
    int split = static_cast<int>(floor(height/2));
    for(int k=0; k<split; k++){
        h_pulse[k] = pow(scale * k * multiplier_h, 2);
        h_pulse[k+split] = pow(- scale * multiplier_h * (split-k), 2);
        //cout << k << "\t" << h_pulse[k] << "\t" << h_pulse[k+split] << endl;
    }
    
    double *w_pulse = new double[width];
    double multiplier_w = 2*3.141592653589/width;
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

    double neg_sigma_sq = -1*pow(sigma, 2.0) / 2.0	;
    double multiplier = 0.0;

    for(std::size_t k=0; k< height; k++){
        for(std::size_t j=0;j< halfwidth; j++){
            multiplier = exp( neg_sigma_sq * (w_pulse[j] + h_pulse[k]) );
            //tempReal[k+j*nRows] = Areal[k+j*nRows] * multiplier;  // multiply by A while we are here (we need to do it
            //tempImag[k+j*nRows] = Aimag[k+j*nRows] * multiplier;    //   in the next step anyway)
        }
    }
    
    delete[] B;
    delete[] h_pulse;
    delete[] w_pulse;
  
    Image output;
    return output;
};

Image multiply_scalar(Image inImg, double scalar){
    // Utility function: multiply each entry in the image by a scalar value.
    Image output;
    return output;
};

std::vector<double> threshold(Image inImg, param model){
    std::vector<double> output(2);
    return output;
};

std::vector<double> project_onto_model(std::vector<double> features, param model){
    std::vector<double> output(3);
    return output;
};