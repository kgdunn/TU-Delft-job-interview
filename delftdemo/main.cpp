#include <iostream>
#include <string>
#include <vector>
#include "flotation.h"
#include <fftw3.h>
#include "Eigen/Core"

using namespace std;

int main() {
    
    int n_profiles = 1000;
    auto begin = std::chrono::high_resolution_clock::now();
    
    for (int k=0; k < n_profiles; k++){
        cout << k << endl;
        string directory = "/Users/kevindunn/Delft/demo/gui/source/";
        string parameters_file = "model-parameters.xml";
        
        param coefficients = load_model_parameters(directory, parameters_file);
        
        //for-loop here
        string dummy_filename = "/Users/kevindunn/Delft/DelftDemo/delftdemo/delftdemo/testing-image.bmp";
        
        
        // The image processing pipeline:
        Image raw_image_nD  = read_image(dummy_filename);
        Image image_nD_sub  = subsample_image(raw_image_nD);
        Image image_1D      = colour2gray(image_nD_sub);
        fftw_complex *image_complex = fft2_image(image_1D);
        
        fftw_complex *wavelet_image = gauss_cwt(image_complex, 1, 1,
                                                image_1D.height(), image_1D.width());
        MatrixRM restored = ifft2_cImage_to_matrix(wavelet_image,
                                             image_1D.height(),
                                             image_1D.width());
        vector<double> f1f2 = threshold(restored, coefficients);
        //features <- f1f2
        //vector<double> features; //diff(features)
        //vector<double> calc_outputs = project_onto_model(features, coefficients);
        
        // Cleanup memory
        fftw_free(image_complex);
        fftw_free(wavelet_image);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count()/1000000.0/n_profiles << "ms per iteration" << std::endl;

    
    return 0;
}

