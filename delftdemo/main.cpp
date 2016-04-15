#include <iostream>
#include <string>
#include <vector>
#include "flotation.h"

using namespace std;

int main() {
    
    string directory = "/Users/kevindunn/Delft/demo/gui/source/";
    string parameters_file = "model-parameters.xml";
    
    Image raw_image_nD;
    Image image_nD_sub;
    Image image_1D;
    Image image_complex;
    Image wavelet_image;
    vector<double> f1f2;
    vector<double> features;
    vector<double> calc_outputs;
    
    param coefficients = load_model_parameters(directory, parameters_file);
    
    //for-loop here
    string dummy_filename = "/Users/kevindunn/Delft/DelftDemo/delftdemo/delftdemo/testing-image.bmp";
    
    // The start of the image processing pipeline:
    raw_image_nD = read_image(dummy_filename);
    image_nD_sub = subsample_image(raw_image_nD);
    image_1D = colour2gray(image_nD_sub);
    image_complex = fft2_image(image_1D);
    wavelet_image = gauss_cwt(image_complex, coefficients);
    f1f2  = threshold(wavelet_image, coefficients);
    //features <- f1f2
    //diff(features)
    calc_outputs = project_onto_model(features, coefficients);

    
    return 0;
}

