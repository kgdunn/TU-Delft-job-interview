#include <iostream>
#include <string>
#include <vector>
#include "flotation.h"

using namespace std;

int main() {
    
    string directory = "/Users/kevindunn/Delft/demo/gui/source/";
    string parameters_file = "model-parameters.xml";
    
    param coefficients = load_model_parameters(directory, parameters_file);
    
    //for-loop here
    string dummy_filename = "/Users/kevindunn/Delft/DelftDemo/delftdemo/delftdemo/testing-image.bmp";
    
    
    // The image processing pipeline:
    Image raw_image_nD  = read_image(dummy_filename);
    Image image_nD_sub  = subsample_image(raw_image_nD);
    Image image_1D      = colour2gray(image_nD_sub);
    Image image_complex = fft2_image(image_1D);
    Image wavelet_image = gauss_cwt(image_complex, coefficients);
    vector<double> f1f2 = threshold(wavelet_image, coefficients);
    //features <- f1f2
    vector<double> features; //diff(features)
    vector<double> calc_outputs = project_onto_model(features, coefficients);

    
    return 0;
}

