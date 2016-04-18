// This code processes images from a flotation cell
// https://en.wikipedia.org/wiki/Froth_flotation
// and extracts textures from the image. These texture features are projected
// onto a multivariate (principal component analysis, PCA) model, to determine
// how close the appearance of the image is to the "desirable" appearance.
//
// People that operate the flotation cells can therefore use this model to
// adjust settings on the process, so that the process moves to a region of
// stable operation. The outputs can (and have) been used for automatic control
// of image appearance.
//
// All code (except for external libraries), have been written by Kevin Dunn
// 14 to 18 April 2016, as demonstration for a job interview at TU Delft.


#include <iostream>
#include <string>
#include <vector>

// 3rd party libraries. Please see README.txt to see how to install and set up.
#include <fftw3.h>
#include "bitmap_image.hpp"
#include "Eigen/Core"
//#include "opencv2/core/persistence.hpp"
#include <boost/filesystem.hpp>
//#include <boost/circular_buffer.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include "GraphUtils.h"

// Our libraries
#include "flotation.h"

using namespace std;
using namespace boost::filesystem;

int main() {
    
    int n_profiles = 1;
    auto begin = chrono::high_resolution_clock::now();
    vector<string> all_files;
    
    for (int k=0; k < n_profiles; k++){
        // This outer loop is used for profiling the code and checking for
        // egregious memory leaks. None noticed when profiling for 1000's
        // of iterations.
        cout << k << "\t";
        
        string directory = "/Users/kevindunn/Delft/DelftDemo/delftdemo/working-directory/";
        string parameters_file = "model-parameters.yml";
        param model = load_model_parameters(directory, parameters_file);
        model.working_dir = directory;
        string filename_extenion_filter = ".bmp";        
        try{
            if (exists(directory) && is_directory(directory)){
                for (auto&& x : directory_iterator(directory))
                    if (x.path().extension() == filename_extenion_filter)
                        all_files.push_back(x.path().filename().string());
            sort(all_files.begin(), all_files.end());
            }
        }
        catch(const filesystem_error& ex){
            cout << ex.what() << '\n';
        }
        
        //string filename = "/Users/kevindunn/Delft/DelftDemo/delftdemo/delftdemo/testing-image.bmp";
        for (auto filename : all_files){
            // Process each new flotation image in the pipeline below.
            // Each function is mostly modular, and can be replaced with an
            // alternative, to suit the researcher's preference.
            
            // The image processing pipeline:
            Image raw_image_nD  = read_image((directory+filename).c_str());
            Image image_nD_sub  = subsample_image(raw_image_nD);
            Image image_1D      = colour2gray(image_nD_sub);
            fftw_complex *image_complex = fft2_image(image_1D);
            fftw_complex *wavelet_image;
            MatrixRM restored;
            Eigen::VectorXf f1f2;
            Eigen::VectorXf features(model.n_features+1);
            int index = 0;
            for (double scale=model.start_level; scale <= model.end_level;
                                                                    scale+=2){
                wavelet_image = gauss_cwt(image_complex, scale, 1,
                                          image_1D.height(), image_1D.width());
                restored = ifft2_cImage_to_matrix(wavelet_image, scale,
                                                  image_1D.height(),
                                                  image_1D.width(), model);
                f1f2 = threshold(restored, model);
                features(index++) = f1f2[0];
            }
            Eigen::VectorXf calc_outputs = project_onto_model(features, model);
            cout << calc_outputs.transpose() << endl;
            
            // Cleanup memory
            fftw_free(image_complex);
        }
        
    }// k=0; k<n_profiles; profiling loop
    
    auto end = chrono::high_resolution_clock::now();
    cout << chrono::duration_cast<chrono::nanoseconds>(end-begin).count()
             /1000000.0/n_profiles/(all_files.size()) << "ms per image" << endl;
    return 0;
}

