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
#include "opencv2/core/persistence.hpp"
#include <boost/filesystem.hpp>
#include <boost/circular_buffer.hpp>
#include <opencv2/highgui/highgui.hpp>

// Our libraries
#include "flotation.h"

using namespace std;
using namespace boost::filesystem;

int main() {
    
    int n_profiles = 5;
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
        
        // We wish to keep the last 10 results in a buffer (to display)
        int show_results = 10;
        boost::circular_buffer<Eigen::VectorXf> buffer(show_results);
        
        
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
            Eigen::VectorXf features(7);
            int index = 0;
            for (double scale=1; scale <= 13; scale+=2){
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
            
            // Store the results in a circular buffer for display:
            buffer.push_back(calc_outputs);            
            
            if (model.display_results){
                vector<cv::Mat> results(1);
                cv::Mat raw_data = cv::imread((directory+filename).c_str());
                results[0] = raw_data;
                cv::Mat image_canvas_results = makeCanvas(results, 500, 2);
                cv::namedWindow(filename, cv::WINDOW_AUTOSIZE);
                cv::imshow(filename, raw_data);
                cv::waitKey(0);
                cv::destroyWindow(filename);
            }
            
            // Cleanup memory
            fftw_free(image_complex);
         
            
        }
        
    }// k=0; k<n_profiles; profiling loop
    
    auto end = chrono::high_resolution_clock::now();
    cout << chrono::duration_cast<chrono::nanoseconds>(end-begin).count()
             /1000000.0/n_profiles/(all_files.size()) << "ms per image" << endl;
    return 0;
}

