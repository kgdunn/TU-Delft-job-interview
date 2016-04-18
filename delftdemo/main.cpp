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
#include <fstream>
#include <string>
#include <vector>

// 3rd party libraries. Please see README.txt to see how to install and set up.
#include <fftw3.h>
#include "bitmap_image.hpp"
#include "Eigen/Core"
#include <boost/filesystem.hpp>
#include "opencv2/opencv.hpp"

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
        int img_index = 0;
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
                wavelet_image = gauss_cwt(image_complex, scale, model,
                                          image_1D.height(), image_1D.width());
                restored = ifft2_cImage_to_matrix(wavelet_image, scale,
                                                  image_1D.height(),
                                                  image_1D.width(), model);
                f1f2 = threshold(restored, model);
                features(index++) = f1f2[0];
            }
            Eigen::VectorXf calc_outputs = project_onto_model(features, model);
            
            // MATLAB: bubble_size = sqrt(sum(features)/sum(features./(1:6).^2))
            VectorRM bub_size_num, bub_size_den;
            bub_size_num.resize(1, model.n_features);
            bub_size_den.resize(1, model.n_features);
            for (int k=0; k < model.n_features; k++){
                bub_size_num(k) = (features(k+1) - features(k));
                bub_size_den(k) = (features(k+1) - features(k)) / pow(k+1, 2);
            }
            double bubble_size = bub_size_num.sum()/bub_size_den.sum();
            
            // Display inside the software
            cout << img_index << "\t" << calc_outputs.transpose()
                 << "\t" << bubble_size << endl;
            img_index++;
            
            // Cleanup memory
            fftw_free(image_complex);
            
            if(model.display_results){
                // Write the results to a CSV file. This will be displayed in MATLAB
                std::ofstream computed_results;
                computed_results.open ((directory + "temp/features.csv").c_str());
                computed_results << calc_outputs(0) << "," << calc_outputs(1)
                                 << "," << calc_outputs(2) << ","
                                 << bubble_size << endl;
                computed_results.close();
                
                // Write the raw image result to a JPG file,.
                string fname = model.working_dir + "temp/raw-image.bmp";
                int height = raw_image_nD.height();
                int width = raw_image_nD.width();
                cv::Mat outputI_cv(height, raw_image_nD.width(), CV_8UC3);
                unsigned char* outputI_ptr = outputI_cv.ptr<unsigned char>(0);
                unsigned char* start_ptr = raw_image_nD.start();
                std::size_t idx = 0;
                for (std::size_t i = 0; i < height; i++ ){
                    for (std::size_t j  = 0; j < width; j++ ){
                        outputI_ptr[i*width*3+j*3+0] = start_ptr[idx++];
                        outputI_ptr[i*width*3+j*3+1] = start_ptr[idx++];
                        outputI_ptr[i*width*3+j*3+2] = start_ptr[idx++];
                    }
                }
                cv::imwrite(fname, outputI_cv);
                
            }
        }
        
    }// k=0; k<n_profiles; profiling loop
    
    auto end = chrono::high_resolution_clock::now();
    cout << chrono::duration_cast<chrono::nanoseconds>(end-begin).count()
             /1000000.0/n_profiles/(all_files.size()) << "ms per image" << endl;
    return 0;
}

