#include <iostream>
#include <string>
#include <vector>

// 3rd party libraries. You     make install (installs the library to /usr/local/include)
#include <fftw3.h> // make install this, and add the appropriate locations to
// to both the Header and the Library search paths.
// Linker flags required: "-lfftw3 -lm"
#include "bitmap_image.hpp"
#include "Eigen/Core"
#include "opencv2/core/persistence.hpp"
#include <boost/filesystem.hpp>
//#include <boost/regex.hpp>

// Our libraries
#include "flotation.h"

using namespace std;
using namespace boost::filesystem;


int main() {
    
    int n_profiles = 1;
    auto begin = std::chrono::high_resolution_clock::now();
    
    for (int k=0; k < n_profiles; k++){
        // This outer loop is used for profiling the code and checking for
        // egregious memory leaks.
        cout << k << "\t";
        
        string directory = "/Users/kevindunn/Delft/DelftDemo/delftdemo/working-directory/";
        string parameters_file = "model-parameters.yml";
        param model = load_model_parameters(directory, parameters_file);
        
        std::vector<std::string> all_files;
        string filename_extenion_filter = ".bmp";
        try{
            if (exists(directory) && is_directory(directory)){
                for (auto&& x : directory_iterator(directory))
                    if (x.path().extension() == filename_extenion_filter)
                        all_files.push_back(x.path().filename().string());
            std::sort(all_files.begin(), all_files.end());
            }
        }
        catch(const filesystem_error& ex){
            cout << ex.what() << '\n';
        }
        
        //string filename = "/Users/kevindunn/Delft/DelftDemo/delftdemo/delftdemo/testing-image.bmp";
        for (auto filename : all_files){
            // The image processing pipeline:
            Image raw_image_nD  = read_image(filename);
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
                                                  image_1D.width());
                f1f2 = threshold(restored, model);
                features(index++) = f1f2[0];
            }
            Eigen::VectorXf calc_outputs = project_onto_model(features, model);
            cout << calc_outputs.transpose() << endl;
            // Cleanup memory
            fftw_free(image_complex);
        }
        
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count()/1000000.0/n_profiles << "ms per iteration" << std::endl;

    
    return 0;
}

