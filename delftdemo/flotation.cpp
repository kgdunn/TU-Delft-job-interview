#include <iostream>
#include <memory>
#include <algorithm>
#include <vector>
#include "flotation.h"

// Constructor: default type
Image::Image(){
    rows_ = cols_ = layers_ = 0;
    is_complex = false;
    filename = "";
}

// Constructor: when creating from a filename
Image::Image(const string & filename){
    // Store filename
    // Read image into src
    // Set rows, cols, layers
    is_complex = false;
}
// Constructor: when creating a zero'd image of a specific size
Image::Image(int rows, int cols, int layers, bool is_complex_image)
{
    rows_ = rows;
    cols_ = cols;
    layers_ = layers;
    is_complex = is_complex_image;
    if(is_complex){
        src = new float [rows_ * cols_ * layers_ * 2];
        std::fill(src, src + rows_*cols_*layers_*2, 0.0);
    }
    else{
        src = new float [rows_ * cols_ * layers * 1];
        std::fill(src, src + rows_*cols_*layers_*1, 0.0);
    }
}
// Destructor
Image::~Image(){
    delete[] src;
}


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


Image read_image(std::string filename){
    Image output;
    //FILE *f = fopen(filename.c_str(), "r");
    return output;
};

Image subsample_image(Image inImg){
    Image output;
    return output;
};

Image colour2gray(Image inImg){
    Image output;
    return output;
};

Image fft2_image(Image inImg){
    Image output;
    return output;
};

Image iff2_image(Image inImg){
    Image output;
    return output;
};

Image gauss_cwt(Image inImg, param model){
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