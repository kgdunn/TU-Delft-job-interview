
#include <cstdio>
#include <iostream>
#include <memory>
#include <algorithm>
#include <vector>
#include "flotation.h"
#include "bitmap_image.hpp"

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

// Constructor: when creating it from a filename
Image::Image(const string & image_filename){
    // Reads a bitmap BMP image (using the bitmap_image.hpp library).
    // Reads into our Image class.
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
        src = new unsigned char[length_];
        // The image data is aligned as BGR, BGR triplets
        // Starting at (0,0) top left, and going down columns first,
        // then across the image, from left to right.
        unsigned char* start = image.data();
        for (int i=0; i< length_; i++){
            src[i] = start[i];            
        }
    } 
}
// Constructor: when creating a zero'd image of a specific size
Image::Image(int rows, int cols, int layers, bool is_complex_image)
{
    rows_ = rows;
    cols_ = cols;
    layers_ = layers;
    is_complex = is_complex_image;
    if(is_complex){
        length_ = rows_ * cols_ * layers_ * 2;
        src = new unsigned char [length_];
        std::fill(src, src + length_, 0.0);
    }
    else{
        length_ = rows_ * cols_ * layers_ * 1;
        src = new unsigned char [length_];
        std::fill(src, src + length_, 0.0);
    }
}
// Destructor
Image::~Image(){
    delete[] src;
}

Image read_image(std::string filename){
    // Use the Image constructor to do the work for us.
    return Image(filename);
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