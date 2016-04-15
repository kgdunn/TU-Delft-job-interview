
#include <cstdio>
#include <iostream>
#include <memory>
#include <algorithm>
#include <vector>
#include <cmath>
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

Image::Image(const Image & incoming){
    // Copy constructor: creates a deepcopy
    rows_ = incoming.cols_;
    cols_ = incoming.rows_;
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
    
    unsigned char * inI = inImg.start();
    unsigned char * outI = output.start();
    int idx = 0;
    
    for (std::size_t i=0; i<inW; i+=2){
        for (std::size_t j=0; j<inH; j+=2){
            for (std::size_t k=0; k < layers; k++){
                outI[idx++] = inI[j*inW*layers+i*layers+k];
                //cout << static_cast<int>(outI[idx-1]);//(inI[j*inW*layers+i*layers+k]) << endl;;
            }
        }
    }
//    idx = 0;
//    for (std::size_t i=0; i<width; i++){
//        for (std::size_t j=0; j<height; j++){
//            for (std::size_t k=0; k < layers; k++){
//                cout << static_cast<int>(outI[idx++]) << endl;
//            }
//        }
//    }
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