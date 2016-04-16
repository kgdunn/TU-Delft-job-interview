#ifndef flotation_h
#define flotation_h

#include <iostream>
#include <string>
using namespace std;

struct param{};

class Image
{
private:
    int rows_, cols_, layers_, length_;
    unsigned char *src_;
public:
    string filename;
    bool is_complex;
    
    // Three different constructor options
    Image();
    Image(int rows, int cols, int layers, bool is_complex = false);
    Image(const Image &);
    Image(const string & filename);         // Copy constructor
    //Image& operator=(const Image& input);   // Assignment operator
    ~Image();
  
    // Public member functions
    inline int width() { return cols_; }
    inline int height() { return rows_; }
    inline int layers() { return layers_; }
    inline int length() { return length_; }
    inline unsigned char* start() {return src_;}
    void offset_then_scale(float offset, float scale);

    //void FourierTransform2D();
    //void FourierRadialProfile(float* dst, int radialSteps, int angularSteps, float minradius, float maxradius);
    
//    void Normalize(float *image=0);
    
    
};
// Function prototypes, roughly in the order they are used
param load_model_parameters(std::string directory, std::string filename);
Image read_image(std::string filename);
Image subsample_image(Image inImg);
Image colour2gray(Image inImg);
Image fft2_image(Image inImg);
Image iff2_image(Image inImg);
Image gauss_cwt(Image inImg, param model);
Image multiply_scalar(Image inImg, double scalar);
std::vector<double> threshold(Image inImg, param model);
std::vector<double> project_onto_model(std::vector<double> features, param model);



#endif /* flotation_h */
