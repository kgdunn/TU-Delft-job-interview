#ifndef flotation_h
#define flotation_h

#include <iostream>
#include <string>
using namespace std;

struct param{};

class Image
{
private:
    int rows_, cols_, layers_;
    float *src;
    string filename;
public:
    bool is_complex;
    
    // Three different constructor options
    Image();
    Image(const string & filename);
    Image(int rows, int cols, int layers, bool is_complex = false);
    ~Image();
  
    // Public member functions
    float& get_pixel(int i, int j) { return src[cols_*j + i]; }
    int width() { return cols_; }
    int height() { return rows_; }
    void offset_then_scale(float offset, float scale);
    
//    template<typename TPixel> void SetImage(TPixel* srcImage, uint srcpitch);
//    void SetImage16Bit(ushort* srcImage, uint srcpitch) { SetImage(srcImage, srcpitch); }
//    void SetImage8Bit(uchar* srcImage, uint srcpitch) { SetImage(srcImage, srcpitch); }
//    void SetImageFloat(float* srcImage);
//    void SaveImage(const char *filename);
    
//    vector2f ComputeMeanAndCOM(float bgcorrection=0.0f);
//    void ComputeRadialProfile(float* dst, int radialSteps, int angularSteps, float minradius, float maxradius, vector2f center, bool crp, bool* boundaryHit=0, bool normalize=true);
//    void ComputeQuadrantProfile(scalar_t* dst, int radialSteps, int angularSteps, int quadrant, float minRadius, float maxRadius, vector2f center, float* radialWeights=0);
//    
//    float ComputeZ(vector2f center, int angularSteps, int zlutIndex, bool* boundaryHit=0, float* profile=0, float* cmpprof=0, bool normalizeProfile=true)
//    {
//        float* prof = profile ? profile : ALLOCA_ARRAY(float, zlut_res);
//        ComputeRadialProfile(prof,zlut_res,angularSteps, zlut_minradius, zlut_maxradius, center, false, boundaryHit, normalizeProfile);
//        return LUTProfileCompare(prof, zlutIndex, cmpprof, LUTProfMaxQuadraticFit);
//    }
    
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
