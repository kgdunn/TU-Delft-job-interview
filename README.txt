This code processes images from a flotation cell and extracts textures from
the image. These texture features are regressed onto a multivariate
(principal component analysis, PCA) model, to determine how close the
appearance of the image is to the "desirable" appearance.

People that operate the flotation cells can therefore use this model to
adjust settings on the process, so that process moves to stable operation.
The outputs can (and have) been used for automatic control of appearance.

All code (except for external libraries), have been written by Kevin Dunn
14 to 18 April 2016, as demonstration for a job interview at TU Delft.

This demonstration code was written and tested in XCode 7.3. 

The code will be explained in detail on Tuesday, 19 April, but if you 
would like browse, please go ahead. Note that (minor) changes will
occur before Tuesday. Top of this list is the addition of OpenCV
display windows, to visually illustrate what the code is doing.

A demonstration image has been added to the repository to check the code.
You will need to alter the directory paths in the code to adjust for 
file locations on your computer.

To compile and run the code you will require 5 other libraries. 
Details for installing and testing these are given below.


1. FFTW: http://www.fftw.org/download.html
1a) ./configure
1b) make
1c) make install (installs the library to /usr/local/include)
1d) Update the XCode settings for the project to include the above 
    directory path: Build Settings --> Search Paths --> "Header Search Paths"
1e) Now you have use it in the code: #include <fftw3.h>
1f) You also need to adjust the "Other Linker Flags" setting: -lfftw3 -lm
1g) And you need to adjust where the library files (the *.a files) are: 
    Build Settings --> Search Paths --> "Library Search Paths" and add "/usr/local/lib"

2. BMP library: http://partow.net/programming/bitmap/
2a) download the files and add them to the C++ project

3.  OpenCV: http://opencv.org/downloads.html
3a. Unzip, and go into the directory 
3b. Download https://cmake.org/download/ CMake (to make life easier) 
3c. Follow these instructions http://blogs.wcode.org/2014/10/howto-install-build-and-use-opencv-macosx-10-10/
3d. The last part of this directory will change: 
    /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.11.sdk/
3e. It may require and internet connection to download extra files
3f. Making from the command line then takes a while
3g. sudo make install
Also see https://www.youtube.com/watch?v=XJeP1juuHHY
3h. Pay attention to where the .hpp files are copied. 
    For example: "/usr/local/include/"
3i. Add this path to the Build Settings --> Search Paths --> "Header Search Paths"
3j. Pay attention to where the .a files are copied. For example: "/usr/local/lib"
3k. Add this path to the Build Settings --> Search Paths --> "Library Search Paths
3l. Add the Linker Flags:  

 -lopencv_calib3d -lopencv_core -lopencv_features2d -lopencv_flann -lopencv_highgui
 -lopencv_imgcodecs -lopencv_imgproc -lopencv_ml -lopencv_objdetect -lopencv_photo 
 -lopencv_shape -lopencv_stitching -lopencv_superres -lopencv_ts -lopencv_video 
 -lopencv_videoio -lopencv_videostab

3m. Check with this code:
#include "opencv2/opencv.hpp"
using namespace std;
using namespace cv;
int main( )
{
    std::cout << CV_VERSION << endl;
    return 0;
}

4. Matrix library Eigen: http://eigen.tuxfamily.org/index.php?title=Main_Page
4a. Download and unzip
4b. Copy the Eigen/ subdirectory to the source tree
4c. Add that directory to the project 
4d. An alternative is to make; make install it in the usual way


5. Boost
5a. Download the latest version: 
    http://www.boost.org/doc/libs/1_60_0/more/getting_started/unix-variants.html
5b. Unpack it: tar -zxfv boost_1_60_0.tar.bz2
5c. ./bootstrap.sh
5d. ./b2
5e. ./b2 install
5f. As with OpenCV, you will need to update your search paths for the Library and 
    Header files respectively, but likely
    the settings above to "/usr/local/include" are sufficient.
5g. You will require these Linker flags, in addition to those from OpenCV:

	-lboost_system -lboost_filesystem
	
