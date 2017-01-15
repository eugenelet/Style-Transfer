#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;


/** @function main */
int main( int argc, char** argv )
{
  /// Load source image and convert it to gray
  Mat img = imread( argv[1], 1 );

  imshow("input", img );


  Mat g1, g2, result;
  GaussianBlur(img, g1, Size(1,1), 0);
  GaussianBlur(img, g2, Size(3,3), 0);
  result = g1 - g2;
  imshow("result", result);

  waitKey(0);
  return(0);
}
