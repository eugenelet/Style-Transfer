#include <opencv2/opencv.hpp>

#include <iostream>
#include <cmath>
#include <fstream>
#include <complex>

using namespace cv;
using namespace std;

double deg_2_rad = M_PI / 180;


#define SIGMA_CLIP 6.0f
inline int sigma2radius(float sigma);
inline float radius2sigma(int r);
void fftShift(Mat magI);
void computeDFT(Mat& image, Mat& dest);
double MSE(Mat original, Mat output, int channel);
double PSNR(Mat original, Mat output);
// void convertMagPhase(Mat& image, Mat& dftOut, Mat& mag, Mat& phase);
void swapMagPhase(Mat& dft1, Mat& dft2);

int main(int argc, char ** argv)
{
    if(argc < 3)
        cout << "./executable [input] [input2]" << endl;


    char* filename = argv[1];
    char* filename2 = argv[2];
    char* outFilename;



    Mat inputImage = imread(filename);
    if( inputImage.empty())
        return -1;

    Mat inputImage2 = imread(filename2);
    if( inputImage2.empty())
        return -1;
    

    resize(inputImage2, inputImage2, inputImage.size());

    /*Prepare container for final image output*/
    Mat mergedImage = inputImage.clone();

    imshow("input", inputImage);
    imshow("input2", inputImage2);


    Mat channel[3];
    split(inputImage, channel);

    Mat channel2[3];
    split(inputImage2, channel2);

    Mat I, I2;
    Mat outputChannel[3], outputChannel2[3];

    for (int i = 0; i < 3; i++){

        /*Work on a single channel*/
        I = channel[i].clone();
        I2 = channel2[i].clone();

        /*FFT*/
        Mat dftMat;
        computeDFT(I, dftMat);
        Mat dftMat2;
        computeDFT(I2, dftMat2);


        swapMagPhase(dftMat,dftMat2);


        /*Inverse DFT to Spacial domain*/
        Mat inverseTransform;
        idft(dftMat, inverseTransform,DFT_REAL_OUTPUT+DFT_SCALE);
        Mat inverseTransform2;
        idft(dftMat2, inverseTransform2,DFT_REAL_OUTPUT+DFT_SCALE);


        /*Store results in temporary channel*/
        cv::Mat finalImage;
        inverseTransform.convertTo(finalImage, CV_8U);
        outputChannel[i] = finalImage.clone();
        inverseTransform2.convertTo(finalImage, CV_8U);
        outputChannel2[i] = finalImage.clone();

    }


    Mat outPlane[] = {outputChannel[0], outputChannel[1], outputChannel[2]};
    merge(outPlane, 3, mergedImage);

    imshow("result", mergedImage);

    imwrite("swap1.jpg", mergedImage);


    Mat outPlane2[] = {outputChannel2[0], outputChannel2[1], outputChannel2[2]};
    merge(outPlane2, 3, mergedImage);

    imshow("result2", mergedImage);

    imwrite("swap2.jpg", mergedImage);

    waitKey();


    // cout << "PSNR: " << PSNR(answer, mergedImage) << endl;

    return 0;
}

/*Convert sigma to radius for Gaussian Blur*/
inline int sigma2radius(float sigma)
{
    return (int)(SIGMA_CLIP*sigma+0.5f);
}

inline float radius2sigma(int r)
{
    return (r/SIGMA_CLIP+0.5f);
}




double PSNR(Mat original, Mat output){
    double sum = 0;
    for(int i = 0; i < 3; i++)
        sum += 10 * log10(pow(255,2) / MSE(original,output,i));

    return sum;
}

double MSE(Mat original, Mat output, int channel){
    int rows = original.rows;
    int cols = original.cols;
    double sum = 0;
    for(int row = 0; row < rows; row++)
        for(int col = 0; col < cols; col++)
            sum += pow(output.at<Vec3b>(row,col).val[channel] - original.at<Vec3b>(row,col).val[channel], 2);

    return sum / (rows*cols); 
}

/*move quadrants*/
void fftShift(Mat magI)
{
    // crop if it has an odd number of rows or columns
    magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));

    int cx = magI.cols/2;
    int cy = magI.rows/2;

    Mat q0(magI, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
    Mat q1(magI, Rect(cx, 0, cx, cy));  // Top-Right
    Mat q2(magI, Rect(0, cy, cx, cy));  // Bottom-Left
    Mat q3(magI, Rect(cx, cy, cx, cy)); // Bottom-Right

    Mat tmp;                            // swap quadrants (Top-Left with Bottom-Right)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    q1.copyTo(tmp);                     // swap quadrant (Top-Right with Bottom-Left)
    q2.copyTo(q1);
    tmp.copyTo(q2);
}

/*Compute DFT*/
void computeDFT(Mat& image, Mat& dest)
{
    Mat padded;                            //expand input image to optimal size
    int m = getOptimalDFTSize( image.rows );
    int n = getOptimalDFTSize( image.cols ); // on the border add zero values

    copyMakeBorder(image, padded, 0, m - image.rows, 0, n - image.cols, BORDER_REPLICATE);

    Mat imgf;
    padded.convertTo(imgf,CV_32F);  
    dft(imgf, dest, DFT_COMPLEX_OUTPUT);  // furier transform

}

void swapMagPhase(Mat& dft1, Mat& dft2){
    Mat planes1[2], planes2[2];
    split(dft1, planes1);// planes[0] = Re(DFT(I)), planes[1] = Im(DFT(I))
    split(dft2, planes2);// planes[0] = Re(DFT(I)), planes[1] = Im(DFT(I))

    Mat ph1, mag1;
    mag1.zeros(planes1[0].rows, planes1[0].cols, CV_32F);
    ph1.zeros(planes1[0].rows, planes1[0].cols, CV_32F);
    cartToPolar(planes1[0], planes1[1], mag1, ph1);

    Mat ph2, mag2;
    mag2.zeros(planes2[0].rows, planes2[0].cols, CV_32F);
    ph2.zeros(planes2[0].rows, planes2[0].cols, CV_32F);
    cartToPolar(planes2[0], planes2[1], mag2, ph2);

    polarToCart(mag1, ph2, planes1[0], planes1[1]);
    polarToCart(mag2, ph1, planes2[0], planes2[1]);

    merge(planes1, 2, dft1);
    merge(planes2, 2, dft2);

}