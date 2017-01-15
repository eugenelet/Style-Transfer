
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;
using namespace cv;

int thres_slider, thres_slider_t;
int partition_slider, partition_slider_t;
Mat *segmentedPlanes_target, *segmentedPlanes_src;
vector<Vec3b> segmentedColors_src;
Mat partitionedSrc;
// Mat src, originalImg;
void on_trackbar(int, void*);
void partition(Mat src, Mat &markers, Mat &dst, vector<vector<Point> > &contours, vector<Vec3b> &colors);
void on_trackbar_t(int, void*);

/*FFT TOOLS*/
void fftShift(Mat magI);
void computeDFT(Mat& image, Mat& dest);
void swapMagPhase(Mat& dft1, Mat& dft2);
void perform_Swap(Mat &inputImage, Mat &inputImage2, Mat &mergedImage, Mat &mergedImage2);

// Fill gaps in segmented plane
void fillSegmentedPlane(Mat &plane);

// Generate mapping between src and tar
void generateSwapHash(int, Mat*, Mat*, int*);
// void partition_t(Mat src, Mat &markers, Mat &dst, vector<vector<Point> > &contours, vector<Vec3b> &colors);

struct src_target
{
	Mat image;
	string identity;
};


Mat laplac;
int main(int, char** argv)
{
	Mat src;
    // Load the image
    src = imread(argv[1]);
    // Check if everything was fine
    if (!src.data)
        return -1;

    // Mat laplac;
    // Laplacian(src, laplac, src.depth());
    // imshow("lap", laplac);
	Mat target;
    // Load the image
    target = imread(argv[2]);
    // Check if everything was fine
    if (!target.data)
        return -1;

    // originalImg = src.clone();
    // Show source image
    imshow("Source Image", src);
    imshow("Target Image", target);
    // Change the background from white to black, since that will help later to extract
    // better results during the use of Distance Transform
    /*for( int x = 0; x < src.rows; x++ ) {
      for( int y = 0; y < src.cols; y++ ) {
          if ( src.at<Vec3b>(x, y) == Vec3b(255,255,255) ) {
            src.at<Vec3b>(x, y)[0] = 0;
            src.at<Vec3b>(x, y)[1] = 0;
            src.at<Vec3b>(x, y)[2] = 0;
          }
        }
    }
    // Show output image
    imshow("Black Background Image", src);*/
    // Create a kernel that we will use for accuting/sharpening our image
    

    /// Initialize values
    thres_slider_t = thres_slider = 3; 
    partition_slider_t = partition_slider = 2;
  
    src_target src_struct, target_struct;
    src_struct.image = src;
    src_struct.identity = "source";
    target_struct.image = target;
    target_struct.identity = "target";

    /// Create Windows
    namedWindow("source", 1); 
    /// Create Trackbars
    char TrackbarName[50];
    sprintf( TrackbarName, "Dilation Threshold (Markers) x %d", 20 );
    createTrackbar( TrackbarName, "source", &thres_slider, 20, on_trackbar, &src_struct );
    sprintf( TrackbarName, "Partition Number x %d", 10 );
    createTrackbar( TrackbarName, "source", &partition_slider, 10, on_trackbar, &src_struct );
    on_trackbar( thres_slider, &src_struct);

    /// Create Windows
    namedWindow("target", 1); 
    /// Create Trackbars
    sprintf( TrackbarName, "Dilation Threshold (Markers) x %d", 20 );
    createTrackbar( TrackbarName, "target", &thres_slider_t, 20, on_trackbar_t, &target_struct );
    sprintf( TrackbarName, "Partition Number x %d", 10 );
    createTrackbar( TrackbarName, "target", &partition_slider, 10, on_trackbar_t, &target_struct );
    on_trackbar_t( thres_slider_t, &target_struct );

    waitKey(0);

    return 0;
}

void on_trackbar(int, void* ptr){
	Mat src =  ((src_target*)ptr)->image;
	string w_name = ((src_target*)ptr)->identity;
	Mat originalImg = src.clone();
	Mat kernel = (Mat_<float>(3,3) <<
            1,  1, 1,
            1, -8, 1,
            1,  1, 1); 
    Mat imgLaplacian;
    Mat sharp = src; // copy source image to another temporary one
    filter2D(sharp, imgLaplacian, CV_32F, kernel);
    src.convertTo(sharp, CV_32F);
    Mat imgResult = sharp - imgLaplacian;
    // convert back to 8bits gray scale
    imgResult.convertTo(imgResult, CV_8UC3);
    imgLaplacian.convertTo(imgLaplacian, CV_8UC3);
    // imshow( "Laplace Filtered Image", imgLaplacian );
    // imshow( "New Sharped Image", imgResult );
    src = imgResult; // copy back
    // Create binary image from source image
    


    Mat bw;
    cvtColor(src, bw, CV_BGR2GRAY);
    // Convert to binary image for easier computation
    threshold(bw, bw, 40, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
    // imshow("Binary Image", bw);

    // Perform the distance transform algorithm
    Mat dist;
    distanceTransform(bw, dist, CV_DIST_L2, 3);
    // Normalize the distance image for range = {0.0, 1.0}
    // so we can visualize and threshold it
    normalize(dist, dist, 0, 1., NORM_MINMAX);
    // imshow("Distance Transform Image", dist);

    // Generate Markers
    // Threshold to obtain the peaks
    // This will be the markers for the foreground objects
    threshold(dist, dist, .4, 1., CV_THRESH_BINARY);

    // Dilate a bit the dist image
    // thres_slider determines the degree of dilation
    Mat kernel1 = Mat::ones(thres_slider, thres_slider, CV_8UC1); 
    dilate(dist, dist, kernel1);
    // imshow("Peaks", dist);

    // Create the CV_8U version of the distance image
    // It is needed for findContours()
    Mat dist_8u;
    dist.convertTo(dist_8u, CV_8U);

    // Find total markers
    vector<vector<Point> > contours;
    findContours(dist_8u, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    // Create the marker image for the watershed algorithm
    Mat markers = Mat::zeros(dist.size(), CV_32SC1);
    // Draw the foreground markers
    // cout << contours.size() << endl;
    for (size_t i = 0; i < contours.size(); i++){
        drawContours(markers, contours, static_cast<int>(i), Scalar::all(static_cast<int>(i)+1), -1);
        // imshow("Mark", markers*10000);
        // waitKey(0);
    }
    // Draw the background marker
    circle(markers, Point(5,5), 3, CV_RGB(255,255,255), -1);
    // imshow("Markers", markers*10000);

    // Perform the watershed algorithm
    watershed(src, markers);
    Mat mark = Mat::zeros(markers.size(), CV_8UC1);
    markers.convertTo(mark, CV_8UC1);
    // imshow("test", markers*10000);
    bitwise_not(mark, mark);
    // imshow("Markers_v2", mark); // uncomment this if you want to see how the mark
                                  // image looks like at that point
    // Generate random colors
    vector<Vec3b> colors;
    for (size_t i = 0; i < contours.size(); i++)
    {
        int b = theRNG().uniform(0, 255);
        int g = theRNG().uniform(0, 255);
        int r = theRNG().uniform(0, 255);
        colors.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
    }
    // Create the result image
    Mat dst = Mat::zeros(markers.size(), CV_8UC3);
    
    cout << "!!" << endl;
    partition(src, markers, dst, contours, colors);
    cout << "!!" << endl;

    // Visualize the final image
    imshow(w_name, dst);

    vector<Vec3b> segmentedColors(partition_slider + 1);

    segmentedColors[0] = Vec3b(0,0,0);
    for(int i=1; i<=partition_slider; i++)
    	for(int row=0; row<dst.rows; row++){
    		for(int col=0; col<dst.cols; col++){
    			bool new_color = true;
    			for(int j=0; j<i; j++)
	    			if(dst.at<Vec3b>(row,col) == segmentedColors[j])
	    				new_color = false;
			    if(new_color){
			    	segmentedColors[i] = dst.at<Vec3b>(row,col);
			    	break;	
			    }
    		}
			if(segmentedColors[i-1]!=segmentedColors[i] && segmentedColors[i]!=Vec3b(0,0,0))
				break;
    	}


    Mat *segmentedPlanes = new Mat[partition_slider + 1];

    for(int i=0; i<partition_slider + 1; i++)
    	segmentedPlanes[i] = originalImg.clone();

    for(int i=0; i<partition_slider + 1; i++)
    	for(int row=0; row<segmentedPlanes[i].rows; row++)
    		for(int col=0; col<segmentedPlanes[i].cols; col++)
    			if(dst.at<Vec3b>(row,col) != segmentedColors[i])
    				segmentedPlanes[i].at<Vec3b>(row,col) = Vec3b(0,0,0);


    segmentedPlanes_src = new Mat[partition_slider + 1];
    for(int i=0; i<partition_slider + 1; i++)
    	segmentedPlanes_src[i] = segmentedPlanes[i].clone();

    segmentedColors_src = segmentedColors;
    partitionedSrc = dst.clone();
}


void partition(Mat src, Mat &markers, Mat &dst, vector<vector<Point> > &contours, vector<Vec3b> &colors){
    vector<vector<Point> > coordinates(contours.size()+1, vector<Point>(0));


    // Fill labeled objects with random colors
    for (int i = 0; i < markers.rows; i++)
    {
        for (int j = 0; j < markers.cols; j++)
        {
            int index = markers.at<int>(i,j);

            if (index > 0 && index <= static_cast<int>(contours.size())){
                coordinates[index].push_back(Point(i,j));
                dst.at<Vec3b>(i,j) = colors[index-1];
            }
            else
                dst.at<Vec3b>(i,j) = Vec3b(0,0,0);
        }
    }

    int max = 0;
    for (int i = 0; i < markers.rows; i++)
        for (int j = 0; j < markers.cols; j++)
            if (markers.at<int>(i,j) > max)
            	max = markers.at<int>(i,j);


    // Computes the total number of pixels for each region
    int *accu = new int[256];
    for(int i=0;i<256;i++)
    	accu[i] = 0;
    for (int i = 0; i < markers.rows; i++)
        for (int j = 0; j < markers.cols; j++)
            if (markers.at<int>(i,j)>0 && markers.at<int>(i,j)<256)
	            ++accu[markers.at<int>(i,j)];

	// Sort histogram of value
    vector<int> sortVector (accu, accu+contours.size()+1);
    sort(sortVector.begin(), sortVector.end());

    // Stores the index of region containing the top n number of pixels
    int *top_index = new int[partition_slider];
    for(int i=0; i<partition_slider; i++)
    	top_index[i] = 0;
    for(int i=0; i<partition_slider; i++)
        for(int j=0; j<256; j++)
            if(accu[j]==sortVector[sortVector.size()-i-1])
                top_index[i] = j;
            


    // Create grayscale image to compute mean
    Mat gray_src;
    cvtColor(src, gray_src, CV_BGR2GRAY);

    // Compute mean of each region
    int *gray_mean = new int[coordinates.size()];
    int count = 0;
    for(int i=0; i<coordinates.size(); i++){
        gray_mean[i] = 0;
        for(int j=0; j<coordinates[i].size(); j++){
            count++;
            gray_mean[i] += (int)gray_src.at<uchar>(coordinates[i][j].x, coordinates[i][j].y);
        }
        if(count==0)
            continue;
        gray_mean[i] /= count;
        count = 0;
    }



    // Compute Variance of each region
    int *gray_variance = new int[coordinates.size()];
    count = 0;
    for(int i=0; i<coordinates.size(); i++){
        gray_variance[i] = 0;
        for(int j=0; j<coordinates[i].size(); j++){
            count++;
            gray_variance[i] += pow((int)gray_src.at<uchar>(coordinates[i][j].x, coordinates[i][j].y) - gray_mean[i], 2);
        }
        if(count==0)
            continue;
        gray_variance[i] /= count;
        count = 0;
    }

    // Categorize all regions into n major regions based on MMSE
    for (int i = 0; i < markers.rows; i++)
    {
        for (int j = 0; j < markers.cols; j++)
        {
            int index = markers.at<int>(i,j);
            if (index > 0 && index <= static_cast<int>(contours.size())){
                int *choice = new int[partition_slider];
                bool non_top = true;
                for(int p_count=0; p_count<partition_slider; p_count++)
                	if(index==top_index[p_count]){
                		non_top = false;
                		break;
                	}
                if(non_top){
                	int min = INT_MAX;
                	int min_index = 0;
                	for(int c_count=0; c_count<partition_slider; c_count++){
                		if(top_index[c_count]> coordinates.size() || index> coordinates.size())
                			cout << top_index[c_count] << "_" << index << "!!" << coordinates.size() << endl;
                		choice[c_count] = abs(gray_variance[top_index[c_count]] - gray_variance[index]);
                		if(choice[c_count] < min){
                			min = choice[c_count];
                			min_index = c_count;
                		}
                	}
                    markers.at<int>(i,j) = top_index[min_index];
                }
                delete[] choice;
            }
        }
    }


    // Fill color for pre-categorized regions
    for (int i = 0; i < markers.rows; i++)
    {
        for (int j = 0; j < markers.cols; j++)
        {
            int index = markers.at<int>(i,j);

            if (index > 0 && index <= static_cast<int>(contours.size())){
                dst.at<Vec3b>(i,j) = colors[index-1];
            }
            else
                dst.at<Vec3b>(i,j) = Vec3b(0,0,0);
        }
    }

    // Fill in gaps caused by watershed segmentation
    medianBlur(dst,dst,5);

}



// For target

void on_trackbar_t(int, void* ptr){
	Mat src = ((src_target*)ptr)->image;
	string w_name = ((src_target*)ptr)->identity;
	Mat originalImg = src.clone();
	Mat kernel = (Mat_<float>(3,3) <<
            1,  1, 1,
            1, -8, 1,
            1,  1, 1); 
    Mat imgLaplacian;
    Mat sharp = src; // copy source image to another temporary one
    filter2D(sharp, imgLaplacian, CV_32F, kernel);
    src.convertTo(sharp, CV_32F);
    Mat imgResult = sharp - imgLaplacian;
    // convert back to 8bits gray scale
    imgResult.convertTo(imgResult, CV_8UC3);
    imgLaplacian.convertTo(imgLaplacian, CV_8UC3);
    // imshow( "Laplace Filtered Image", imgLaplacian );
    // imshow( "New Sharped Image", imgResult );
    src = imgResult; // copy back
    // Create binary image from source image
    


    Mat bw;
    cvtColor(src, bw, CV_BGR2GRAY);
    // Convert to binary image for easier computation
    threshold(bw, bw, 40, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
    // imshow("Binary Image", bw);

    // Perform the distance transform algorithm
    Mat dist;
    distanceTransform(bw, dist, CV_DIST_L2, 3);
    // Normalize the distance image for range = {0.0, 1.0}
    // so we can visualize and threshold it
    normalize(dist, dist, 0, 1., NORM_MINMAX);
    // imshow("Distance Transform Image", dist);

    // Generate Markers
    // Threshold to obtain the peaks
    // This will be the markers for the foreground objects
    threshold(dist, dist, .4, 1., CV_THRESH_BINARY);

    // Dilate a bit the dist image
    // thres_slider_t determines the degree of dilation
    Mat kernel1 = Mat::ones(thres_slider_t, thres_slider_t, CV_8UC1); 
    dilate(dist, dist, kernel1);
    // imshow("Peaks", dist);

    // Create the CV_8U version of the distance image
    // It is needed for findContours()
    Mat dist_8u;
    dist.convertTo(dist_8u, CV_8U);

    // Find total markers
    vector<vector<Point> > contours;
    findContours(dist_8u, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    // Create the marker image for the watershed algorithm
    Mat markers = Mat::zeros(dist.size(), CV_32SC1);
    // Draw the foreground markers
    // cout << contours.size() << endl;
    for (size_t i = 0; i < contours.size(); i++){
        drawContours(markers, contours, static_cast<int>(i), Scalar::all(static_cast<int>(i)+1), -1);
        // imshow("Mark", markers*10000);
        // waitKey(0);
    }
    // Draw the background marker
    circle(markers, Point(5,5), 3, CV_RGB(255,255,255), -1);
    // imshow("Markers", markers*10000);

    // Perform the watershed algorithm
    watershed(src, markers);
    Mat mark = Mat::zeros(markers.size(), CV_8UC1);
    markers.convertTo(mark, CV_8UC1);
    // imshow("test", markers*10000);
    bitwise_not(mark, mark);
    // imshow("Markers_v2", mark); // uncomment this if you want to see how the mark
                                  // image looks like at that point
    // Generate random colors
    vector<Vec3b> colors;
    for (size_t i = 0; i < contours.size(); i++)
    {
        int b = theRNG().uniform(0, 255);
        int g = theRNG().uniform(0, 255);
        int r = theRNG().uniform(0, 255);
        colors.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
    }
    // Create the result image
    Mat dst = Mat::zeros(markers.size(), CV_8UC3);
    

    partition(src, markers, dst, contours, colors);

    // Visualize the final image
    imshow(w_name, dst);

    vector<Vec3b> segmentedColors(partition_slider + 1);

    segmentedColors[0] = Vec3b(0,0,0);
    for(int i=1; i<=partition_slider; i++)
    	for(int row=0; row<dst.rows; row++){
    		for(int col=0; col<dst.cols; col++){
    			bool new_color = true;
    			for(int j=0; j<i; j++)
	    			if(dst.at<Vec3b>(row,col) == segmentedColors[j])
	    				new_color = false;
			    if(new_color){
			    	segmentedColors[i] = dst.at<Vec3b>(row,col);
			    	break;	
			    }
    		}
			if(segmentedColors[i-1]!=segmentedColors[i] && segmentedColors[i]!=Vec3b(0,0,0))
				break;
    	}

    
    Mat *segmentedPlanes = new Mat[partition_slider + 1];

    for(int i=0; i<partition_slider + 1; i++)
    	segmentedPlanes[i] = originalImg.clone();

    for(int i=0; i<partition_slider + 1; i++)
    	for(int row=0; row<segmentedPlanes[i].rows; row++)
    		for(int col=0; col<segmentedPlanes[i].cols; col++)
    			if(dst.at<Vec3b>(row,col) != segmentedColors[i])
    				segmentedPlanes[i].at<Vec3b>(row,col) = Vec3b(0,0,0);



    // Fill empty space with mean value
    for(int i=0; i<partition_slider + 1; i++)
	    fillSegmentedPlane(segmentedPlanes[i]);

    // Move to global
    segmentedPlanes_target = new Mat[partition_slider + 1];
    for(int i=0; i<partition_slider + 1; i++)
    	segmentedPlanes_target[i] = segmentedPlanes[i].clone();

    delete[] segmentedPlanes;

    // Generate mapping between src and target partitions
    int *target_hash = new int[partition_slider + 1];
    generateSwapHash(partition_slider, segmentedPlanes_src, segmentedPlanes_target, target_hash);

    // Print mapping relation
    for(int i=0; i<partition_slider+1; i++)
    	cout << i << ' ' << target_hash[i] << endl;
    // Swap magnitude and phase spectrum of image
    Mat *swap1 = new Mat[partition_slider + 1];
    Mat *swap2 = new Mat[partition_slider + 1];
    for(int i=0; i<partition_slider + 1; i++){
	    perform_Swap(segmentedPlanes_src[i], segmentedPlanes_target[target_hash[i]], swap1[i], swap2[i]);
	   	// imshow("1swap1", swap1);
	    // imshow("swap2 " + to_string(i), swap2[i]);
    }


    // Merge swaped partition
    Mat mergedSwap = segmentedPlanes_src[0].clone();
    for(int i=0; i<partition_slider + 1; i++)
    	for(int row=0; row<segmentedPlanes_src[i].rows; row++)
    		for(int col=0; col<segmentedPlanes_src[i].cols; col++)
    			if(partitionedSrc.at<Vec3b>(row,col) == segmentedColors_src[i])
    				mergedSwap.at<Vec3b>(row,col) = swap2[i].at<Vec3b>(row,col);

   	Laplacian(mergedSwap, laplac, mergedSwap.depth());
   	imshow("laplace", laplac);
    mergedSwap = mergedSwap + laplac;
    imshow("swaped image", mergedSwap);
}

void generateSwapHash(	int partition_slider, 
						Mat *segmentedPlanes_src, 
						Mat *segmentedPlanes_target, 
						int *target_hash)
{
	Mat *src = new Mat[partition_slider + 1];
	Mat *tar = new Mat[partition_slider + 1];
	/// Mean
	int *mean_src = new int[partition_slider + 1];
	int *mean_tar = new int[partition_slider + 1];
	int *count_src = new int[partition_slider + 1];
	int *count_tar = new int[partition_slider + 1];

	// Initialize mean
	for(int i=0; i<partition_slider+1; i++){
		mean_src[i] = 0;
		mean_tar[i] = 0;
		count_src[i] = 0;
		count_tar[i] = 0;
	}


	// Compute Mean
	for(int i=0; i<partition_slider+1; i++){
		cvtColor(segmentedPlanes_src[i], src[i], CV_BGR2GRAY);
		cvtColor(segmentedPlanes_target[i], tar[i], CV_BGR2GRAY);
		for(int row=0; row<src[i].rows; row++)
			for(int col=0; col<src[i].cols; col++)
				if(src[i].at<uchar>(row,col) != 0){
					mean_src[i] += src[i].at<uchar>(row,col);
					count_src[i]++;
				}
				
		for(int row=0; row<tar[i].rows; row++)
			for(int col=0; col<tar[i].cols; col++)
				if(tar[i].at<uchar>(row,col) != 0){
					mean_tar[i] += tar[i].at<uchar>(row,col);
					count_tar[i]++;
				}
	}
	for(int i=0; i<partition_slider+1; i++){
		mean_src[i] /= count_src[i];
		mean_tar[i] /= count_tar[i];
	}


	/// Variance
	int *var_src = new int[partition_slider + 1];
	int *var_tar = new int[partition_slider + 1];

	// Initialize variance
	for(int i=0; i<partition_slider+1; i++){
		var_src[i] = 0;
		var_tar[i] = 0;
		count_src[i] = 0;
		count_tar[i] = 0;
	}

	// Compute Variance
	for(int i=0; i<partition_slider+1; i++){
		for(int row=0; row<src[i].rows; row++)
			for(int col=0; col<src[i].cols; col++)
				if(src[i].at<uchar>(row,col) != 0){
					var_src[i] += pow(src[i].at<uchar>(row,col) - mean_src[i], 2);
					count_src[i]++;
				}
				
		for(int row=0; row<tar[i].rows; row++)
			for(int col=0; col<tar[i].cols; col++)
				if(tar[i].at<uchar>(row,col) != 0){
					var_tar[i] += pow(tar[i].at<uchar>(row,col) - mean_tar[i], 2);
					count_tar[i]++;
				}
	}
	for(int i=0; i<partition_slider+1; i++){
		var_src[i] /= count_src[i];
		var_tar[i] /= count_tar[i];
	}

	int min;
	for(int i=0; i<partition_slider+1; i++){
		min = INT_MAX;
		for(int j=0; j<partition_slider+1; j++)
			if(abs(var_src[i]-var_tar[j]) < min){
				min = abs(var_src[i]-var_tar[j]);
				target_hash[i] = j;
			}
		var_tar[target_hash[i]] = INT_MAX;
	}

}


void fillSegmentedPlane(Mat &plane){
	int mean_r = 0;
	int mean_g = 0;
	int mean_b = 0;
	int count = 0;
	for(int row=0; row<plane.rows; row++)
		for(int col=0; col<plane.cols; col++)
			if(plane.at<Vec3b>(row,col) != Vec3b(0,0,0)){
				mean_r += plane.at<Vec3b>(row,col)[0];
				mean_g += plane.at<Vec3b>(row,col)[1];
				mean_b += plane.at<Vec3b>(row,col)[2];
				count++;
			}

	mean_r /= count;
	mean_g /= count;
	mean_b /= count;
	Vec3b mean = Vec3b(mean_r,mean_g,mean_b);

	for(int row=0; row<plane.rows; row++)
		for(int col=0; col<plane.cols; col++)
			if(plane.at<Vec3b>(row,col) == Vec3b(0,0,0))
				plane.at<Vec3b>(row,col) = mean;
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



void perform_Swap(Mat &inputImage, Mat &inputImage2, Mat &mergedImage, Mat &mergedImage2){

    resize(inputImage2, inputImage2, inputImage.size());

    /*Prepare container for final image output*/
    mergedImage = inputImage.clone();
    mergedImage2 = inputImage.clone();


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


    Mat outPlane2[] = {outputChannel2[0], outputChannel2[1], outputChannel2[2]};
    merge(outPlane2, 3, mergedImage2);

}
