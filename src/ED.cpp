#include "ED.h"
#include "EDColor.h"
#include <fstream>

using namespace cv;
using namespace std;
cv::Mat line_width_h = cv::Mat::ones(3,3,CV_64FC1);

ED::ED(Mat _srcImage, GradientOperator _op, int _gradThresh, int _anchorThresh,int _scanInterval, int _minPathLen ,double _sigma, bool _sumFlag)
{	
	// Check parameters for sanity
	if (_gradThresh < 1) _gradThresh = 1;
	if (_anchorThresh < 0) _anchorThresh = 0;
	if (_sigma < 1.0) _sigma = 1.0;

	

	//srcImage = _srcImage(Rect(0,150,639,209));
	srcImage = _srcImage;


	height = srcImage.rows;
	width = srcImage.cols;
	
	op = _op;
	gradThresh = _gradThresh;
	anchorThresh = _anchorThresh;
	scanInterval = _scanInterval;
	minPathLen = _minPathLen;
	sigma = _sigma;
	sumFlag = _sumFlag;

	segmentNos = 0;
	segmentPoints.push_back(vector<Point>()); // create empty vector of points for segments

	edgeImage = Mat(height, width, CV_8UC1, Scalar(0)); // initialize edge Image
	anchorsImage = Mat(height, width, CV_8UC1, Scalar(0)); // initialize anchors Image
	smoothImage = Mat(height, width, CV_8UC1, Scalar(0));
	gradImage = Mat(height, width, CV_16SC1, Scalar(0)); // gradImage contains short values
	selectgradImage = Mat(height, width, CV_16SC1, Scalar(0)); // gradImage contains short values
	gradlinewidthImage = Mat(height, width, CV_16SC1); // gradImage contains short values 
	testImage = Mat(height, width, CV_8UC1, Scalar(0)); // initialize test Image
	srcImg = srcImage.data;
	smoothImg = smoothImage.data;
	//// Detect Edges By Edge Drawing Algorithm  ////
	
	/*------------ SMOOTH THE IMAGE BY A GAUSSIAN KERNEL -------------------*/
	if (sigma == 1.0){
		//GaussianBlur(srcImage, smoothImage, Size(5, 5), sigma);
		GaussianBlur(srcImage, smoothImage, Size(5,5), sigma);
		/*
		for(int i = 0; i < height; i++){
			uchar* srcdata_prev = srcImage.ptr<uchar>(i - 1);
			uchar* srcdata_curr = srcImage.ptr<uchar>(i);
			uchar* srcdata_next = srcImage.ptr<uchar>(i + 1);
			for(int j = 0; j < width; j++){
				
				int com1 = j - 1;
				int com2 = j + 1;
				int avg = (srcdata_prev[com1] + srcdata_prev[j] + srcdata_prev[com2] + 
						   srcdata_curr[com1] + srcdata_curr[j] + srcdata_curr[com2] + 
						   srcdata_next[com1] + srcdata_next[j] + srcdata_next[com2]) / 9;
				//smoothImg[i * width + j]=avg;
				
				
				int A1 = srcdata_prev[j - 1] - avg;
				int A2 = srcdata_prev[j] - avg;
				int A3 = srcdata_prev[j + 1] - avg;
				int A4 = srcdata_curr[j - 1] - avg;
				int A5 = srcdata_curr[j] - avg;
				int A6 = srcdata_curr[j + 1] - avg;
				int A7 = srcdata_next[j - 1] - avg;
				int A8 = srcdata_next[j] - avg;
				int A9 = srcdata_next[j + 1] - avg;
				int D = A1 * A1 + A2 * A2 + A3 * A3 + 
						A4 * A4 + A5 * A5 + A6 * A6 +
						A7 * A7 + A8 * A8 + A9 * A9;
				float fD = D / 9.f;
				//R = D * sigma^2 * exp(1/sigma^2), R = 1
				float f1 = 1.f/(2.f*CV_PI*fD*fD)*exp(-(2.f)/(2.f*fD*fD));
				float f2 = 1.f/(2.f*CV_PI*fD*fD)*exp(-(1.f)/(2.f*fD*fD));
				float f3 = 1.f/(2.f*CV_PI*fD*fD);
				
				float f = (srcdata_prev[com1] * f1 + srcdata_prev[j] * f2 + srcdata_prev[com2] * f1+ 
						   srcdata_curr[com1] * f2 + srcdata_curr[j] * f3 + srcdata_curr[com2] * f2+ 
						   srcdata_next[com1] * f1 + srcdata_next[j] * f2 + srcdata_next[com2] * f1);
				if (f > 255.f) f =255.f;
				smoothImg[i * width + j]=f;
				//cout<<"fD"<< fD<<endl;
				
			}
		}
		*/
	}
	else{
		GaussianBlur(srcImage, smoothImage, Size(), sigma); // calculate kernel from sigma
	}
	// Assign Pointers from Mat's data
	//imshow("srcImage", smoothImage);
	gradImg = (short*)gradImage.data;
	selectgradImg = (short*)selectgradImage.data;
	gradlinewidthImg = (short*)gradlinewidthImage.data;
	testImg = testImage.data;
	edgeImg = edgeImage.data;
	anchorsImg = anchorsImage.data;

	dirImg = new unsigned char[width*height];
	memset(dirImg, 0, width*height);

	/*
	HomographyMatrix.at<double>(0,0) = 3.739862280030613;
	HomographyMatrix.at<double>(0,1) = -0.08851745041492516;
	HomographyMatrix.at<double>(0,2) = -1154.035195103293;
	HomographyMatrix.at<double>(1,0) = 1.942214879979242e-15;
	HomographyMatrix.at<double>(1,1) = 34.31228297333891;
	HomographyMatrix.at<double>(1,2) = -1338.179035960218;
	HomographyMatrix.at<double>(2,0) = 2.115629703434065e-18;
	HomographyMatrix.at<double>(2,1) = 0.02307103760814556;
	HomographyMatrix.at<double>(2,2) = 1;
	*/
	
	
	/*------------ COMPUTE GRADIENT & EDGE DIRECTION MAPS -------------------*/
	ComputeGradient();

	/*------------ COMPUTE ANCHORS -------------------*/
	//ComputeAnchorPoints();
	ComputeLaneLineWidth();
	OComputeAnchorPoints();

	/*------------ COMPUTE LANE LINE WIDTH -------------------*/
	

	/*------------ JOIN ANCHORS -------------------*/
	JoinAnchorPointsUsingSortedAnchors();

	

	delete[] dirImg;
}



// This constructor for use of EDLines and EDCircle with ED given as constructor argument
// only the necessary attributes are coppied
ED::ED(const ED & cpyObj)
{
	height = cpyObj.height;
	width = cpyObj.width;

	srcImage = cpyObj.srcImage.clone();
	
	op = cpyObj.op;
	gradThresh = cpyObj.gradThresh;
	anchorThresh = cpyObj.anchorThresh;
	scanInterval = cpyObj.scanInterval;
	minPathLen = cpyObj.minPathLen;
	sigma = cpyObj.sigma;
	sumFlag = cpyObj.sumFlag;

	edgeImage = cpyObj.edgeImage.clone();
	smoothImage = cpyObj.smoothImage.clone();
	gradImage = cpyObj.gradImage.clone();
	selectgradImage = cpyObj.selectgradImage.clone();
	srcImg = srcImage.data;

	smoothImg = smoothImage.data;
	gradImg = (short*)gradImage.data;
	selectgradImg = (short*)selectgradImage.data;
	edgeImg = edgeImage.data;

	segmentPoints = cpyObj.segmentPoints;
	segmentNos = cpyObj.segmentNos;
}

// This constructor for use of EDColor with use of direction and gradient image
// It finds edge image for given gradient and direction image
ED::ED(short *_gradImg, uchar *_dirImg, int _width, int _height, int _gradThresh, int _anchorThresh, int _scanInterval, int _minPathLen, bool selectStableAnchors)
{
	height = _height;
	width = _width;

	gradThresh = _gradThresh;
	anchorThresh = _anchorThresh;
	scanInterval = _scanInterval;
	minPathLen = _minPathLen;

	gradImg = _gradImg;
	dirImg = _dirImg;

	edgeImage = Mat(height, width, CV_8UC1, Scalar(0)); // initialize edge Image

	edgeImg = edgeImage.data;

	if (selectStableAnchors) {

		// Compute anchors with the user supplied parameters
		anchorThresh = 0; // anchorThresh used as zero while computing anchor points if selectStableAnchors set. 
						  // Finding higher number of anchors is OK, because we have following validation steps in selectStableAnchors.
		ComputeAnchorPoints();
		anchorThresh = _anchorThresh; // set it to its initial argument value for further anchor validation.
		anchorPoints.clear(); // considering validation step below, it should constructed again.

		for (int i = 1; i<height - 1; i++) {
			for (int j = 1; j<width - 1; j++) {
				if (edgeImg[i*width + j] != ANCHOR_PIXEL) continue;

				// Take only "stable" anchors
				// 0 degree edge
				if (edgeImg[i*width + j - 1] && edgeImg[i*width + j + 1]) {
					int diff1 = gradImg[i*width + j] - gradImg[(i - 1)*width + j];
					int diff2 = gradImg[i*width + j] - gradImg[(i + 1)*width + j];
					if (diff1 >= anchorThresh && diff2 >= anchorThresh) edgeImg[i*width + j] = 255;

					continue;
				} //end-if

				  // 90 degree edge
				if (edgeImg[(i - 1)*width + j] && edgeImg[(i + 1)*width + j]) {
					int diff1 = gradImg[i*width + j] - gradImg[i*width + j - 1];
					int diff2 = gradImg[i*width + j] - gradImg[i*width + j + 1];
					if (diff1 >= anchorThresh && diff2 >= anchorThresh) edgeImg[i*width + j] = 255;

					continue;
				} //end-if

				  // 135 degree diagonal
				if (edgeImg[(i - 1)*width + j - 1] && edgeImg[(i + 1)*width + j + 1]) {
					int diff1 = gradImg[i*width + j] - gradImg[(i - 1)*width + j + 1];
					int diff2 = gradImg[i*width + j] - gradImg[(i + 1)*width + j - 1];
					if (diff1 >= anchorThresh && diff2 >= anchorThresh) edgeImg[i*width + j] = 255;
					continue;
				} //end-if

				  // 45 degree diagonal
				if (edgeImg[(i - 1)*width + j + 1] && edgeImg[(i + 1)*width + j - 1]) {
					int diff1 = gradImg[i*width + j] - gradImg[(i - 1)*width + j - 1];
					int diff2 = gradImg[i*width + j] - gradImg[(i + 1)*width + j + 1];
					if (diff1 >= anchorThresh && diff2 >= anchorThresh) edgeImg[i*width + j] = 255;
				} //end-if

			} //end-for
		} //end-for

		for (int i = 0; i<width*height; i++) 
			if (edgeImg[i] == ANCHOR_PIXEL) 
				edgeImg[i] = 0; 
			else if (edgeImg[i] == 255) {
				edgeImg[i] = ANCHOR_PIXEL;
				int y = i / width;
				int x = i % width;
				anchorPoints.push_back(Point(x, y)); // push validated anchor point to vector
			}
			
		anchorNos = (int)anchorPoints.size(); // get # of anchor pixels
	}

	else {
		// Compute anchors with the user supplied parameters
		ComputeAnchorPoints(); // anchorThresh used as given as argument. No validation applied. (No stable anchors.)
	} //end-else

	segmentNos = 0;
	segmentPoints.push_back(vector<Point>()); // create empty vector of points for segments

	JoinAnchorPointsUsingSortedAnchors();
}

ED::ED(EDColor &obj) 
{
	width = obj.getWidth();
	height = obj.getHeight();
	segmentPoints = obj.getSegments();
	segmentNos = obj.getSegmentNo();
}

ED::ED()
{
	//
}

Mat ED::getSelectGradImage()
{
	Mat result8UC1;
	convertScaleAbs(selectgradImage, result8UC1);
	return result8UC1;
}


Mat ED::getEdgeImage()
{
	return edgeImage;
}

Mat ED::getAnchorImage()
{
	Mat anchorImage = Mat(edgeImage.size(), edgeImage.type(), Scalar(0));

	std::vector<Point>::iterator it;

	for (it = anchorPoints.begin(); it != anchorPoints.end(); it++)
		anchorImage.at<uchar>(*it) = 255;

	return anchorImage;
}

Mat ED::getSmoothImage()
{
	return smoothImage;
}

Mat ED::getGradImage()
{	
	Mat result8UC1;
	convertScaleAbs(gradImage, result8UC1);
	
	return result8UC1;
}

Mat ED::gettestImage()
{
	return testImage;
}

int ED::getSegmentNo()
{
	return segmentNos;
}

int ED::getAnchorNo()
{
	return anchorNos;
}

std::vector<Point> ED::getAnchorPoints()
{
	return anchorPoints;
}

std::vector<std::vector<Point>> ED::getSegments()
{
	return segmentPoints;
}

std::vector<std::vector<Point>> ED::getSortedSegments()
{
		// sort segments from largest to smallest
		std::sort(segmentPoints.begin(), segmentPoints.end(), [](const std::vector<Point> & a, const std::vector<Point> & b) { return a.size() > b.size(); });

		return segmentPoints;
}

Mat ED::drawParticularSegments(std::vector<int> list)
{
	Mat segmentsImage = Mat(edgeImage.size(), edgeImage.type(), Scalar(0));

	std::vector<Point>::iterator it;
	std::vector<int>::iterator itInt;

	for (itInt = list.begin(); itInt != list.end(); itInt++)
		for (it = segmentPoints[*itInt].begin(); it != segmentPoints[*itInt].end(); it++)
			segmentsImage.at<uchar>(*it) = 255;
	
	return segmentsImage;
}


void ED::ComputeGradient()
{	
	// Initialize gradient image for row = 0, row = height-1, column=0, column=width-1 
	for (int j = 0; j<width; j++) { selectgradImg[j] = selectgradImg[(height - 1)*width + j] = gradImg[j] = gradImg[(height - 1)*width + j] = gradThresh - 1; }
	for (int i = 1; i<height - 1; i++) { selectgradImg[i*width] = selectgradImg[(i + 1)*width - 1] = gradImg[i*width] = gradImg[(i + 1)*width - 1] = gradThresh - 1; }
	
	for (int i = 1; i<height - 1; i++) {
		for (int j = 1; j<width - 1; j++) {
			// Prewitt Operator in horizontal and vertical direction
			// A B C
			// D x E
			// F G H
			// gx = (C-A) + (E-D) + (H-F)
			// gy = (F-A) + (G-B) + (H-C)
			//
			// To make this faster: 
			// com1 = (H-A)
			// com2 = (C-F)
			// 
			// For Prewitt
			// Then: gx = com1 + com2 + (E-D) = (H-A) + (C-F) + (E-D) = (C-A) + (E-D) + (H-F)
			//       gy = com1 - com2 + (G-B) = (H-A) - (C-F) + (G-B) = (F-A) + (G-B) + (H-C)
			// 
			// For Sobel
			// Then: gx = com1 + com2 + 2*(E-D) = (H-A) + (C-F) + 2*(E-D) = (C-A) + 2*(E-D) + (H-F)
			//       gy = com1 - com2 + 2*(G-B) = (H-A) - (C-F) + 2*(G-B) = (F-A) + 2*(G-B) + (H-C)
			//
			// For Scharr
			// Then: gx = 3*(com1 + com2) + 10*(E-D) = 3*(H-A) + 3*(C-F) + 10*(E-D) = 3*(C-A) + 10*(E-D) + 3*(H-F)
			//       gy = 3*(com1 - com2) + 10*(G-B) = 3*(H-A) - 3*(C-F) + 10*(G-B) = 3*(F-A) + 10*(G-B) + 3*(H-C)
			//
			// For LSD
			// A B
			// C D
			// gx = (B-A) + (D-C)
			// gy = (C-A) + (D-B)
			//
			// To make this faster: 
			// com1 = (D-A)
			// com2 = (B-C)
			// Then: gx = com1 + com2 = (D-A) + (B-C) = (B-A) + (D-C)
			//       gy = com1 - com2 = (D-A) - (B-C) = (C-A) + (D-B)

			int com1 = smoothImg[(i + 1)*width + j + 1] - smoothImg[(i - 1)*width + j - 1];
			int com2 = smoothImg[(i - 1)*width + j + 1] - smoothImg[(i + 1)*width + j - 1];

			int gx;
			int gy;
			
			switch (op)
			{
			case PREWITT_OPERATOR:
				gx = abs(com1 + com2 + (smoothImg[i*width + j + 1] - smoothImg[i*width + j - 1]));
				gy = abs(com1 - com2 + (smoothImg[(i + 1)*width + j] - smoothImg[(i - 1)*width + j]));
				break;
			case SOBEL_OPERATOR:
				gx = abs(com1 + com2 + 2 * (smoothImg[i*width + j + 1] - smoothImg[i*width + j - 1]));
				gy = abs(com1 - com2 + 2 * (smoothImg[(i + 1)*width + j] - smoothImg[(i - 1)*width + j]));
				break;
			case SCHARR_OPERATOR:
				gx = abs(3 * (com1 + com2) + 10 * (smoothImg[i*width + j + 1] - smoothImg[i*width + j - 1]));
				gy = abs(3 * (com1 - com2) + 10 * (smoothImg[(i + 1)*width + j] - smoothImg[(i - 1)*width + j]));
				break;
			case LSD_OPERATOR:
				// com1 and com2 differs from previous operators, because LSD has 2x2 kernel
				int LSDcom1 = smoothImg[(i + 1)*width + j + 1] - smoothImg[i*width + j];
				int LSDcom2 = smoothImg[i*width + j + 1] - smoothImg[(i + 1)*width + j];
				gx = abs(LSDcom1 + LSDcom2);
				gy = abs(LSDcom1 - LSDcom2);
				break;
			}
			
			int sum;
			
			if(sumFlag)
				sum = gx + gy;
			else
				sum = (int)sqrt((double)gx*gx + gy*gy);

			int index = i*width + j;
			gradImg[index] = sum;

			if (sum >= gradThresh) {
				if (gx >= gy) dirImg[index] = EDGE_VERTICAL;
				else          dirImg[index] = EDGE_HORIZONTAL;
				

				//printf("x = %d\n",j);
				//printf("y = %d\n",i);
			} //end-if
		} // end-for
	} // end-for
}


void ED::ComputeAnchorPoints()
{
	//memset(edgeImg, 0, width*height);
	for (int i = 2; i<height - 2; i++) {
		int start = 2;
		int inc = 1;
		if (i%scanInterval != 0) { start = scanInterval; inc = scanInterval; }
		
		for (int j = start; j<width - 2; j += inc) {
			if (gradImg[i*width + j] < gradThresh) continue;

			if (dirImg[i*width + j] == EDGE_VERTICAL) {
				// vertical edge
				int diff1 = gradImg[i*width + j] - gradImg[i*width + j - 1];
				int diff2 = gradImg[i*width + j] - gradImg[i*width + j + 1];
				if (diff1 >= anchorThresh && diff2 >= anchorThresh) {
					edgeImg[i*width + j] = ANCHOR_PIXEL;
					anchorPoints.push_back(Point(j, i)); 
				}

			}
			else {
				// horizontal edge
				int diff1 = gradImg[i*width + j] - gradImg[(i - 1)*width + j];
				int diff2 = gradImg[i*width + j] - gradImg[(i + 1)*width + j];
				if (diff1 >= anchorThresh && diff2 >= anchorThresh) {
					edgeImg[i*width + j] = ANCHOR_PIXEL;
					anchorPoints.push_back(Point(j, i)); 
				}
			} // end-else
		} //end-for-inner
	} //end-for-outer

	anchorNos = (int)anchorPoints.size(); // get the total number of anchor points
}

void ED::OComputeAnchorPoints()
{
	//memset(edgeImg, 0, width*height);
	for (int i = 2; i<height - 2; i++) {
		int start = 2;
		int inc = 1;
		if (i%scanInterval != 0) { start = scanInterval; inc = scanInterval; }
		
		for (int j = start; j<width - 2; j += inc) {
			//if (gradImg[i*width + j] < gradThresh) continue;

			if (dirImg[i*width + j] == EDGE_VERTICAL) {
				// vertical edge
				int diff1 = selectgradImg[i*width + j] - selectgradImg[i*width + j - 1];
				int diff2 = selectgradImg[i*width + j] - selectgradImg[i*width + j + 1];
				if (diff1 >= anchorThresh && diff2 >= anchorThresh) {
					edgeImg[i*width + j] = ANCHOR_PIXEL;
					anchorPoints.push_back(Point(j, i)); 
				}
				
			}
			else if (dirImg[i*width + j] == EDGE_HORIZONTAL){
				// horizontal edge
				int diff1 = selectgradImg[i*width + j] - selectgradImg[(i - 1)*width + j];
				int diff2 = selectgradImg[i*width + j] - selectgradImg[(i + 1)*width + j];
				if (diff1 >= anchorThresh && diff2 >= anchorThresh) {
					edgeImg[i*width + j] = ANCHOR_PIXEL;
					anchorPoints.push_back(Point(j, i)); 
				}
				
			} // end-else
		} //end-for-inner
	} //end-for-outer

	anchorNos = (int)anchorPoints.size(); // get the total number of anchor points
}

/*
void ED::OComputeAnchorPoints()
{
	//memset(edgeImg, 0, width*height);
	for (int i = 2; i<height - 2; i++) {
		int start = 2;
		int inc = 1;
		if (i%scanInterval != 0) { start = scanInterval; inc = scanInterval; }
		
		for (int j = start; j<width - 2; j += inc) {
			if (gradImg[i*width + j] < gradThresh) continue;

			if (dirImg[i*width + j] == EDGE_VERTICAL) {
				// vertical edge
				int diff1 = gradImg[i*width + j] - gradImg[i*width + j - 1];
				int diff2 = gradImg[i*width + j] - gradImg[i*width + j + 1];
				if (diff1 >= anchorThresh && diff2 >= anchorThresh) {
					edgeImg[i*width + j] = ANCHOR_PIXEL;
					anchorPoints.push_back(Point(j, i)); 
				}

			}
			else {
				// horizontal edge
				int diff1 = gradImg[i*width + j] - gradImg[(i - 1)*width + j];
				int diff2 = gradImg[i*width + j] - gradImg[(i + 1)*width + j];
				if (diff1 >= anchorThresh && diff2 >= anchorThresh) {
					edgeImg[i*width + j] = ANCHOR_PIXEL;
					anchorPoints.push_back(Point(j, i)); 
				}
			} // end-else
		} //end-for-inner
	} //end-for-outer

	anchorNos = (int)anchorPoints.size(); // get the total number of anchor points
}
*/


void ED::ComputeLaneLineWidth()
{
	//memset(edgeImg, 0, width*height);

	
	//cv::Mat HomographyMatrix= H(0,359-130, 280,130-130, 337,130-130, 639,359-130, -188, -7000, -188,0, 188,0, 188,-7000);
	//
	
	/*
	double d1 = line_width(HomographyMatrix, -298, 208, 0.f);
	double d2 = line_width(HomographyMatrix, -298, 208, 0.f);
	printf("d1 = %2.2f, d2 = %2.2f\n", d1, d2);
	
	//-----------------25m----------------------------//
	3.739862280030613, -0.08851745041492516, -1154.035195103293;
	1.942214879979242e-15, 34.31228297333891, -1338.179035960218;
	2.115629703434065e-18, 0.02307103760814556, 1
	//-----------------70m----------------------------//
	4.787847503373763, 0.2234477945876702, -1479.605668016177;
 	2.080314581414045e-13, -354.1255327082849, -6.826855033730424e-11;
 	-2.192982456143311e-05, 0.04401041444704828, 1
	*/
    
	int w = 2;
	for (int i = 1; i<height - 1; i++) {
		int serch_en = 0, serch_start_j = 0;
		for (int j = 1; j<width - 1; j++) {
			if (gradImg[i*width + j] < gradThresh) continue;
			int index = i*width + j;
			int lum_diff1 = (smoothImg[index - 1 + w] - smoothImg[index - 1 - w]);
			int lum_diff2 = (smoothImg[index + w] - smoothImg[index - w]);
			int lum_diff3 = (smoothImg[index + 1 + w] - smoothImg[index + 1 - w]);
			int lum_diff4 = (smoothImg[index + 2 + w] - smoothImg[index + 2 - w]);
			int lum_threshold =  smoothImg[index] * 0.07 + 5; 
			//std::cout << "lum_threshold: " << lum_threshold << std::endl;
			if ((lum_diff1 >lum_threshold)
				&& (lum_diff2 > lum_threshold)
				&& (lum_diff3 > lum_threshold) // Smaller than 150
				&& (lum_diff4 > lum_threshold) // Smaller than 150
				&& (serch_en == 0))
				{
					serch_start_j = j;
					serch_en = 1;
				} 
			else if (serch_en == 1)
			{
				if (j - serch_start_j > 60)
				{
					serch_en = 0;
					j = serch_start_j + 1;
					serch_start_j = 0;
					continue;
				}
				else if((-lum_diff1 > lum_threshold)
				&& (-lum_diff2 > lum_threshold)
				&& (-lum_diff3 > lum_threshold) // Smaller than 150
				&& (-lum_diff4 > lum_threshold)
				) // Smaller than 150
				{
					double d1 = line_width(serch_start_j, i, 1.f);
					double d2 = line_width(j, i, 1.f);
					//double d1 = line_width(HomographyMatrix, 34, 208, 0.f);
					//double d2 = line_width(HomographyMatrix, 50, 208, 0.f);
					//printf("d1 = %2.2f, d2 = %2.2f\n", d1, d2);
					//int lane_w= j - serch_start_j;
					double lane_w= abs(d2 - d1);
					if (lane_w >8.f && lane_w <32.f)
					{
						selectgradImg[(i-1)*width + serch_start_j + 1] = gradImg[(i-1)*width + serch_start_j + 1];
						selectgradImg[(i-1)*width + serch_start_j]     = gradImg[(i-1)*width + serch_start_j];
						selectgradImg[(i-1)*width + serch_start_j - 1] = gradImg[(i-1)*width + serch_start_j - 1];
						selectgradImg[i*width + serch_start_j + 1]     = gradImg[i*width + serch_start_j + 1];
						selectgradImg[i*width + serch_start_j] 		   = gradImg[i*width + serch_start_j];
						selectgradImg[i*width + serch_start_j - 1] 	   = gradImg[i*width + serch_start_j - 1];
						selectgradImg[(i+1)*width + serch_start_j + 1] = gradImg[(i+1)*width + serch_start_j + 1];
						selectgradImg[(i+1)*width + serch_start_j]     = gradImg[(i+1)*width + serch_start_j];
						selectgradImg[(i+1)*width + serch_start_j - 1] = gradImg[(i+1)*width + serch_start_j - 1];
						//j += 1;
						serch_start_j = 0;
						serch_en = 0;
					}
					//
					//serch_start_j = 0;
					//serch_en = 0;
				}
			}
		}
	}
}


void ED::JoinAnchorPointsUsingSortedAnchors()
{
	int *chainNos = new int[(width + height) * 8];

	Point *pixels = new Point[width*height];
	StackNode *stack = new StackNode[width*height];
	Chain *chains = new Chain[width*height];

	// sort the anchor points by their gradient value in decreasing order
	int *A = sortAnchorsByGradValue1();

	// Now join the anchors starting with the anchor having the greatest gradient value
	int totalPixels = 0;

	for (int k = anchorNos - 1; k >= 0; k--) {
		int pixelOffset = A[k];

		int i = pixelOffset / width;
		int j = pixelOffset % width;

		//int i = anchorPoints[k].y;
		//int j = anchorPoints[k].x;

		if (edgeImg[i*width + j] != ANCHOR_PIXEL) continue;
		//if (dirImg[i*width + j] != EDGE_VERTICAL && dirImg[i*width + j] != EDGE_HORIZONTAL) continue;


		chains[0].len = 0;
		chains[0].parent = -1;
		chains[0].dir = 0;
		chains[0].children[0] = chains[0].children[1] = -1;
		chains[0].pixels = NULL;


		int noChains = 1;
		int len = 0;
		int duplicatePixelCount = 0;
		int top = -1;  // top of the stack 

		if (dirImg[i*width + j] == EDGE_VERTICAL) {
			stack[++top].r = i;
			stack[top].c = j;
			stack[top].dir = DOWN;
			stack[top].parent = 0;

			stack[++top].r = i;
			stack[top].c = j;
			stack[top].dir = UP;
			stack[top].parent = 0;

		}
		else if(dirImg[i*width + j] == EDGE_HORIZONTAL) {
			
			stack[++top].r = i;
			stack[top].c = j;
			stack[top].dir = RIGHT;
			stack[top].parent = 0;

			stack[++top].r = i;
			stack[top].c = j;
			stack[top].dir = LEFT;
			stack[top].parent = 0;
		} //end-else
		
		  // While the stack is not empty
	StartOfWhile:
		while (top >= 0) {
			int r = stack[top].r;
			int c = stack[top].c;
			int dir = stack[top].dir;
			int parent = stack[top].parent;
			top--;

			if (edgeImg[r*width + c] != EDGE_PIXEL) duplicatePixelCount++;

			chains[noChains].dir = dir;   // traversal direction
			chains[noChains].parent = parent;
			chains[noChains].children[0] = chains[noChains].children[1] = -1;


			int chainLen = 0;

			chains[noChains].pixels = &pixels[len];

			pixels[len].y = r;
			pixels[len].x = c;
			len++;
			chainLen++;
			
			if (dir == LEFT) {
				while (dirImg[r*width + c] == EDGE_HORIZONTAL) {
					
					edgeImg[r*width + c] = EDGE_PIXEL;

					// The edge is horizontal. Look LEFT
					//
					//   A
					//   B x 
					//   C 
					//
					// cleanup up & down pixels
					if (edgeImg[(r - 1)*width + c] == ANCHOR_PIXEL) edgeImg[(r - 1)*width + c] = 0;
					if (edgeImg[(r + 1)*width + c] == ANCHOR_PIXEL) edgeImg[(r + 1)*width + c] = 0;

					// Look if there is an edge pixel in the neighbors
					if (edgeImg[r*width + c - 1] >= ANCHOR_PIXEL) { c--; }
					else if (edgeImg[(r - 1)*width + c - 1] >= ANCHOR_PIXEL) { r--; c--; }
					else if (edgeImg[(r + 1)*width + c - 1] >= ANCHOR_PIXEL) { r++; c--; }
					else {
						// else -- follow max. pixel to the LEFT
						int A = selectgradImg[(r - 1)*width + c - 1];
						int B = selectgradImg[r*width + c - 1];
						int C = selectgradImg[(r + 1)*width + c - 1];

						if (A > B) {
							if (A > C) r--;
							else       r++;
						}
						else  if (C > B) r++;
						c--;
					} //end-else

					if (edgeImg[r*width + c] == EDGE_PIXEL || selectgradImg[r*width + c] < gradThresh) {
						if (chainLen > 0) {
							chains[noChains].len = chainLen;
							chains[parent].children[0] = noChains;
							noChains++;
						} // end-if
						goto StartOfWhile;
					} //end-else


					pixels[len].y = r;
					pixels[len].x = c;
					len++;
					chainLen++;
					
				} //end-while
				//printf("top1 = %d\n", top);
				stack[++top].r = r;
				stack[top].c = c;
				stack[top].dir = DOWN;
				stack[top].parent = noChains;
				//printf("top2 = %d\n", top);
				stack[++top].r = r;
				stack[top].c = c;
				stack[top].dir = UP;
				stack[top].parent = noChains;
				//printf("top3 = %d\n", top);
				len--;
				chainLen--;

				chains[noChains].len = chainLen;
				chains[parent].children[0] = noChains;
				noChains++;

			}
			else if (dir == RIGHT) {
				while (dirImg[r*width + c] == EDGE_HORIZONTAL) {
					edgeImg[r*width + c] = EDGE_PIXEL;

					// The edge is horizontal. Look RIGHT
					//
					//     A
					//   x B
					//     C
					//
					// cleanup up&down pixels
					if (edgeImg[(r + 1)*width + c] == ANCHOR_PIXEL) edgeImg[(r + 1)*width + c] = 0;
					if (edgeImg[(r - 1)*width + c] == ANCHOR_PIXEL) edgeImg[(r - 1)*width + c] = 0;

					// Look if there is an edge pixel in the neighbors
					if (edgeImg[r*width + c + 1] >= ANCHOR_PIXEL) { c++; }
					else if (edgeImg[(r + 1)*width + c + 1] >= ANCHOR_PIXEL) { r++; c++; }
					else if (edgeImg[(r - 1)*width + c + 1] >= ANCHOR_PIXEL) { r--; c++; }
					else {
						// else -- follow max. pixel to the RIGHT
						int A = selectgradImg[(r - 1)*width + c + 1];
						int B = selectgradImg[r*width + c + 1];
						int C = selectgradImg[(r + 1)*width + c + 1];

						if (A > B) {
							if (A > C) r--;       // A
							else       r++;       // C
						}
						else if (C > B) r++;  // C
						c++;
					} //end-else

					if (edgeImg[r*width + c] == EDGE_PIXEL || selectgradImg[r*width + c] < gradThresh) {
						if (chainLen > 0) {
							chains[noChains].len = chainLen;
							chains[parent].children[1] = noChains;
							noChains++;
						} // end-if
						goto StartOfWhile;
					} //end-else


					pixels[len].y = r;
					pixels[len].x = c;
					len++;
					chainLen++;
				} //end-while

				stack[++top].r = r;
				stack[top].c = c;
				stack[top].dir = DOWN;  // Go down
				stack[top].parent = noChains;

				stack[++top].r = r;
				stack[top].c = c;
				stack[top].dir = UP;   // Go up
				stack[top].parent = noChains;

				len--;
				chainLen--;

				chains[noChains].len = chainLen;
				chains[parent].children[1] = noChains;
				noChains++;

			}

			else if (dir == UP) {
				//if(dirImg[r*width + c] != EDGE_VERTICAL) continue;
				while (dirImg[r*width + c] == EDGE_VERTICAL) {
					edgeImg[r*width + c] = EDGE_PIXEL;

					// The edge is vertical. Look UP
					//
					// A B C D E
					//   A B C
					//     x
					//
					// Cleanup left & right pixels
					if (edgeImg[r*width + c - 1] == ANCHOR_PIXEL) edgeImg[r*width + c - 1] = 0;
					if (edgeImg[r*width + c + 1] == ANCHOR_PIXEL) edgeImg[r*width + c + 1] = 0;

					// Look if there is an edge pixel in the neighbors
					if (edgeImg[(r - 1)*width + c] >= ANCHOR_PIXEL) { r--; }
					else if (edgeImg[(r - 1)*width + c - 1] >= ANCHOR_PIXEL) { r--; c--; }
					else if (edgeImg[(r - 1)*width + c + 1] >= ANCHOR_PIXEL) { r--; c++; }
					else {
						// else -- follow the max. pixel UP
						int A = selectgradImg[(r - 1)*width + c - 1];
						int B = selectgradImg[(r - 1)*width + c];
						int C = selectgradImg[(r - 1)*width + c + 1];

						if (A > B) {
							if (A > C) c--;
							else       c++;
						}
						else if (C > B) c++;
						r--;
					} //end-else

					if (edgeImg[r*width + c] == EDGE_PIXEL || selectgradImg[r*width + c] < gradThresh) {
						if (chainLen > 0) {
							chains[noChains].len = chainLen;
							chains[parent].children[0] = noChains;
							noChains++;
						} // end-if
						goto StartOfWhile;
					} //end-else


					pixels[len].y = r;
					pixels[len].x = c;

					len++;
					chainLen++;
				} //end-while

				stack[++top].r = r;
				stack[top].c = c;
				stack[top].dir = RIGHT;
				stack[top].parent = noChains;

				stack[++top].r = r;
				stack[top].c = c;
				stack[top].dir = LEFT;
				stack[top].parent = noChains;

				len--;
				chainLen--;

				chains[noChains].len = chainLen;
				chains[parent].children[0] = noChains;
				noChains++;
				//printf("len =%d \n", len);
			}
			else if (dir == DOWN){ // dir == DOWN
				//if(dirImg[r*width + c] != EDGE_VERTICAL) continue;
				while (dirImg[r*width + c] == EDGE_VERTICAL) {
					edgeImg[r*width + c] = EDGE_PIXEL;

					// The edge is vertical
					//
					//     x
					//   A B C
					//
					// cleanup side pixels
					if (edgeImg[r*width + c + 1] == ANCHOR_PIXEL) edgeImg[r*width + c + 1] = 0;
					if (edgeImg[r*width + c - 1] == ANCHOR_PIXEL) edgeImg[r*width + c - 1] = 0;

					// Look if there is an edge pixel in the neighbors
					if (edgeImg[(r + 1)*width + c] >= ANCHOR_PIXEL) { r++; }
					else if (edgeImg[(r + 1)*width + c + 1] >= ANCHOR_PIXEL) { r++; c++; }
					else if (edgeImg[(r + 1)*width + c - 1] >= ANCHOR_PIXEL) { r++; c--; }
					else {
						// else -- follow the max. pixel DOWN
						int A = selectgradImg[(r + 1)*width + c - 1];
						int B = selectgradImg[(r + 1)*width + c];
						int C = selectgradImg[(r + 1)*width + c + 1];

						if (A > B) {
							if (A > C) c--;       // A
							else       c++;       // C
						}
						else if (C > B) c++;  // C
						r++;
					} //end-else

					if (edgeImg[r*width + c] == EDGE_PIXEL || selectgradImg[r*width + c] < gradThresh) {
						if (chainLen > 0) {
							chains[noChains].len = chainLen;
							chains[parent].children[1] = noChains;
							noChains++;
						} // end-if
						goto StartOfWhile;
					} //end-else

					pixels[len].y = r;
					pixels[len].x = c;

					len++;
					chainLen++;
				} //end-while

				stack[++top].r = r;
				stack[top].c = c;
				stack[top].dir = RIGHT;
				stack[top].parent = noChains;

				stack[++top].r = r;
				stack[top].c = c;
				stack[top].dir = LEFT;
				stack[top].parent = noChains;

				len--;
				chainLen--;

				chains[noChains].len = chainLen;
				chains[parent].children[1] = noChains;
				noChains++;
			} //end-else

		} //end-while

		if (len - duplicatePixelCount < minPathLen) {
			for (int k = 0; k<len; k++) {

				edgeImg[pixels[k].y*width + pixels[k].x] = 0;
				edgeImg[pixels[k].y*width + pixels[k].x] = 0;

			} //end-for

		}
		else {

			int noSegmentPixels = 0;

			int totalLen = LongestChain(chains, chains[0].children[1]);

			if (totalLen > 0) {
				// Retrieve the chainNos
				int count = RetrieveChainNos(chains, chains[0].children[1], chainNos);

				// Copy these pixels in the reverse order
				for (int k = count - 1; k >= 0; k--) {
					int chainNo = chainNos[k];

#if 1
					/* See if we can erase some pixels from the last chain. This is for cleanup */

					int fr = chains[chainNo].pixels[chains[chainNo].len - 1].y;
					int fc = chains[chainNo].pixels[chains[chainNo].len - 1].x;

					int index = noSegmentPixels - 2;
					while (index >= 0) {
						int dr = abs(fr - segmentPoints[segmentNos][index].y);
						int dc = abs(fc - segmentPoints[segmentNos][index].x);

						if (dr <= 1 && dc <= 1) {
							// neighbors. Erase last pixel
							segmentPoints[segmentNos].pop_back();
							noSegmentPixels--;
							index--;
						}
						else break;
					} //end-while

					if (chains[chainNo].len > 1 && noSegmentPixels > 0) {
						fr = chains[chainNo].pixels[chains[chainNo].len - 2].y;
						fc = chains[chainNo].pixels[chains[chainNo].len - 2].x;

						int dr = abs(fr - segmentPoints[segmentNos][noSegmentPixels - 1].y);
						int dc = abs(fc - segmentPoints[segmentNos][noSegmentPixels - 1].x);

						if (dr <= 1 && dc <= 1) chains[chainNo].len--;
					} //end-if
#endif

					for (int l = chains[chainNo].len - 1; l >= 0; l--) {
						segmentPoints[segmentNos].push_back(chains[chainNo].pixels[l]);
						noSegmentPixels++;
					} //end-for

					chains[chainNo].len = 0;  // Mark as copied
				} //end-for
			} //end-if

			totalLen = LongestChain(chains, chains[0].children[0]);
			if (totalLen > 1) {
				// Retrieve the chainNos
				int count = RetrieveChainNos(chains, chains[0].children[0], chainNos);

				// Copy these chains in the forward direction. Skip the first pixel of the first chain
				// due to repetition with the last pixel of the previous chain
				int lastChainNo = chainNos[0];
				chains[lastChainNo].pixels++;
				chains[lastChainNo].len--;

				for (int k = 0; k<count; k++) {
					int chainNo = chainNos[k];

#if 1
					/* See if we can erase some pixels from the last chain. This is for cleanup */
					int fr = chains[chainNo].pixels[0].y;
					int fc = chains[chainNo].pixels[0].x;

					int index = noSegmentPixels - 2;
					while (index >= 0) {
						int dr = abs(fr - segmentPoints[segmentNos][index].y);
						int dc = abs(fc - segmentPoints[segmentNos][index].x);

						if (dr <= 1 && dc <= 1) {
							// neighbors. Erase last pixel
							segmentPoints[segmentNos].pop_back();
							noSegmentPixels--;
							index--;
						}
						else break;
					} //end-while

					int startIndex = 0;
					int chainLen = chains[chainNo].len;
					if (chainLen > 1 && noSegmentPixels > 0) {
						int fr = chains[chainNo].pixels[1].y;
						int fc = chains[chainNo].pixels[1].x;

						int dr = abs(fr - segmentPoints[segmentNos][noSegmentPixels - 1].y);
						int dc = abs(fc - segmentPoints[segmentNos][noSegmentPixels - 1].x);

						if (dr <= 1 && dc <= 1) { startIndex = 1; }
					} //end-if
#endif

					  /* Start a new chain & copy pixels from the new chain */
					for (int l = startIndex; l<chains[chainNo].len; l++) {
						segmentPoints[segmentNos].push_back(chains[chainNo].pixels[l]);
						noSegmentPixels++;
					} //end-for

					chains[chainNo].len = 0;  // Mark as copied
				} //end-for
			} //end-if


			  // See if the first pixel can be cleaned up
			int fr = segmentPoints[segmentNos][1].y;
			int fc = segmentPoints[segmentNos][1].x;


			int dr = abs(fr - segmentPoints[segmentNos][noSegmentPixels - 1].y);
			int dc = abs(fc - segmentPoints[segmentNos][noSegmentPixels - 1].x);


			if (dr <= 1 && dc <= 1) {
				segmentPoints[segmentNos].erase(segmentPoints[segmentNos].begin());
				noSegmentPixels--;
			} //end-if

			segmentNos++;
			segmentPoints.push_back(vector<Point>()); // create empty vector of points for segments

													  // Copy the rest of the long chains here
			for (int k = 2; k<noChains; k++) {
				if (chains[k].len < 2) continue;

				totalLen = LongestChain(chains, k);

				if (totalLen >= 10) {

					// Retrieve the chainNos
					int count = RetrieveChainNos(chains, k, chainNos);

					// Copy the pixels
					noSegmentPixels = 0;
					for (int k = 0; k<count; k++) {
						int chainNo = chainNos[k];

#if 1					
						/* See if we can erase some pixels from the last chain. This is for cleanup */
						int fr = chains[chainNo].pixels[0].y;
						int fc = chains[chainNo].pixels[0].x;

						int index = noSegmentPixels - 2;
						while (index >= 0) {
							int dr = abs(fr - segmentPoints[segmentNos][index].y);
							int dc = abs(fc - segmentPoints[segmentNos][index].x);

							if (dr <= 1 && dc <= 1) {
								// neighbors. Erase last pixel
								segmentPoints[segmentNos].pop_back();
								noSegmentPixels--;
								index--;
							}
							else break;
						} //end-while

						int startIndex = 0;
						int chainLen = chains[chainNo].len;
						if (chainLen > 1 && noSegmentPixels > 0) {
							int fr = chains[chainNo].pixels[1].y;
							int fc = chains[chainNo].pixels[1].x;

							int dr = abs(fr - segmentPoints[segmentNos][noSegmentPixels - 1].y);
							int dc = abs(fc - segmentPoints[segmentNos][noSegmentPixels - 1].x);

							if (dr <= 1 && dc <= 1) { startIndex = 1; }
						} //end-if
#endif
						  /* Start a new chain & copy pixels from the new chain */
						for (int l = startIndex; l<chains[chainNo].len; l++) {
							segmentPoints[segmentNos].push_back(chains[chainNo].pixels[l]);
							noSegmentPixels++;
						} //end-for

						chains[chainNo].len = 0;  // Mark as copied
					} //end-for
					segmentPoints.push_back(vector<Point>()); // create empty vector of points for segments
					segmentNos++;
				} //end-if          
			} //end-for

		} //end-else

	} //end-for-outer

	// pop back last segment from vector
	// because of one preallocation in the beginning, it will always empty
	segmentPoints.pop_back();

	// Clean up
	delete[] A;
	delete[] chains;
	delete[] stack;
	delete[] chainNos;
	delete[] pixels;
}

void ED::sortAnchorsByGradValue()
{
	auto sortFunc = [&](const Point &a, const Point &b)
	{
		return selectgradImg[a.y * width + a.x] > selectgradImg[b.y * width + b.x];
	};
	
	std::sort(anchorPoints.begin(), anchorPoints.end(), sortFunc);

	/*
	ofstream myFile;
	myFile.open("anchorsNew.txt");
	for (int i = 0; i < anchorPoints.size(); i++) {
		int x = anchorPoints[i].x;
		int y = anchorPoints[i].y;

		myFile << i << ". value: " << gradImg[y*width + x] << "  Cord: (" << x << "," << y << ")" << endl;
	}
	myFile.close(); 
	
	
	vector<Point> temp(anchorNos);

	int x, y, i = 0;
	char c;
	std::ifstream infile("cords.txt");
	while (infile >> x >> c >> y && c == ',') {
		temp[i] = Point(x, y);
		i++;
	} 

	anchorPoints = temp; 
	*/
}

int * ED::sortAnchorsByGradValue1()
{
	int SIZE = 128 * 256;
	int *C = new int[SIZE];
	memset(C, 0, sizeof(int)*SIZE);

	// Count the number of grad values
	for (int i = 1; i<height - 1; i++) {
		for (int j = 1; j<width - 1; j++) {
			if (edgeImg[i*width + j] != ANCHOR_PIXEL) continue;

			int grad = selectgradImg[i*width + j];
			C[grad]++;
		} //end-for
	} //end-for 

	// Compute indices
	for (int i = 1; i<SIZE; i++) C[i] += C[i - 1];

	int noAnchors = C[SIZE - 1];
	int *A = new int[noAnchors];
	memset(A, 0, sizeof(int)*noAnchors);


	for (int i = 1; i<height - 1; i++) {
		for (int j = 1; j<width - 1; j++) {
			if (edgeImg[i*width + j] != ANCHOR_PIXEL) continue;

			int grad = selectgradImg[i*width + j];
			int index = --C[grad];
			A[index] = i*width + j;    // anchor's offset 
		} //end-for
	} //end-for  

	delete[] C;

	/*
	ofstream myFile;
	myFile.open("aNew.txt");
	for (int i = 0; i < noAnchors; i++)
		myFile << A[i] << endl;

	myFile.close(); */

	return A;

}


int ED::LongestChain(Chain *chains, int root) {
	if (root == -1 || chains[root].len == 0) return 0;

	int len0 = 0;
	if (chains[root].children[0] != -1) len0 = LongestChain(chains, chains[root].children[0]);

	int len1 = 0;
	if (chains[root].children[1] != -1) len1 = LongestChain(chains, chains[root].children[1]);

	int max = 0;

	if (len0 >= len1) {
		max = len0;
		chains[root].children[1] = -1;

	}
	else {
		max = len1;
		chains[root].children[0] = -1;
	} //end-else

	return chains[root].len + max;
} //end-LongestChain

int ED::RetrieveChainNos(Chain * chains, int root, int chainNos[])
{
	int count = 0;

	while (root != -1) {
		chainNos[count] = root;
		count++;

		if (chains[root].children[0] != -1) root = chains[root].children[0];
		else                                root = chains[root].children[1];
	} //end-while

	return count;
}


cv::Mat ED::H(int ps1x, int ps1y, int ps2x, int ps2y, int ps3x, int ps3y, int ps4x, int ps4y,
					int pd1x, int pd1y, int pd2x, int pd2y, int pd3x, int pd3y, int pd4x, int pd4y) {

	std::vector<cv::Point2f> pts_src;
 	pts_src.push_back(Point2f(ps1x, ps1y));
	pts_src.push_back(Point2f(ps2x, ps2y));
	pts_src.push_back(Point2f(ps3x, ps3y));
	pts_src.push_back(Point2f(ps4x, ps4y));

	std::vector<cv::Point2f> pts_dst;
	pts_dst.push_back(Point2f(pd1x, pd1y));
	pts_dst.push_back(Point2f(pd2x, pd2y));
	pts_dst.push_back(Point2f(pd3x, pd3y));
	pts_dst.push_back(Point2f(pd4x, pd4y));
	
	line_width_h = findHomography(pts_src, pts_dst);
	return line_width_h;
}

double ED::line_width(double x_data, double y_data, double z_data) {

    m.at<double>(0,0)=x_data; //test point x
    m.at<double>(1,0)=y_data;//test point y
    m.at<double>(2,0)=z_data; ////test point z set 1
	cv::Mat res=line_width_h*m;
	double x=res.at<double>(0,0);
	double y=res.at<double>(0,1);
	double z=res.at<double>(0,2);
	double distance_x=x/z ;
	double distance_y=y/z ;
	//double distance=sqrt(distance_x*distance_x+distance_y*distance_y);
	return distance_x;
}