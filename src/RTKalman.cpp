#include "RTKalman.h"
#include <fstream>

using namespace cv;
using namespace std;

void RTKalman::init_kf()
{
	int stateNum = 4;
	int measureNum = 2;
	kf = KalmanFilter(stateNum, measureNum, 0);

	measurement = Mat::zeros(measureNum, 1, CV_32F);

//	kf.transitionMatrix = *(Mat_<float>(stateNum, stateNum) <<
	kf.transitionMatrix = (Mat_<float>(stateNum, stateNum) <<
		1, 0, 1, 0,
		0, 1, 0, 1,
		0, 0, 1, 0,
		0, 0, 0, 1);

	setIdentity(kf.measurementMatrix); //H
	setIdentity(kf.processNoiseCov, Scalar::all(0.05)); //Q
	setIdentity(kf.measurementNoiseCov, Scalar::all(1)); //R
	setIdentity(kf.errorCovPost, Scalar::all(100)); //P
	
	randn(kf.statePost, Scalar::all(-90.f), Scalar::all(-90.f));
}

void RTKalman::init_kf(StateType stateMat)
{
	int stateNum = 4;
	int measureNum = 2;
	kf = KalmanFilter(stateNum, measureNum, 0);

	measurement = Mat::zeros(measureNum, 1, CV_32F);

//	kf.transitionMatrix = *(Mat_<float>(stateNum, stateNum) <<
	kf.transitionMatrix = (Mat_<float>(stateNum, stateNum) <<
		1, 0, 1, 0,
		0, 1, 0, 1,
		0, 0, 1, 0,
		0, 0, 0, 1);

	kf.measurementMatrix = (Mat_<float>(measureNum, stateNum) <<
		1, 0, 0, 0,
		0, 1, 0, 0);

	kf.processNoiseCov = (Mat_<float>(stateNum, stateNum) <<
		0.05, 0, 0, 0,
		0, 0.05, 0, 0,
		0, 0, 0.05, 0,
		0, 0, 0, 0.05);

	kf.measurementNoiseCov = (Mat_<float>(measureNum, measureNum) <<
		1, 0,
		0, 1);

	kf.processNoiseCov = (Mat_<float>(stateNum, stateNum) <<
		100, 0, 0, 0,
		0, 100, 0, 0,
		0, 0, 100, 0,
		0, 0, 0, 100);

	//setIdentity(kf.measurementMatrix); //H
	//setIdentity(kf.processNoiseCov, Scalar::all(0.05)); //Q
	//setIdentity(kf.measurementNoiseCov, Scalar::all(1)); //R
	//setIdentity(kf.errorCovPost, Scalar::all(100)); //P
	
	// initialize state vector with bounding box in [r,theta,u,w] style
	kf.statePost.at<float>(0, 0) = stateMat.r;
	kf.statePost.at<float>(1, 0) = stateMat.theta;
	//kf.statePost.at<float>(2, 0) = stateMat.radial;
	//kf.statePost.at<float>(3, 0) = stateMat.angular;
}

// Predict the estimated bounding box.
cv::Mat RTKalman::predict()
{
	// predict
	Mat p = kf.predict();
	return p;
}

// Update the state vector with observed bounding box.
void RTKalman::update(StateType stateMat)
{
	// measurement
	//randn(measurement, Scalar::all(0.f), Scalar::all(1.f));
	measurement.at<float>(0, 0) = stateMat.r;
	measurement.at<float>(1, 0) = stateMat.theta;
	// update
	kf.correct(measurement);
}

cv::Mat RTKalman::get_state()
{
	Mat s = kf.statePost;
	return s;
}

