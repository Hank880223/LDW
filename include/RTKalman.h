/**************************************************************************************************************
* Edge Drawing (ED) and Edge Drawing Parameter Free (EDPF) source codes.
* Copyright (C) Cihan Topal & Cuneyt Akinlar 
* E-mails of the authors:  cihantopal@gmail.com, cuneytakinlar@gmail.com
*
* Please cite the following papers if you use Edge Drawing library:
*
* [1] C. Topal and C. Akinlar, “Edge Drawing: A Combined Real-Time Edge and Segment Detector,”
*     Journal of Visual Communication and Image Representation, 23(6), 862-872, DOI: 10.1016/j.jvcir.2012.05.004 (2012).
*
* [2] C. Akinlar and C. Topal, “EDPF: A Real-time Parameter-free Edge Segment Detector with a False Detection Control,”
*     International Journal of Pattern Recognition and Artificial Intelligence, 26(1), DOI: 10.1142/S0218001412550026 (2012).
**************************************************************************************************************/

#ifndef _RTKalman_
#define _RTKalman_

#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <cmath>

using namespace std;
using namespace cv;

#define StateType RTK
#define KALMAN_Q 0.02      //Ｑ越大，速度越快，收斂穩定性不好
#define KALMAN_R 1.0000     //Ｒ越大，速度越慢，收斂穩定性很好
//標量卡爾曼濾波
typedef struct {
    float x;  // 系統的狀態量
    float A;  // x(n)=A*x(n-1)+u(n),u(n)~N(0,q)
    float H;  // z(n)=H*x(n)+w(n),w(n)~N(0,r)
    float q;  // 預測過程噪聲協方差
    float r;  // 測量過程噪聲協方差
    float p;  // 估計誤差協方差
    float gain;//卡爾曼增益
}KalmanStructTypedef;

struct RTK {
	float r;
	float theta;
	//float radial;
	//float angular;
	RTK(float _r, float _theta)
	{
		r = _r;
		theta = _theta;
		//radial = _radial;
		//angular = _angular;
	}
};

class RTKalman {
							
public:
	
	RTKalman()
	{
		init_kf();
		m_time_since_update = 0;
		m_hits = 0;
		m_hit_streak = 0;
		m_age = 0;
		//m_id = kf_count;
		//kf_count++;
	}

	RTKalman(StateType init)
	{
		init_kf(init);
		m_time_since_update = 0;
		m_hits = 0;
		m_hit_streak = 0;
		m_age = 0;
		//m_id = kf_count;
		//kf_count++;
	}

	~RTKalman()
	{
	}

	cv::Mat predict();
	void update(StateType stateMat);
	cv::Mat get_state();

	static int Rkf_count;
	static int Lkf_count;
	cv::Mat Lstate = cv::Mat::zeros(2, 1, CV_32FC1);
	cv::Mat Rstate = cv::Mat::zeros(2, 1, CV_32FC1);
	int m_time_since_update;
	int m_hits;
	int m_hit_streak;
	int m_age;
	int m_id;
	
private:
	void init_kf();
	void init_kf(StateType stateMat);

	cv::KalmanFilter kf;
	cv::Mat measurement;


};


#endif
