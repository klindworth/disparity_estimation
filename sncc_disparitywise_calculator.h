#ifndef SNCC_CALCULATOR_H
#define SNCC_CALCULATOR_H

#include <opencv2/core/core.hpp>

class sncc_disparitywise_calculator
{
public:
	typedef float result_type;
	sncc_disparitywise_calculator(const cv::Mat& pbase, const cv::Mat& pmatch);
	cv::Mat_<float> operator()(int d);

private:
	cv::Mat_<float> base_float, match_float;
	cv::Mat_<float> mu_base, mu_match, sigma_base_inv, sigma_match_inv;
	std::vector<cv::Mat_<float> > temp;
};

#endif // SNCC_CALCULATOR_H
