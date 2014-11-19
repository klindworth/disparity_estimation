#ifndef SNCC_CALCULATOR_H
#define SNCC_CALCULATOR_H

#include <opencv2/core/core.hpp>

struct sncc_task_cache
{
	sncc_task_cache(int cols) : coltemp(3, std::vector<float>(cols+2)), box_temp(cols), boxcol_temp(cols+2), replace_idx(2) {}
	std::vector<std::vector<float>> coltemp;
	std::vector<float> box_temp;
	std::vector<float> boxcol_temp;
	int replace_idx;
};

class sncc_disparitywise_calculator
{
public:
	typedef float result_type;
	sncc_disparitywise_calculator(const cv::Mat& pbase, const cv::Mat& pmatch);
	cv::Mat_<float> operator()(int d);

private:
	cv::Mat_<float> base_float, match_float;
	cv::Mat_<float> mu_base, mu_match, sigma_base_inv, sigma_match_inv;
	std::vector<sncc_task_cache> cache;
};

#endif // SNCC_CALCULATOR_H
