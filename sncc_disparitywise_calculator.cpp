#include "sncc_disparitywise_calculator.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <omp.h>

#include "disparitywise_calculator.h"

sncc_disparitywise_calculator::sncc_disparitywise_calculator(const cv::Mat& pbase, const cv::Mat& pmatch)
{
	long long start = cv::getCPUTickCount();
	cv::Size box_size(3,3);

	cv::Mat_<float> base_float_org;
	pbase.convertTo(base_float_org, CV_32F);
	cv::copyMakeBorder(base_float_org, base_float, 1,1,1,1, cv::BORDER_DEFAULT);
	cv::Mat base_square;
	cv::pow(base_float_org, 2, base_square);
	cv::boxFilter(base_float_org, mu_base, -1, box_size);
	cv::Mat mu_base_sq;
	cv::pow(mu_base, 2, mu_base_sq);
	cv::Mat sq_base_box;
	cv::boxFilter(base_square, sq_base_box, -1, box_size);
	cv::sqrt(sq_base_box - mu_base_sq, sigma_base_inv);
	cv::pow(sigma_base_inv, -1, sigma_base_inv);

	cv::Mat match_float_org;
	pmatch.convertTo(match_float_org, CV_32F);
	cv::copyMakeBorder(match_float_org, match_float, 1,1,1,1, cv::BORDER_DEFAULT);
	cv::Mat match_square;
	cv::pow(match_float_org, 2, match_square);
	cv::boxFilter(match_float_org, mu_match, -1, box_size);
	cv::Mat mu_match_sq;
	cv::pow(mu_match, 2, mu_match_sq);
	cv::Mat sq_match_box;
	cv::boxFilter(match_square, sq_match_box, -1, box_size);
	cv::sqrt(sq_match_box - mu_match_sq, sigma_match_inv);
	cv::pow(sigma_match_inv, -1, sigma_match_inv);

	int threads = omp_get_max_threads();
	//int threads = 1;
	for(int i = 0; i < threads; ++i)
		temp.emplace_back(base_float.size());

	std::cout << "init: " << cv::getCPUTickCount() - start << std::endl;
}

void preparation_kernel(float* temp, const float* base, const float* match, int cols, int rows, int row_stride)
{
	for(int y = 0; y < rows; ++y)
	{
		const float* base_d_ptr = base + y*row_stride;
		const float* match_d_ptr = match + y*row_stride;
		for(int x = 0; x < cols; ++x)
			*temp++ = *base_d_ptr++ * *match_d_ptr++;
	}
}

void sncc_kernel(float* result, const float* temp, const float* mu_base, const float* mu_match, const float* sigma_base_inv, const float* sigma_match_inv, int rows, int cols, int row_stride)
{
	const float norm_factor = 1.0/9.0f;
	float* result_ptr = result;
	for(int y = 0; y < rows; ++y)
	{
		int y_offset = y*row_stride;
		//float* result_ptr = result + y_offset;
		const float* mu_base_ptr = mu_base + y_offset;
		const float* mu_match_ptr = mu_match + y_offset;
		const float* sigma_base_inv_ptr = sigma_base_inv + y_offset;
		const float* sigma_match_inv_ptr = sigma_match_inv + y_offset;

		for(int x = 0; x < cols; ++x)
		{
			float sum = 0.0f;
			for(int dy = 0; dy < 3; ++dy)
			{
				const float *temp_ptr = temp + (y+dy)*(cols+2)+x;
				for(int dx = 0; dx < 3; ++dx)
					sum += *temp_ptr++;
			}
			sum *= norm_factor;

			*result_ptr++ = 1.0f - (sum - *mu_base_ptr++ * *mu_match_ptr++) * *sigma_base_inv_ptr++ * *sigma_match_inv_ptr++;
		}
	}
}

cv::Mat_<float> sncc_disparitywise_calculator::operator()(int d)
{
	const int offset_base = calculate_base_offset(d);
	const int offset_match = calculate_match_offset(d);
	const int row_length = calculate_row_length(base_float, d) - 2;
	const int rows = base_float.rows - 2;

	//cv::Mat_<float> result = prepare_result(cv::Size(row_length, rows), d, 100.0f);
	cv::Mat_<float> result(cv::Size(row_length, rows), 100.0f);

	int thread_idx = omp_get_thread_num();
	preparation_kernel(temp[thread_idx][0], base_float.ptr<float>() + offset_base, match_float.ptr<float>() + offset_match, row_length+2, rows+2, base_float.cols);
	//sncc_kernel(result[0] + offset_base, temp[thread_idx][0], mu_base[0] + offset_base, mu_match[0] + offset_match, sigma_base_inv[0] + offset_base, sigma_match_inv[0] + offset_match, rows, row_length, base_float.cols - 2);
	sncc_kernel(result[0], temp[thread_idx][0], mu_base[0] + offset_base, mu_match[0] + offset_match, sigma_base_inv[0] + offset_base, sigma_match_inv[0] + offset_match, rows, row_length, base_float.cols - 2);

	return result;
}

