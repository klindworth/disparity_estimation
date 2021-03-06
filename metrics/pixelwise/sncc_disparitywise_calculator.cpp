/*
Copyright (c) 2014, Kai Klindworth
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "sncc_disparitywise_calculator.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <omp.h>

#include "disparitywise_calculator.h"

#include <numeric>

bool any_nan(cv::Mat_<float> src)
{
	for(auto it = src.begin(); it != src.end(); ++it)
	{

		if(std::isnan(*it))
			return true;
	}
	return false;
}

void mat_clamp(cv::Mat_<float>& src, float min, float max)
{
	for(auto it = src.begin(); it != src.end(); ++it)
	{
		*it = std::max(std::min(*it, max), min);
	}
}

sncc_disparitywise_calculator::sncc_disparitywise_calculator(const cv::Mat& pbase, const cv::Mat& pmatch)
{
	//long long start = cv::getCPUTickCount();
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

	mat_clamp(sigma_base_inv, -500, 500);

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

	//sigma_match_inv = cv::max(sigma_match_inv, 500);
	mat_clamp(sigma_match_inv, -500, 500);

	cache = std::vector<sncc_task_cache>(omp_get_max_threads(), sncc_task_cache(pbase.cols));

	//std::cout << "init: " << cv::getCPUTickCount() - start << std::endl;

	assert(!any_nan(match_float));
	assert(!any_nan(base_float));
	assert(!any_nan(mu_base));
	assert(!any_nan(mu_match));
	assert(!any_nan(sigma_base_inv));
	assert(!any_nan(sigma_match_inv));
}


typedef float* __restrict__ data_ptr;
typedef const float* __restrict__ data_cptr;

inline void prepare_line(data_ptr temp, data_cptr base, data_cptr match, int cols)
{
	for(int x = 0; x < cols; ++x)
		*temp++ = *base++ * *match++;
}

void sncc_kernel_row(data_ptr result, data_cptr mu_base, data_cptr mu_match, data_cptr sigma_base_inv, data_cptr sigma_match_inv, int cols, int row_stride, data_cptr base, data_cptr match, sncc_task_cache& cache, int y)
{
	const float norm_factor = 1.0f/9.0f;

	prepare_line(cache.coltemp[cache.replace_idx].data(), base+(y+2)*(row_stride+2), match+(y+2)*(row_stride+2), cols+2);
	cache.replace_idx = (cache.replace_idx+1) %3;

	data_cptr temp_1 = cache.coltemp[0].data();
	data_cptr temp_2 = cache.coltemp[1].data();
	data_cptr temp_3 = cache.coltemp[2].data();

	for(int x = 0; x < cols+2; ++x)
	{
		cache.boxcol_temp[x] = *temp_1++ + *temp_2++ + *temp_3++;
	}

	data_cptr boxcol_temp_ptr = cache.boxcol_temp.data();
	for(int x = 0; x < cols; ++x)
	{
		float sum = 0.0f;
		for(int dx = 0; dx < 3; ++dx)
			sum += *(boxcol_temp_ptr + dx);
		boxcol_temp_ptr++;

		sum *= norm_factor;
		cache.box_temp[x] = sum;

		assert(!std::isnan(sum));
	}

	data_cptr box_ptr = cache.box_temp.data();

	for(int x = 0; x < cols; ++x)
	{
		/*assert(!std::isnan(*box_ptr));
		assert(!std::isnan(*mu_base));
		assert(!std::isnan(sum));
		assert(!std::isnan(sum));
		assert(!std::isnan(sum));*/

		assert(!std::isnan(1.0f - (*box_ptr - *mu_base * *mu_match) * *sigma_base_inv * *sigma_match_inv));
		*result++ = 1.0f - (*box_ptr++ - *mu_base++ * *mu_match++) * *sigma_base_inv++ * *sigma_match_inv++;
	}
}

void sncc_kernel(data_ptr result, const data_cptr mu_base, const data_cptr mu_match, const data_cptr sigma_base_inv, const data_cptr sigma_match_inv, int rows, int cols, int row_stride, const data_cptr base, const data_cptr match, sncc_task_cache& cache)
{
	for(int i = 0; i < 2; ++i)
		prepare_line(cache.coltemp[i].data(), base + i*(row_stride+2), match + i*(row_stride+2), cols+2);
	cache.replace_idx = 2;

	for(int y = 0; y < rows; ++y)
	{
		const int y_offset = y*row_stride;
		const data_cptr mu_base_ptr = mu_base + y_offset;
		const data_cptr mu_match_ptr = mu_match + y_offset;
		const data_cptr sigma_base_inv_ptr = sigma_base_inv + y_offset;
		const data_cptr sigma_match_inv_ptr = sigma_match_inv + y_offset;
		data_ptr result_ptr = result+y*cols;

		sncc_kernel_row(result_ptr, mu_base_ptr, mu_match_ptr, sigma_base_inv_ptr, sigma_match_inv_ptr, cols, row_stride, base, match, cache, y);
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

	assert(!any_nan(result));

	unsigned int thread_idx = static_cast<unsigned int>(omp_get_thread_num());
	assert(cache.size() > thread_idx);
	sncc_kernel(result[0], mu_base[0] + offset_base, mu_match[0] + offset_match, sigma_base_inv[0] + offset_base, sigma_match_inv[0] + offset_match, rows, row_length, base_float.cols - 2, base_float.ptr<float>() + offset_base, match_float.ptr<float>() + offset_match, cache[thread_idx]);

	assert(!any_nan(result));

	return result;
}

