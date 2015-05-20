/*
Copyright (c) 2013, Kai Klindworth
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

#include "gradient_disparitywise.h"

#include "genericfunctions.h"
#include "stereotask.h"
#include "costmap_creators.h"
#include "disparitywise_calculator.h"
#include <cmath>
#include <opencv2/imgproc/imgproc.hpp>
#ifdef __SSE__
#include <xmmintrin.h>
#endif

cv::Mat norm2(const cv::Mat& mat1, const cv::Mat& mat2)
{
	cv::Mat norm = cv::Mat(mat1.size(), CV_32FC1, cv::Scalar(0));
	float *norm_ptr = norm.ptr<float>();
	const float *mat1_ptr = mat1.ptr<float>();
	const float *mat2_ptr = mat2.ptr<float>();

	const int total = mat1.total();

	for(int i = 0; i < total; ++i)
	{
		/**norm_ptr++ = std::max(1.0f, std::sqrt(*mat1_ptr * *mat1_ptr + *mat2_ptr * *mat2_ptr));
		++mat1_ptr;
		++mat2_ptr;*/
		*norm_ptr++ = std::max(1.0f, std::hypot(*mat1_ptr++, *mat2_ptr++));
	}

	return norm;
}

cv::Mat gradient_image(const cv::Mat& base, const cv::Mat& base2, const cv::Mat& match, const cv::Mat& match2, int d)
{
	const int simd_size = 4;

	cv::Mat match_shifted  = prepare_match(match, d).clone();
	cv::Mat match2_shifted = prepare_match(match2, d).clone();

	cv::Mat base_cutted  = prepare_base(base, d).clone();
	cv::Mat base2_cutted = prepare_base(base2, d).clone();

	assert(match2_shifted.rows == base2_cutted.rows && match2_shifted.cols == base2_cutted.cols);

	cv::Mat base_norm  = norm2(base_cutted, base2_cutted);
	cv::Mat match_norm = norm2(match_shifted, match2_shifted);

	cv::Mat result(base_cutted.size(), CV_32FC1, cv::Scalar(-std::numeric_limits<float>::max()));

	const int counter_max = base_cutted.total();
	int i = 0;

	#if __SSE__
	const __m128* simd_base1  = (const __m128*) base_cutted.data;
	const __m128* simd_base2  = (const __m128*) base2_cutted.data;
	const __m128* simd_match1 = (const __m128*) match_shifted.data;
	const __m128* simd_match2 = (const __m128*) match2_shifted.data;
	const __m128* simd_norm_match = (const __m128*) match_norm.data;
	const __m128* simd_norm_base  = (const __m128*) base_norm.data;

	__m128 simd_anglearg;
	__m128* simd_result = (__m128*) result.data;

	const int bound = counter_max-simd_size+1;
	for(; i < bound; i += simd_size)
	{
		__m128 sse_dotprod = _mm_add_ps( _mm_mul_ps(*simd_base1++, *simd_match1++), _mm_mul_ps(*simd_base2++, *simd_match2++));
		simd_anglearg = _mm_div_ps(sse_dotprod, _mm_mul_ps(*simd_norm_base, *simd_norm_match));
		simd_anglearg = _mm_mul_ps(simd_anglearg, simd_anglearg);
		__m128 sse_min_norm = _mm_min_ps(*simd_norm_base++, *simd_norm_match++);

		*simd_result = _mm_mul_ps(simd_anglearg, sse_min_norm);
		++simd_result;
	}
	#endif

	//process remaining pixels
	for(;i < counter_max; ++i)
	{
		const float cbase  = base_cutted.at<float>(i);
		const float cmatch = match_shifted.at<float>(i);
		const float cbase2  = base2_cutted.at<float>(i);
		const float cmatch2 = match2_shifted.at<float>(i);
		const float cnormMatch = match_norm.at<float>(i);
		const float cnormBase  = base_norm.at<float>(i);
		float anglearg = (cbase*cmatch+cbase2*cmatch2)/(cnormBase*cnormMatch);
		float weight = anglearg*anglearg;
		result.at<float>(i) = weight*std::min(cnormBase, cnormMatch);
	}

	return result;
}

void derived_mat(const cv::Mat& input, cv::Mat& grad_x, cv::Mat& grad_y, bool blur)
{
	cv::Mat temp;

	if(blur)
		cv::GaussianBlur(input, temp, cv::Size(3,3), 0, 0, cv::BORDER_DEFAULT );
	else
		temp = input;

	//Gradient X
	cv::Scharr( temp, grad_x, CV_32FC1, 1, 0);

	//Gradient Y
	cv::Scharr( temp, grad_y, CV_32FC1, 0, 1);
}

gradient_calculator::gradient_calculator(cv::Mat base, cv::Mat match)
{
	bool blur = false;
	derived_mat(base,  gradLeftX,  gradLeftY,  blur);
	derived_mat(match, gradRightX, gradRightY, blur);
}

cv::Mat_<float> gradient_calculator::operator()(int d)
{
	return gradient_image(gradLeftX, gradLeftY, gradRightX, gradRightY, d);
}

cv::Mat sliding_gradient(const single_stereo_task& task, int windowsize)
{
	gradient_calculator calc(task.baseGray, task.matchGray);

	return simple_window_disparitywise_calculator(calc, cv::Size(windowsize, windowsize), task.base.size(), task.range);
}
