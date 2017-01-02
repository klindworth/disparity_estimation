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

#ifndef DISPARITYWISE_CALCULATOR_H
#define DISPARITYWISE_CALCULATOR_H

#include <opencv2/core/core.hpp>

inline cv::Mat prepare_base(const cv::Mat& src, int d)
{
	if(d > 0)
		return src(cv::Range(0, src.rows), cv::Range(0, src.cols-d));
	else
		return src(cv::Range(0, src.rows), cv::Range(-d, src.cols));
}

inline cv::Mat prepare_match(const cv::Mat& src, int d)
{
	if(d > 0)
		return src(cv::Range(0, src.rows), cv::Range(d, src.cols));
	else
		return src(cv::Range(0, src.rows), cv::Range(0, src.cols+d));
}

inline cv::Rect prepare_result_rect(cv::Size input, int d)
{
	return cv::Rect(std::max(-d, 0), 0, input.width, input.height);
}

inline cv::Rect prepare_border_rect(cv::Size input, int d)
{
	if(d > 0)
		return cv::Rect(input.width, 0, d, input.height);
	else
		return cv::Rect(0,0,-d,input.height);
}

inline cv::Mat prepare_result(cv::Size src, int d, float val)
{
	cv::Rect rect_result = prepare_result_rect(src, d);
	cv::Rect rect_border = prepare_border_rect(src, d);
	cv::Size nsize(rect_result.width + rect_border.width, src.height);

	cv::Mat_<float> result(nsize);
	result(rect_border) = val;

	return result;
}

inline int calculate_row_length(const cv::Mat& src, int d)
{
	return src.cols - std::abs(d);
}

inline constexpr int calculate_base_offset(int d)
{
	return std::max(-d, 0);
}

inline constexpr int calculate_match_offset(int d)
{
	return std::max(d, 0);
}

#endif
