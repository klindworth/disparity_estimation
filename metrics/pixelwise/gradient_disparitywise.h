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

#ifndef SLIDINGGRADIENT_H
#define SLIDINGGRADIENT_H

#include <opencv2/core/core.hpp>

class single_stereo_task;

cv::Mat norm2(const cv::Mat& mat1, const cv::Mat& mat2);
cv::Mat gradient_image(const cv::Mat &base, const cv::Mat &base2, const cv::Mat &match, const cv::Mat &match2, int d);
void derived_mat(const cv::Mat& input, cv::Mat& grad_x, cv::Mat& grad_y, bool blur);

class gradient_calculator
{
public:
	typedef float result_type;

	gradient_calculator(cv::Mat base, cv::Mat match);
	cv::Mat_<float> operator()(int d);

protected:
	cv::Mat gradLeftX, gradLeftY, gradRightX, gradRightY;
};

//! Calulates a dense costmap for the disparity with gradients as metric
cv::Mat sliding_gradient(const single_stereo_task &task, int windowsize);

#endif // SLIDINGGRADIENT_H
