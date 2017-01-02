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

#ifndef SNCC_CALCULATOR_H
#define SNCC_CALCULATOR_H

#include <opencv2/core/core.hpp>
#include <vector>
#include <array>

struct sncc_task_cache
{
	sncc_task_cache(std::size_t cols) : box_temp(cols), boxcol_temp(cols+2), replace_idx(2)
	{
		coltemp.fill(std::vector<float>(cols+2));
	}
	std::array<std::vector<float>, 3> coltemp;
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
