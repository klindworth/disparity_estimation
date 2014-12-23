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

#ifndef WINDOW_SIZE_H
#define WINDOW_SIZE_H

#include <opencv2/core/core.hpp>
#include "disparity_region.h"
#include "slidingEntropy.h"

template<int quantizer>
inline float compareEntropies(const cv::Mat_<float>& log_table, const cv::Mat& plabel_window, int clabel, const std::vector<disparity_region>& regions, const cv::Mat& pimg_window)
{
	cv::Mat label_window = plabel_window.clone();
	cv::Mat img_window = pimg_window.clone();
	const int bins = 256/quantizer;
	assert(label_window.total() == img_window.total());
	fast_array<unsigned short, bins+2> counter_total, counter_own;

	int cdisparity = regions[clabel].disparity;
	const int* csubwindow_ptr = label_window.ptr<int>(0);
	const unsigned char* cimg_window_ptr = img_window.data;

	int label_total = 0, label_own = 0;

	for(int i = 0; i < label_window.cols*label_window.rows; ++i)
	{
		int current_label = *csubwindow_ptr++;
		unsigned char current_pixel = 1 + *cimg_window_ptr++;

		counter_total(current_pixel)++;
		++label_total;
		if(std::abs(regions[current_label].disparity - cdisparity) <= 1)
		{
			counter_own(current_pixel)++;
			++label_own ;
		}
	}

	if(label_own == label_total)
		return 1.0f;


	float entropy_total = std::max(costmap_creators::entropy::calculate_entropy_unnormalized<float>(counter_total, log_table, bins), std::numeric_limits<float>::min());
	float entropy_own   = std::max(costmap_creators::entropy::calculate_entropy_unnormalized<float>(counter_own, log_table, bins), std::numeric_limits<float>::min());

	if(label_own/(float)label_total > 0.3f)
		return entropy_own/entropy_total;
	else
		return 0.0f;
}

inline float compareLabels(const cv::Mat_<float>&, const cv::Mat& plabel_window, int clabel, const std::vector<disparity_region>& regions, const cv::Mat&)
{
	cv::Mat label_window = plabel_window.clone();
	assert(label_window.isContinuous());
	int cdisparity = regions[clabel].disparity;
	const int* csubwindow_ptr = label_window.ptr<int>(0);
	const unsigned short totalpixel = label_window.cols*label_window.rows;

	const int trunc = 3;
	std::array<int, trunc+1> label_counter;
	std::fill(label_counter.begin(), label_counter.end(), 0);

	for(unsigned short i = 0; i < totalpixel; ++i)
	{
		int current_label = *csubwindow_ptr++;
		unsigned char disp_dev = std::min(std::abs(regions[current_label].disparity - cdisparity), trunc);

		assert(disp_dev >= 0 && disp_dev <= trunc);
		++(label_counter[disp_dev]);
	}
	int pos_labels = 3*label_counter[0]+2*label_counter[1]+label_counter[2];
	int total = pos_labels+3*label_counter[3];
	return pos_labels/(float)total;
}

cv::Mat findWindowSizeEntropy(const cv::Mat& image, const cv::Mat& labels, const float& threshold, const int& minsize, const int& maxsize, const std::vector<disparity_region>& regions, std::function<float(cv::Mat_<float>&, cv::Mat, int, const std::vector<disparity_region>&, cv::Mat)> func);
void showWindowSizes(cv::Mat& sizes);

#endif // WINDOW_SIZE_H
