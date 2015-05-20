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

#include "window_size.h"

#include "disparity_toolkit/genericfunctions.h"
#include "debugmatstore.h"
#include "disparity_region.h"

void showWindowSizes(cv::Mat& sizes)
{
	int conf[] = {0,0, 1,1};

	cv::Mat y_sizes(sizes.size[0], sizes.size[1], CV_8UC1);
	cv::Mat x_sizes(sizes.size[0], sizes.size[1], CV_8UC1);
	cv::Mat out[] = {y_sizes, x_sizes};

	cv::mixChannels(&sizes, 1, out, 2, conf, 2);

	matstore.add_mat(value_scaled_image<unsigned char, unsigned char>(y_sizes), "sizesY");
	matstore.add_mat(value_scaled_image<unsigned char, unsigned char>(x_sizes), "sizesX");
}

inline int countLabels(cv::Mat window, int clabel)
{
	assert(window.isContinuous());
	int* csubwindow_ptr = window.ptr<int>(0);
	int labelcount = 0;
	for(int i = 0; i < window.cols*window.rows; ++i)
	{
		if(*csubwindow_ptr++ == clabel)
			++labelcount;
	}
	return labelcount;
}

inline float countLabelsDisp(cv::Mat window, int clabel, const std::vector<disparity_region>& regions)
{
	assert(window.isContinuous());
	int cdisparity = regions[clabel].disparity;
	int* csubwindow_ptr = window.ptr<int>(0);

	int disp_error = 0;
	int trunc = 3;
	for(int i = 0; i < window.cols*window.rows; ++i)
	{
		int current_label = *csubwindow_ptr++;
		disp_error += std::min(std::abs(regions[current_label].disparity - cdisparity), trunc);
	}
	int total = window.cols*window.rows*trunc;
	return std::max(1.0f-disp_error/(float)total, 0.0f);
}

inline float compare_disparity(const cv::Mat& plabel_window, int clabel, const std::vector<short>& disparities)
{
	cv::Mat label_window = plabel_window.clone();
	assert(label_window.isContinuous());
	short cdisparity = disparities[clabel];
	const int* csubwindow_ptr = label_window.ptr<int>(0);
	const unsigned short totalpixel = label_window.cols*label_window.rows;
	const int* end_ptr = csubwindow_ptr + totalpixel;

	const int trunc = 3;
	std::array<int, trunc+1> label_counter;
	std::fill(label_counter.begin(), label_counter.end(), 0);

	//for(unsigned short i = 0; i < totalpixel; ++i)
	while(csubwindow_ptr != end_ptr)
	{
		int current_label = *csubwindow_ptr++;
		//unsigned char disp_dev = std::min(std::abs(disparities[current_label] - cdisparity), trunc);
		auto disp_dev = std::min(std::abs(disparities[current_label] - cdisparity), trunc);

		assert(disp_dev <= trunc); //disp_dev >= 0
		++(label_counter[disp_dev]);
	}
	int pos_labels = 3*label_counter[0]+2*label_counter[1]+label_counter[2];
	int total = pos_labels+3*label_counter[3];
	return pos_labels/(float)total;
}


template<typename lambda_type>
cv::Mat_<cv::Vec2b> adaptive_window_size(const cv::Mat& image, const cv::Mat_<int>& labels, float threshold, int minsize, int maxsize, const std::vector<disparity_region>& regions, lambda_type func)
{
	assert(minsize > 0);
	assert(maxsize > 0);

	std::vector<short> disparities(regions.size());
	for(std::size_t i = 0; i < regions.size(); ++i)
		disparities[i] = regions[i].disparity;

	cv::Mat_<cv::Vec2b> result = cv::Mat::zeros(labels.size(), CV_8UC2);
	const int onesidesizeMin = (minsize-1)/2;
	const int onesidesizeMax = (maxsize-1)/2;
	#pragma omp parallel for default(none) shared(labels, image, result, disparities, threshold, func, minsize, maxsize)
	for(int y = onesidesizeMin; y < labels.rows - onesidesizeMin; ++y)
	{
		int lastWindowSize = onesidesizeMin*2+1;

		for(int x = onesidesizeMin; x < labels.cols - onesidesizeMin; ++x)
		{
			int clabel = labels(y,x);
			int maxposs = std::min( std::min(labels.cols - x-1, labels.rows - y-1), onesidesizeMax );
			maxposs = std::min( maxposs, std::min( std::min(y-1, x-1), onesidesizeMax ) );

			maxposs = maxposs*2+1;

			int windowsizeX = std::min(lastWindowSize, maxposs);
			int windowsizeY = std::min(lastWindowSize, maxposs);

			bool grow = (lastWindowSize == minsize) ? true : false;

			while(true)
			{
				float measured = func(subwindow(labels, x,y, windowsizeX, windowsizeY), clabel, disparities);
				if(grow)
				{
					if(measured < threshold)
					{
						//shrink each direction seperatly
						float measured_altY = func(subwindow(labels, x,y, windowsizeX, windowsizeY-2), clabel, disparities);
						float measured_altX = func(subwindow(labels, x,y, windowsizeX-2, windowsizeY), clabel, disparities);

						if(measured_altY > threshold)
							windowsizeY -= 2;
						else if(measured_altX > threshold)
							windowsizeX -= 2;
						else {
							windowsizeX -= 2;
							windowsizeY -= 2;
							break;
						}
					}
					if(windowsizeX < maxposs && windowsizeY < maxposs)
					{
						windowsizeX += 2;
						windowsizeY += 2;
					} else
						break;
				} else {
					if(measured >= threshold)
					{
						grow = true;
						continue;
					}
					if( windowsizeX > minsize && windowsizeY > minsize) {
						windowsizeX -= 2;
						windowsizeY -= 2;
					} else
						break;
				}
			}
			lastWindowSize = std::min(windowsizeX, windowsizeY);
			result(y,x) = cv::Vec2b(std::max(windowsizeY, minsize), std::max(windowsizeX, minsize));
		}
	}

	return result;
}

cv::Mat_<cv::Vec2b> adaptive_window_size(const cv::Mat& image, const cv::Mat_<int>& labels, float threshold, int minsize, int maxsize, const std::vector<disparity_region>& regions)
{
	return adaptive_window_size(image, labels, threshold, minsize, maxsize, regions, compare_disparity);
}
