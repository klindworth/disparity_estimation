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

#include "slidingEntropy.h"
#include "genericfunctions.h"
#include "debugmatstore.h"

void showWindowSizes(cv::Mat& sizes)
{
	int conf[] = {0,0, 1,1};

	cv::Mat y_sizes(sizes.size[0], sizes.size[1], CV_8UC1);
	cv::Mat x_sizes(sizes.size[0], sizes.size[1], CV_8UC1);
	cv::Mat out[] = {y_sizes, x_sizes};

	cv::mixChannels(&sizes, 1, out, 2, conf, 2);

	matstore.addMat(value_scaled_image<unsigned char, unsigned char>(y_sizes), "sizesY");
	matstore.addMat(value_scaled_image<unsigned char, unsigned char>(x_sizes), "sizesX");
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

inline float compareEntropies_old(cv::Mat_<float>& log_table, cv::Mat label_window, int clabel, const std::vector<disparity_region>& regions, cv::Mat img_window)
{
	assert(label_window.total() == img_window.total());
	fast_array<unsigned short, 34> counter_own, counter_foreign;

	int cdisparity = regions[clabel].disparity;
	int* csubwindow_ptr = label_window.ptr<int>(0);
	unsigned char* cimg_window_ptr = img_window.data;

	for(int i = 0; i < label_window.cols*label_window.rows; ++i)
	{
		int current_label = *csubwindow_ptr++;
		unsigned char current_pixel = 1 + *cimg_window_ptr++;

		if(std::abs(regions[current_label].disparity - cdisparity) > 2)
			counter_foreign(current_pixel)++;
		else
			counter_own(current_pixel)++;
	}

	float entropy_own = std::max(costmap_creators::entropy::calculate_entropy_unnormalized<float>(counter_own, log_table, 32), std::numeric_limits<float>::min());
	float entropy_foreign = std::max(costmap_creators::entropy::calculate_entropy_unnormalized<float>(counter_foreign, log_table, 32), std::numeric_limits<float>::min());

	return entropy_own/entropy_foreign;
}

cv::Mat findWindowSizeEntropy(const cv::Mat& image, const cv::Mat& labels, const float& threshold, const int& minsize, const int& maxsize, const std::vector<disparity_region>& regions, std::function<float(cv::Mat_<float>&, cv::Mat, int, const std::vector<disparity_region>&, cv::Mat)> func)
{
	cv::Mat_<float> log_table;
	costmap_creators::entropy::fill_entropytable_unnormalized(log_table, maxsize*maxsize);

	cv::Mat result = cv::Mat::zeros(labels.size(), CV_8UC2);
	const int onesidesizeMin = (minsize-1)/2;
	const int onesidesizeMax = (maxsize-1)/2;
    #pragma omp parallel for default(none) shared(labels, image, result, regions, threshold, log_table, func, minsize, maxsize)
	for(int y = onesidesizeMin; y < labels.rows - onesidesizeMin; ++y)
	{
		int lastWindowSize = onesidesizeMin*2+1;

		for(int x = onesidesizeMin; x < labels.cols - onesidesizeMin; ++x)
		{
			int clabel = labels.at<int>(y,x);
			int maxposs = std::min( std::min(labels.cols - x-1, labels.rows - y-1), onesidesizeMax );
			maxposs = std::min( maxposs, std::min( std::min(y-1, x-1), onesidesizeMax ) );

			maxposs = maxposs*2+1;

			int windowsizeX = std::min(lastWindowSize, maxposs);
			int windowsizeY = std::min(lastWindowSize, maxposs);

			bool grow = (lastWindowSize == minsize) ? true : false;

			while(true)
			{
				float measured = func(log_table, subwindow(labels, x,y, windowsizeX, windowsizeY), clabel, regions, subwindow(image, x,y, windowsizeX, windowsizeY));
				if(grow)
				{
					if(measured < threshold)
					{
						//shrink each direction seperatly
						float measured_altY = func(log_table, subwindow(labels, x,y, windowsizeX, windowsizeY-2), clabel, regions, subwindow(image, x,y, windowsizeX, windowsizeY-2));
						float measured_altX = func(log_table, subwindow(labels, x,y, windowsizeX-2, windowsizeY), clabel, regions, subwindow(image, x,y, windowsizeX-2, windowsizeY));

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
			result.at<cv::Vec2b>(y,x) = cv::Vec2b(std::max(windowsizeY, minsize), std::max(windowsizeX, minsize));
		}
	}

	return result;
}

cv::Mat findWindowSize(cv::Mat& labels, const float& threshold, const int& minsize, const int& maxsize, const std::vector<disparity_region>& regions, std::function<float(cv::Mat, int, const std::vector<disparity_region>& regions)> func)
{
	cv::Mat result = cv::Mat::zeros(labels.size(), CV_8UC2);
	const int onesidesizeMin = (minsize-1)/2;
	const int onesidesizeMax = (maxsize-1)/2;
	for(int y = onesidesizeMin; y < labels.rows - onesidesizeMin; ++y)
	{
		for(int x = onesidesizeMin; x < labels.cols - onesidesizeMin; ++x)
		{
			int windowsizeX = onesidesizeMin*2+1;
			int windowsizeY = onesidesizeMin*2+1;
			int clabel = labels.at<int>(y,x);
			int maxposs = std::min( std::min(labels.cols - x-1, labels.rows - y-1), onesidesizeMax );
			maxposs = std::min( maxposs, std::min( std::min(y-1, x-1), onesidesizeMax ) );

			for(int i = onesidesizeMin; i <= maxposs; ++i)
			{
				float labelcount = func(subwindow(labels, x,y, windowsizeX, windowsizeY).clone(), clabel, regions);
				if(labelcount < threshold)
				{
					//shrink each direction seperatly
					labelcount = func(subwindow(labels, x,y, windowsizeX, windowsizeY-2).clone(), clabel, regions);
					float labelcount2 = func(subwindow(labels, x,y, windowsizeX-2, windowsizeY).clone(), clabel, regions);

					if(labelcount > threshold)
						windowsizeY -= 2;
					else if(labelcount2 > threshold)
						windowsizeX -= 2;
					else {
						windowsizeX -= 2;
						windowsizeY -= 2;
						break;
					}
				}
				if(i != maxposs)
				{
					windowsizeX += 2;
					windowsizeY += 2;
				}
			}
			result.at<cv::Vec2b>(y,x) = cv::Vec2b(std::max(windowsizeY, minsize), std::max(windowsizeX, minsize));
		}
	}
	return result;
}

