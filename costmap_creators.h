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

#ifndef COSTMAP_CREATORS_H
#define COSTMAP_CREATORS_H

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace costmap_creators
{
namespace sliding_window
{
template<typename cost_class>
cv::Mat joint_fixed_size(const cv::Mat& base, const cv::Mat& match, int dispMin, int dispMax, unsigned int windowsize)
{
	int disparityRange = dispMax - dispMin + 1;

	typedef float prob_table_type;
	int sz[] = {base.rows, base.cols, disparityRange};
	cv::Mat cost_map(3, sz, CV_32FC1, cv::Scalar(std::numeric_limits<float>::max()/3));

	const int y_min = windowsize/2;
	const int y_max = base.rows - windowsize/2;

	const int x_min = windowsize/2;
	const int x_max = base.cols - windowsize/2;

	cost_class cost_agg(match, windowsize);
	typename cost_class::thread_type thread_data;

	#pragma omp parallel for private(thread_data)
	for(int y = y_min; y < y_max; ++y)
	{
		cost_agg.prepare_row(thread_data, match, y);
		cv::Mat windowBase = subwindow(base, x_min, y, windowsize);
		for(int x = x_min; x < x_max; ++x)
		{
			cost_agg.prepare_window(thread_data, windowBase);
			int disp_start = std::min(std::max(x+dispMin, x_min), x_max-1) - x;
			int disp_end   = std::max(std::min(x+dispMax, x_max-1), x_min) - x;

			assert(disp_start-dispMin >= 0);
			assert(disp_start-dispMin < disparityRange);
			assert(disp_end-disp_start < disparityRange);
			assert(disp_end-disp_start >= 0);

			prob_table_type *result_ptr = cost_map.ptr<prob_table_type>(y,x, disp_start-dispMin);
			for(int d = disp_start; d <= disp_end; ++d)
			{
				*result_ptr++ = cost_agg.increm(thread_data, x+d);
			}
			windowBase.adjustROI(0,0,-1,1);
		}
	}

	return cost_map;
}

template<typename cost_class>
cv::Mat joint_flexible_size(const cv::Mat& base, const cv::Mat& match, int dispMin, int dispMax, const cv::Mat& windowsizes)
{
	int min_windowsize = 7;
	int disparityRange = dispMax - dispMin + 1;

	typedef float prob_table_type;
	int sz[] = {base.rows, base.cols, disparityRange};
	cv::Mat cost_map(3, sz, CV_32FC1, cv::Scalar(8));

	const int y_min = min_windowsize/2;
	const int y_max = base.rows - min_windowsize/2;

	const int x_min = min_windowsize/2;
	const int x_max = base.cols - min_windowsize/2;

	cost_class cost_agg(match);
	typename cost_class::thread_type thread_data;

	#pragma omp parallel for private(thread_data)
	for(int y = y_min; y < y_max; ++y)
	{
		cost_agg.prepareRow(thread_data, match, y);

		for(int x = x_min; x < x_max; ++x)
		{
			cv::Vec2b cwindowsize = windowsizes.at<cv::Vec2b>(y,x);
			if(cwindowsize[0] > 0 && cwindowsize[1] > 0)
			{
				cv::Mat windowBase = subwindow(base, x, y, cwindowsize[1], cwindowsize[0] );
				cost_agg.prepareWindow(thread_data, windowBase, cwindowsize[1], cwindowsize[0] );
				int cx_min = cwindowsize[1]/2;
				int cx_max = base.cols - cwindowsize[1]/2;
				int disp_start = std::min(std::max(x+dispMin, cx_min), cx_max-1) - x;
				int disp_end   = std::max(std::min(x+dispMax, cx_max-1), cx_min) - x;

				assert(disp_start-dispMin >= 0);
				assert(disp_start-dispMin < dispMax - dispMin + 1);
				assert(disp_end-disp_start < dispMax - dispMin + 1);
				assert(disp_end-disp_start >= 0);

				prob_table_type *result_ptr = cost_map.ptr<prob_table_type>(y,x, disp_start-dispMin);
				for(int d = disp_start; d <= disp_end; ++d)
				{
					*result_ptr++ = cost_agg.increm(thread_data, x+d);
				}
			}
		}
	}

	return cost_map;
}

//berechnet fuer subranges die disparitaet. disp*_comp gibt den gesamten Bereich an, rangeCenter-/+dispRange/2 den Teilbereich
template<typename cost_class>
cv::Mat flexible_size_flexible_disparityrange(const cv::Mat& base, const cv::Mat& match, const cv::Mat& windowsizes, const cv::Mat& rangeCenter, int disparityRange, int dispMin_comp, int dispMax_comp, unsigned int min_windowsize, unsigned int max_windowsize)
{
	typedef float prob_table_type;
	int sz[] = {base.rows, base.cols, disparityRange};
	cv::Mat cost_map(3, sz, CV_32FC1, cv::Scalar(8));

	const int y_min = min_windowsize/2;
	const int y_max = base.rows - min_windowsize/2;

	const int x_min = min_windowsize/2;
	const int x_max = base.cols - min_windowsize/2;

	cost_class cost_agg(match, max_windowsize);
	typename cost_class::thread_type thread_data;

	#pragma omp parallel for private(thread_data)
	for(int y = y_min; y < y_max; ++y)
	{
		cost_agg.prepare_row(thread_data, match, y);

		for(int x = x_min; x < x_max; ++x)
		{
			cv::Vec2b cwindowsize = windowsizes.at<cv::Vec2b>(y,x);

			int dispMin_pre = rangeCenter.at<short>(y,x) - disparityRange/2+1;
			int dispMax_pre = rangeCenter.at<short>(y,x) + disparityRange/2;

			int dispMin = std::max(dispMin_pre, dispMin_comp);
			int dispMax = std::min(dispMax_pre, dispMax_comp);

			int cx_min = cwindowsize[1]/2;
			int cx_max = base.cols - cwindowsize[1]/2;
			int disp_start = std::min(std::max(x+dispMin, cx_min), cx_max-1) - x;
			int disp_end   = std::max(std::min(x+dispMax, cx_max-1), cx_min) - x;

			if(cwindowsize[0] > 0 && cwindowsize[1] > 0 && disp_end > disp_start)
			{
				cv::Mat windowBase = subwindow(base, x, y, cwindowsize[1], cwindowsize[0] );
				cost_agg.prepare_window(thread_data, windowBase, cwindowsize[1], cwindowsize[0] );

				assert(disp_start-dispMin >= 0);
				assert(disp_start-dispMin < dispMax - dispMin + 1);
				assert(disp_end-disp_start < dispMax - dispMin + 1);
				assert(disp_end-disp_start >= 0);

				assert(disp_start >= dispMin_comp);
				assert(disp_end <= dispMax_comp);
				assert(disp_start-dispMin_pre >= 0);
				assert(disp_end-dispMin_pre < disparityRange);

				prob_table_type *result_ptr = cost_map.ptr<prob_table_type>(y,x, disp_start-dispMin_pre);
				for(int d = disp_start; d <= disp_end; ++d)
				{
					*result_ptr++ = cost_agg.increm(thread_data, x+d);
				}
			}
		}
	}

	return cost_map;
}
}
//! Calculates pointwise (no window involved) the disparity. cost aggregator must be passed as a function object, returns a cost map
template<typename cost_class, typename data_type>
cv::Mat calculate_pixelwise(cv::Mat base, data_type data, int dispMin, int dispMax)
{
	int disparityRange = dispMax - dispMin + 1;
	int sz[]  = {base.size[0], base.size[1], disparityRange};
	cv::Mat result = cv::Mat(3, sz, CV_32FC1, cv::Scalar(std::numeric_limits<float>::max()));

	cost_class cost_agg(base, data, dispMin);

	#pragma omp parallel for
	for(int y = 0; y< base.size[0]; ++y)
	{
		for(int x = 0; x < base.size[1]; ++x)
		{
			int disp_start = std::min(std::max(x+dispMin, 0), base.size[1]-1) - x;
			int disp_end   = std::max(std::min(x+dispMax, base.size[1]-1), 0) - x;

			assert(disp_start-dispMin >= 0);
			assert(disp_start-dispMin < dispMax - dispMin + 1);
			assert(disp_end-disp_start < dispMax - dispMin + 1);
			assert(disp_end-disp_start >= 0);

			float *result_ptr = result.ptr<float>(y,x, disp_start-dispMin);

			for(int d = disp_start; d <= disp_end; ++d)
			{
				*result_ptr++ = cost_agg(y,x,d);
			}
		}
	}

	return result;
}

/*template<typename data_type>
cv::Mat flexBoxFilter(cv::Mat src, cv::Mat windowsizes)
{
	cv::Mat cost_map(src.size(), CV_32FC1, cv::Scalar(8));

	#pragma omp parallel for default(none) shared(src, cost_map, windowsizes)
	for(int y = 0; y < src.rows; ++y)
	{
		for(int x = 0; x < src.cols; ++x)
		{
			cv::Vec2b cwindowsize = windowsizes.at<cv::Vec2b>(y,x);

			if(cwindowsize[0] > 0 && cwindowsize[1] > 0)
			{
				cv::Mat windowBase = subwindow(src, x, y, cwindowsize[1], cwindowsize[0] );
				cost_map.at<float>(y,x) = cv::sum(windowBase)/windowBase.total();
			}
		}
	}

	return cost_map;
}*/

}


template<typename cost_type, typename window_type>
cv::Mat disparitywise_calculator(cost_type cost_func, window_type window_sum, cv::Mat base, int dispMin, int dispMax)
{
	int sz[] = {base.rows, base.cols, dispMax - dispMin + 1};
	cv::Mat_<float> result = cv::Mat(3, sz, CV_32FC1, cv::Scalar(-std::numeric_limits<float>::max()));

	#pragma omp parallel for
	for(int d = dispMin; d <= dispMax; ++d)
	{
		cv::Mat temp_result = window_sum(cost_func(d), d);

		for(int y = 0; y < base.rows; ++y)
		{
			if(d < 0)
			{
				for(int x = -d; x < base.cols; ++x)
					result(y,x,d-dispMin) = temp_result.at<float>(y, x+d);
			}
			else
			{
				int max_x = base.cols - d;
				for(int x = 0; x < max_x; ++x)
					result(y,x,d-dispMin) = temp_result.at<float>(y, x);
			}
		}
	}
	return result;
}

template<typename cost_type>
cv::Mat simple_window_disparitywise_calculator(cost_type cost_func, cv::Size size, cv::Mat base, int dispMin, int dispMax)
{
	auto window_sum = [=](const cv::Mat& pre_result, int){
		cv::Mat temp_result;
		cv::boxFilter(pre_result, temp_result, -1, size);
		return temp_result;
	};

	return disparitywise_calculator(cost_func, window_sum, base, dispMin, dispMax);
}

#endif // COSTMAP_CREATORS_H
