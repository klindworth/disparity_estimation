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

#include "disparity_toolkit/genericfunctions.h"
#include "disparity_toolkit/disparity_range.h"

namespace costmap_creators
{

template<typename T>
cv::Mat_<T> create_cost_map(const cv::Mat& image, int range, T default_value)
{
	int sz[] = {image.size[0], image.size[1], range};
	return cv::Mat_<T>(3, sz, default_value);
}

template<typename T>
cv::Mat_<T> create_cost_map(const cv::Mat& image, const disparity_range& range, T default_value)
{
	return create_cost_map<T>(image, range.size(), default_value);
}

template<typename T, typename lambda_func>
inline void transform_range(cv::Mat& cost_map, int y, int x, const disparity_range& crange, lambda_func func)
{
	T *result_ptr = cost_map.ptr<T>(y,x, crange.index(crange.start()));
	for(int d = crange.start(); d <= crange.end(); ++d)
	{
		*result_ptr++ = func(y, x, d);
	}
}

namespace sliding_window
{

template<typename cost_class>
cv::Mat joint_fixed_size(const cv::Mat& base, const cv::Mat& match, const disparity_range range, const int windowsize)
{
	assert(windowsize > 0);

	using prob_table_type = typename cost_class::result_type;
	cv::Mat cost_map = create_cost_map<prob_table_type>(base, range, std::numeric_limits<prob_table_type>::max()/3);

	const int border = windowsize/2;

	const int y_min = border;
	const int y_max = base.rows - border;

	const int x_min = border;
	const int x_max = base.cols - border;

	cost_class cost_agg(base, match, windowsize);
	typename cost_class::thread_type thread_data;

	#pragma omp parallel for private(thread_data)
	for(int y = y_min; y < y_max; ++y)
	{
		cost_agg.prepare_row(thread_data, y);
		cv::Mat windowBase = subwindow(base, x_min, y, windowsize);
		for(int x = x_min; x < x_max; ++x)
		{
			cost_agg.prepare_window(thread_data, windowBase);
			const disparity_range crange = range.restrict_to_image(x, base.cols, border);

			transform_range<prob_table_type>(cost_map, y, x, crange, [&](int, int x, int d){
				return cost_agg.increm(thread_data, x+d);
			});

			windowBase.adjustROI(0,0,-1,1);
		}
	}

	return cost_map;
}

template<typename cost_class>
cv::Mat joint_flexible_size(const cv::Mat& base, const cv::Mat& match, const disparity_range range, const cv::Mat_<cv::Vec2b>& windowsizes)
{
	int min_windowsize = 7;

	typedef float prob_table_type;
	cv::Mat cost_map = create_cost_map<prob_table_type>(base, range, 8);

	const int y_min = min_windowsize/2;
	const int y_max = base.rows - min_windowsize/2;

	const int x_min = min_windowsize/2;
	const int x_max = base.cols - min_windowsize/2;

	cost_class cost_agg(base, match, range);
	typename cost_class::thread_type thread_data;

	#pragma omp parallel for private(thread_data)
	for(int y = y_min; y < y_max; ++y)
	{
		cost_agg.prepare_row(thread_data, y);

		for(int x = x_min; x < x_max; ++x)
		{
			const cv::Vec2b cwindowsize = windowsizes(y,x);
			if(cwindowsize[0] > 0 && cwindowsize[1] > 0)
			{
				cv::Mat windowBase = subwindow(base, x, y, cwindowsize[1], cwindowsize[0] );
				cost_agg.prepareWindow(thread_data, windowBase, cwindowsize[1], cwindowsize[0] );

				const disparity_range crange = range.restrict_to_image(x, base.cols, cwindowsize[1]/2);

				transform_range<prob_table_type>(cost_map, y, x, crange, [&](int, int x, int d) {
					return cost_agg.increm(thread_data, x, d);
				});
			}
		}
	}

	return cost_map;
}

//berechnet fuer subranges die disparitaet. disp*_comp gibt den gesamten Bereich an, rangeCenter-/+dispRange/2 den Teilbereich
template<typename cost_class>
cv::Mat flexible_size_flexible_disparityrange(const cv::Mat& base, const cv::Mat& match, const cv::Mat& windowsizes, const cv::Mat& rangeCenter, int disparity_delta, const disparity_range range_bound, unsigned int min_windowsize, unsigned int max_windowsize)
{
	typedef float prob_table_type;
	cv::Mat cost_map = create_cost_map<prob_table_type>(base, disparity_delta*2+1, 8);

	const int y_min = min_windowsize/2;
	const int y_max = base.rows - min_windowsize/2;

	const int x_min = min_windowsize/2;
	const int x_max = base.cols - min_windowsize/2;

	cost_class cost_agg(base, match, range_bound, max_windowsize);
	typename cost_class::thread_type thread_data;

	#pragma omp parallel for private(thread_data)
	for(int y = y_min; y < y_max; ++y)
	{
		cost_agg.prepare_row(thread_data, y);

		for(int x = x_min; x < x_max; ++x)
		{
			cv::Vec2b cwindowsize = windowsizes.at<cv::Vec2b>(y,x);

			const disparity_range crange  = range_bound.subrange_with_subspace(rangeCenter.at<short>(y,x), disparity_delta).restrict_to_image(x, base.cols, cwindowsize[1]/2);

			if(cwindowsize[0] > 0 && cwindowsize[1] > 0 && crange.end() > crange.start())
			{
				cost_agg.prepare_window(thread_data, x, cwindowsize[1], cwindowsize[0] );

				transform_range<prob_table_type>(cost_map, y, x, crange, [&](int, int x, int d) {
					return cost_agg.increm(thread_data, x, d);
				});
			}
		}
	}

	return cost_map;
}

}
//! Calculates pointwise (no window involved) the disparity. cost aggregator must be passed as a function object, returns a cost map
template<typename cost_class, typename data_type>
cv::Mat calculate_pixelwise(cv::Mat base, data_type data, const disparity_range range)
{
	cv::Mat result = create_cost_map<float>(base, range, std::numeric_limits<float>::max());

	cost_class cost_agg(base, data, range.start());

	#pragma omp parallel for
	for(int y = 0; y< base.size[0]; ++y)
	{
		for(int x = 0; x < base.size[1]; ++x)
		{
			const disparity_range crange = range.restrict_to_image(x, base.size[1]);

			transform_range<float>(result, y, x, crange, [&](int y, int x, int d) {
				return cost_agg(y, x, d);
			});
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
cv::Mat disparitywise_calculator(cost_type cost_func, window_type window_sum, cv::Size base_size, const disparity_range range)
{
	int sz[] = {base_size.height, base_size.width, range.size()};
	cv::Mat_<float> result = cv::Mat(3, sz, CV_32FC1, cv::Scalar(std::numeric_limits<float>::max()));
	//cv::Mat_<float> result = costmap_creators::sliding_window::create_cost_map(base)

	#pragma omp parallel for
	for(int d = range.start(); d <= range.end(); ++d)
	{
		cv::Mat temp_result = window_sum(cost_func(d), d);

		for(int y = 0; y < base_size.height; ++y)
		{
			if(d < 0)
			{
				for(int x = -d; x < base_size.width; ++x)
					result(y,x,range.index(d)) = temp_result.at<float>(y, x+d);
			}
			else
			{
				int max_x = base_size.width - d;
				for(int x = 0; x < max_x; ++x)
					result(y,x,range.index(d)) = temp_result.at<float>(y, x);
			}
		}
	}
	return result;
}

template<typename cost_type>
cv::Mat simple_window_disparitywise_calculator(cost_type cost_func, cv::Size window_size, cv::Size base_size, const disparity_range range)
{
	auto window_sum = [=](const cv::Mat& pre_result, int){
		cv::Mat temp_result;
		cv::boxFilter(pre_result, temp_result, -1, window_size);
		return temp_result;
	};

	return disparitywise_calculator(cost_func, window_sum, base_size, range);
}

#endif // COSTMAP_CREATORS_H
