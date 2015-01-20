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

#include "costmap_utils.h"

#include "genericfunctions.h"

#include <segmentation/intervals_algorithms.h>
#include "disparity_region.h"

#include <numeric>

template<typename value_type, typename InserterIterator>
void convert_minima_ranges(const cv::Mat_<value_type>& values, InserterIterator inserter, value_type threshold)
{
	auto factory = [&](std::size_t y, std::size_t lower, std::size_t upper, value_type value) {
		if(value < threshold)
		{
			*inserter = region_interval(y,lower,upper);
			++inserter;
		}
	};

	auto cmp_func = [=](value_type last, value_type current) {
		return (last > threshold && current > threshold) || (last <= threshold && current <= threshold);
	};

	intervals::convert_generic<value_type>(values, factory, cmp_func);
}

void analyzeDisparityRange2(const disparity_region& region, stat_t& stats)
{
	cv::Mat temp = region.disparity_costs.reshape(0,1);
	std::vector<region_interval> minima_ranges;
	convert_minima_ranges<float>(temp, std::back_inserter(minima_ranges), stats.mean - stats.stddev);

	int minima_width = 0;
	for(const region_interval& interval : minima_ranges)
		minima_width += interval.length();

	stats.confidence_range = minima_ranges.size();
	stats.confidence_variance = (float)minima_width/region.disparity_costs.total();
}

void analyzeDisparityRange(stat_t& cstat, const float* src_ptr, const float* derived_ptr, int range)
{
	double mean = 0.0f;
	double stddev = 0.0f;
	float minval = std::numeric_limits<float>::max();
	float maxval = -std::numeric_limits<float>::max();
	int meancounter = 0;
	int min_disp = 0;

	for(int k = 0; k < range; ++k)
	{
		float cval = src_ptr[k];
		if(cval != std::numeric_limits<float>::max() && cval != -std::numeric_limits<float>::max())
		{
			if(cval < minval)
			{
				minval = cval;
				min_disp = k;
			}
			maxval = std::max(maxval, cval);
			mean += cval;
			++meancounter;
		}
	}
	mean /= meancounter;
	for(int k = 0; k < range; ++k)
	{
		float cval = src_ptr[k];
		if(cval != std::numeric_limits<float>::max() && cval != -std::numeric_limits<float>::max())
			stddev += (cval-mean)*(cval-mean);
	}
	stddev /= meancounter;
	stddev = std::sqrt(stddev);

	int outliecounter = 0;
	float confidencesum = 0.0f;
	for(int k = 0; k < range; ++k)
	{
		float cval = src_ptr[k];
		if(cval < mean-stddev && cval != std::numeric_limits<float>::max() && cval != -std::numeric_limits<float>::max())
		{
			++outliecounter;
			if(k > 0 && k < range)
			{
				bool peak = cval < src_ptr[k-1] && cval < src_ptr[k+1];
				bool lightpeak = cval <= src_ptr[k-1] && cval <= src_ptr[k+1];
				if(peak)
				{
					confidencesum -= (std::abs(src_ptr[k-1]-mean)-stddev)*0.8;
					confidencesum -= (std::abs(src_ptr[k+1]-mean)-stddev)*0.8;
				}
				else if(lightpeak && !peak)
				{
					confidencesum -= (std::abs(src_ptr[k-1]-mean)-stddev)*0.5;
					confidencesum -= (std::abs(src_ptr[k+1]-mean)-stddev)*0.5;
				}
			}
			confidencesum += std::abs(cval-mean)-stddev;
		}
	}
	//confidencesum /= outliecounter;
	float confidence;
	if(minval < mean-stddev)
	{
		float devdiff = std::abs(minval-mean)-stddev;
		confidence = devdiff/confidencesum;
		confidence *= devdiff/stddev;
	}
	else
		confidence = 0.0f;

	//find maxima
	std::vector<short> bad_minima, minima;
	float good_threshold = mean-1.5*stddev;
	float bad_threshold = mean;

	if(derived_ptr[0] > 0 && src_ptr[0] < good_threshold)
		minima.push_back(0);
	else if(derived_ptr[0] > 0 && src_ptr[0] < bad_threshold)
		bad_minima.push_back(0);
	for(short k = 1; k < range-1; ++k)
	{
		if(derived_ptr[k-1] <= 0 && derived_ptr[k] >= 0)
		{
			//value check
			if(src_ptr[k] < good_threshold)
				minima.push_back(k);
			else if(src_ptr[k] < bad_threshold)
				bad_minima.push_back(k);
		}
	}

	typedef std::pair<short, float> rank_type;
	std::vector<rank_type> ranking;
	for(int idx : minima)
		ranking.push_back(std::make_pair(idx, src_ptr[idx]));
	for(int idx : bad_minima)
		ranking.push_back(std::make_pair(idx, src_ptr[idx]));

	if(mean != 0.0f)
	{
		auto less_func = [](const rank_type& lhs, const rank_type& rhs){return lhs.second < rhs.second;};
		std::sort(ranking.begin(), ranking.end(), less_func);
		std::for_each(ranking.begin(), ranking.end(), [&](rank_type& current){
			current.second -= bad_threshold;
			current.second /= -good_threshold;
		});
		float confidence2sum = std::accumulate(ranking.begin(), ranking.end(), 0.0f, [](const float& p1, const rank_type& p2){return p2.second+p1;});
		auto it_max = std::max_element(ranking.begin(), ranking.end(), less_func);
		if(it_max != ranking.end())
			cstat.confidence2 = it_max->second/confidence2sum;
		else
			cstat.confidence2 = 0.0f;
	}
	else
		cstat.confidence2 = 0.0f;


	cstat.mean = mean;
	cstat.stddev = stddev;
	cstat.min = minval;
	cstat.max = maxval;
	//cstat.confidence = confidence;
	cstat.disparity_idx = min_disp;
}

void derivePartialCostmap(const float* cost_map, float *result, int len)
{
	for(int k = 1; k < len; ++k)
	{
		result[k-1] = cost_map[k] - cost_map[k-1];
	}
}

cv::Mat deriveCostmap(const cv::Mat& cost_map)
{
	int sz[] = {cost_map.size[0], cost_map.size[1], cost_map.size[2]-1};
	cv::Mat result = cv::Mat(3, sz, CV_32FC1);

	for(int i = 0; i < cost_map.size[0]; ++i)
	{
		for(int j = 0; j < cost_map.size[1]; ++j)
			derivePartialCostmap(cost_map.ptr<float>(i,j,0), result.ptr<float>(i,j,0), cost_map.size[2]);
	}
	return result;
}

