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

#include "intervals_algorithms.h"
#include "region.h"

#include <numeric>

void analyzeDisparityRange2(SegRegion& region)
{
	cv::Mat temp = region.disparity_costs.reshape(0,1);
	region.stats.minima_ranges.clear();
	intervals::convertMinimaRanges<float>(temp, std::back_inserter(region.stats.minima_ranges), region.stats.mean - region.stats.stddev);

	int minima_width = 0;
	for(const RegionInterval& interval : region.stats.minima_ranges)
		minima_width += interval.length();

	region.stats.confidence_range = region.stats.minima_ranges.size();
	region.stats.confidence_variance = (float)minima_width/region.disparity_costs.total();

	region.confidence3 = 0.0f;
	if(region.stats.minima_ranges.size() > 1)
	{
		std::vector<float> range_sums(region.stats.minima_ranges.size());
		for(std::size_t i = 0; i < region.stats.minima_ranges.size(); ++i)
		{
			float* ptr = region.disparity_costs.ptr<float>(region.stats.minima_ranges[i].lower);
			std::size_t length = region.stats.minima_ranges[i].length();
			range_sums[i] = std::accumulate(ptr, ptr + length, 0.0f) / length;
		}
		float max_range = *(std::max_element(range_sums.begin(), range_sums.end()));
		std::transform(range_sums.begin(), range_sums.end(), range_sums.begin(), [=](const float& cval){return max_range-cval;});
		float conf_sum = std::accumulate(range_sums.begin(), range_sums.end(), 0.0f);
		max_range = *(std::max_element(range_sums.begin(), range_sums.end()));
		region.confidence3 = max_range/conf_sum/region.stats.confidence_variance;
	}
	else if(region.stats.minima_ranges.size() == 1)
		region.confidence3 = 1/region.stats.confidence_variance;
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
	cstat.bad_minima.clear();
	cstat.minima.clear();
	float good_threshold = mean-1.5*stddev;
	float bad_threshold = mean;

	if(derived_ptr[0] > 0 && src_ptr[0] < good_threshold)
		cstat.minima.push_back(0);
	else if(derived_ptr[0] > 0 && src_ptr[0] < bad_threshold)
		cstat.bad_minima.push_back(0);
	for(int k = 1; k < range-1; ++k)
	{
		if(derived_ptr[k-1] <= 0 && derived_ptr[k] >= 0)
		{
			//value check
			if(src_ptr[k] < good_threshold)
				cstat.minima.push_back(k);
			else if(src_ptr[k] < bad_threshold)
				cstat.bad_minima.push_back(k);
		}
	}

	typedef std::pair<int, float> rank_type;
	std::vector<rank_type> ranking;
	for(int idx : cstat.minima)
		ranking.push_back(std::make_pair(idx, src_ptr[idx]));
	for(int idx : cstat.bad_minima)
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
	cstat.confidence = confidence;
	cstat.disparity_idx = min_disp;
}

DataStore2D<stat_t> analyzeCostmap(const cv::Mat& src)
{
	assert(src.dims == 3);

	//cv::Mat derivedCost = deriveCostmap(src);

	DataStore2D<stat_t> stat_store(src.size[0], src.size[1]);

	float *derivedCost = new float[src.size[2]-1];

	for(int i = 0; i < src.size[0]; ++i)
	{
		for(int j = 0; j < src.size[1]; ++j)
		{
			stat_store(i,j) = stat_t();
			derivePartialCostmap(src.ptr<float>(i,j,0), derivedCost, src.size[2]);
			analyzeDisparityRange(stat_store(i,j), src.ptr<float>(i,j,0), derivedCost, src.size[2]);
		}
	}

	delete[] derivedCost;

	return stat_store;
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


//sums all values up in a window and saves the sum in the middlepoint (like boxfilter - but here for 3D matrix)
cv::Mat windowSum(cv::Mat& cost_map_old, int windowsize)
{
	assert(cost_map_old.dims == 3);
	assert(cost_map_old.size[2] % 4 == 0); //needed for SSE
	int sz[] = {cost_map_old.size[0], cost_map_old.size[1], cost_map_old.size[2]};
	cv::Mat cost_map = cv::Mat(3, sz, CV_32FC1);

	const int y_min = windowsize/2;
	const int y_max = cost_map_old.size[0] - windowsize/2;

	const int x_min = windowsize/2;
	const int x_max = cost_map_old.size[1] - windowsize/2;

	#pragma omp parallel for default(none) shared(cost_map, cost_map_old, windowsize)
	for(int y = y_min; y < y_max; ++y)
	{
		for(int x = x_min; x < x_max; ++x)
		{
			float *fixed_dst_ptr = cost_map.ptr<float>(y,x);
			memset(fixed_dst_ptr, 0, cost_map_old.size[2]*cost_map_old.elemSize());

			for(int i = y-windowsize/2; i <= y+windowsize/2; ++i)
			{
				for(int j = x-windowsize/2; j <= x+windowsize/2; ++j)
				{
					__m128* dst_ptr = (__m128*)fixed_dst_ptr;
					__m128* src_ptr = (__m128*)cost_map_old.ptr<float>(i,j);
					for(int k = 0; k < cost_map_old.size[2]; k+=4)
					{
						*dst_ptr = _mm_add_ps(*dst_ptr, *src_ptr++);
						++dst_ptr;
					}
				}
			}
		}
	}

	//TODO borders...

	return cost_map;
}

