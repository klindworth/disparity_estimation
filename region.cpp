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

#include "region.h"
#include "genericfunctions.h"
#include "sparse_counter.h"

#include <iterator>
#include <cstdlib>

#include "intervals.h"
#include "intervals_algorithms.h"
#include "misc.h"
#include "disparity_utils.h"
#include "region_descriptor_algorithms.h"

DisparityRegion::DisparityRegion()
{
	old_dilation = -1;
	dilation = 0;
}

std::vector<RegionInterval> getFilteredPixelIdx(int width, const std::vector<RegionInterval> &pixel_idx, int d)
{
	std::vector<RegionInterval> filtered;
	filtered.reserve(pixel_idx.size());

	for(const RegionInterval& cinterval : pixel_idx)
	{
		int lower = std::max(cinterval.lower+d, 0)-d;
		int upper = std::min(cinterval.upper+d, width)-d;
		if(upper - lower> 0)
			filtered.push_back(RegionInterval(cinterval.y, lower, upper));
	}

	return filtered;
}

float getOtherRegionsAverage(const std::vector<DisparityRegion>& container, const std::vector<MutualRegion>& cdisp, std::function<float(const DisparityRegion&)> func)
{
	float result = 0.0f;
	for(const MutualRegion& cval : cdisp)
	{
		result += cval.percent * func(container[cval.index]);
	}
	return result;
}

std::pair<float,float> getOtherRegionsAverageCond(const std::vector<DisparityRegion>& container, const std::vector<MutualRegion>& cdisp, std::function<float(const DisparityRegion&)> func, std::function<float(const DisparityRegion&)> cond_eval)
{
	float result = 0.0f;
	float cond_true = 0.0f;
	for(const MutualRegion& cval : cdisp)
	{
		if(cond_eval(container[cval.index]))
		{
			result += cval.percent * func(container[cval.index]);
			cond_true += cval.percent;
		}
	}
	return std::make_pair(result/cond_true, cond_true);
}

float getNeighborhoodsAverage(const std::vector<DisparityRegion>& container, const std::vector<std::pair<std::size_t, std::size_t>>& neighbors, std::function<float(const DisparityRegion&)> func)
{
	return getNeighborhoodsAverage(container, neighbors, 0.0f, func);
}

float getWeightedNeighborhoodsAverage(const std::vector<DisparityRegion>& container, const std::vector<std::pair<std::size_t, std::size_t>>& neighbors, std::function<float(const DisparityRegion&)> func)
{
	return getWeightedNeighborhoodsAverage(container, neighbors, 0.0f, func);
}

std::pair<float,float> getColorWeightedNeighborhoodsAverage(const cv::Vec3d& base_color, double color_trunc, const std::vector<DisparityRegion>& container, const std::vector<std::pair<std::size_t, std::size_t>>& neighbors, std::function<float(const DisparityRegion&)> func)
{
	float result = 0.0f;
	float sum_weight = 0.0f;
	for(const auto& cpair : neighbors)
	{
		float diff = color_trunc - std::min(cv::norm(base_color - container[cpair.first].average_color), color_trunc);

		result += diff*func(container[cpair.first]);
		sum_weight += diff;
	}
	sum_weight = std::max(std::numeric_limits<float>::min(), sum_weight);
	return std::make_pair(result/sum_weight, sum_weight);
}

void labelLRCheck(const cv::Mat& labelsBase, const cv::Mat& labelsMatch, DisparityRegion& region, const short dispMin, const short dispMax)
{
	const int dispRange = dispMax-dispMin + 1;
	region.other_regions = std::vector<std::vector<MutualRegion>>(dispRange);
	for(int i = 0; i < dispRange; ++i)
	{
		int cdisparity = i + dispMin;
		sparse_histogramm hist;
		std::vector<RegionInterval> filteredIntervals = getFilteredPixelIdx(labelsBase.cols, region.lineIntervals, cdisparity);
		for(const RegionInterval& cinterval : filteredIntervals)
		{
			for(int x = cinterval.lower; x < cinterval.upper; ++x)
				hist.increment(labelsMatch.at<int>(cinterval.y, x + cdisparity));
		}

		region.other_regions[i].reserve(hist.size());
		for(auto it = hist.begin(); it != hist.end(); ++it)
		{
			double mutual_percent = (double)it->second / hist.total();
			region.other_regions[i].push_back(MutualRegion(it->first, mutual_percent));
		}
	}
}

void labelLRCheck(RegionContainer& base, RegionContainer& match, int delta)
{
	const std::size_t regions_count = base.regions.size();
	#pragma omp parallel for default(none) shared(base, match, delta)
	for(std::size_t j = 0; j < regions_count; ++j)
	{
		std::pair<std::size_t, std::size_t> range;
		if(delta == 0)
			range = std::make_pair(base.task.dispMin, base.task.dispMax);
		else
			range = getSubrange(base.regions[j].base_disparity, delta, base.task);
		labelLRCheck(base.labels, match.labels, base.regions[j], range.first, range.second);
	}
}

int reenumerate(cv::Mat& labels, int old_count)
{
	std::vector<int> map(old_count, -1);
	int* ptr = labels.ptr<int>(0);
	int count = 0;
	for(std::size_t i = 0; i < labels.total(); ++i)
	{
		int old = *ptr;
		assert(old < old_count);
		if(map[old] == -1)
		{
			*ptr++ = count;
			map[old] = count++;
		}
		else
			*ptr++ = map[old];
	}
	return count;
}

void replace_neighbor_idx(std::vector<RegionDescriptor>& regions, std::size_t old_idx, std::size_t new_idx)
{
	for(std::pair<std::size_t, std::size_t>& cpair : regions[old_idx].neighbors)
	{
		for(std::pair<std::size_t, std::size_t>& cpair2 : regions[cpair.first].neighbors)
		{
			if(cpair2.first == old_idx)
				cpair2.first = new_idx;
		}
	}
}

bool checkLabelsIntervalsInvariant(const std::vector<DisparityRegion>& regions, const cv::Mat_<int>& labels, int segcount)
{
	return checkLabelsIntervalsInvariant(regions.begin(), regions.begin() + segcount, labels);
}

void generateStats(std::vector<DisparityRegion>& regions, const StereoSingleTask& task, const int delta)
{
	parallel_region(regions, [&](DisparityRegion& region) {
		generateStats(region, task, delta);
	});
}

void generateStats(DisparityRegion& region, const StereoSingleTask& task, int delta)
{
	auto range = getSubrange(region.base_disparity, delta, task);
	int len = range.second - range.first + 1;
	float *derived = new float[len-1];
	const float *costs = region.disparity_costs[0];
	derivePartialCostmap(costs, derived, len);

	analyzeDisparityRange(region.stats, costs, derived, len);
	analyzeDisparityRange2(region);

	delete[] derived;
}

void calculate_all_average_colors(const cv::Mat& image, std::vector<DisparityRegion>& regions)
{
	calculate_all_average_colors(image, regions.begin(), regions.end());
}

cv::Mat getDisparityBySegments(const RegionContainer& container)
{
	return regionWiseSet<short>(container, [](const DisparityRegion& cregion){return cregion.disparity;});
	//return regionWiseSet<short>(container.task.base.size(), container.regions, [](const DisparityRegion& cregion){return cregion.disparity;});
}

void refreshWarpedIdx(RegionContainer& container)
{
	const std::size_t regions_count = container.regions.size();
	#pragma omp parallel for default(none) shared(container)
	for(std::size_t i = 0; i < regions_count; ++i)
	{
		DisparityRegion& cregion = container.regions[i];
		cregion.warped_interval.clear();
		cregion.warped_interval.reserve(cregion.lineIntervals.size());
		cregion.warped_interval = getFilteredPixelIdx(container.task.base.cols, cregion.lineIntervals, cregion.disparity);
		for(RegionInterval& cinterval : cregion.warped_interval)
		{
			cinterval.lower += cregion.disparity;
			cinterval.upper += cregion.disparity;
		}

		//cregion.out_of_image = cregion.size - getSizeOfRegion(cregion.warped_interval);
	}
}

MutualRegion DisparityRegion::getMutualRegion(std::size_t idx, std::size_t disparity_idx)
{
	assert(disparity_idx < other_regions.size());
	auto it = std::find_if(other_regions[disparity_idx].begin(), other_regions[disparity_idx].end(), [=](const MutualRegion& creg){return (creg.index == idx);});
	if(it == other_regions[disparity_idx].end())
		return MutualRegion(idx);
	else
		return *it;
}
