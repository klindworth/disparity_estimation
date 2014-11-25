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

#include <segmentation/intervals.h>
#include <segmentation/intervals_algorithms.h>
#include "misc.h"
#include "disparity_utils.h"
#include <segmentation/region_descriptor_algorithms.h>

DisparityRegion::DisparityRegion()
{
	old_dilation = -1;
	dilation = 0;
}

std::vector<RegionInterval> filtered_region(int width, const std::vector<RegionInterval> &old_region, int d)
{
	std::vector<RegionInterval> filtered;
	filtered.reserve(old_region.size());

	for(const RegionInterval& cinterval : old_region)
	{
		int lower = std::max(cinterval.lower+d, 0)-d;
		int upper = std::min(cinterval.upper+d, width)-d;
		if(upper - lower> 0)
			filtered.emplace_back(cinterval.y, lower, upper);
	}

	return filtered;
}

void labelLRCheck(const cv::Mat_<int>& labelsMatch, DisparityRegion& region, const short dispMin, const short dispMax)
{
	const int dispRange = dispMax-dispMin + 1;
	region.other_regions = std::vector<std::vector<MutualRegion>>(dispRange);
	//TODO segment boxfilter?
	for(int i = 0; i < dispRange; ++i)
	{
		int cdisparity = i + dispMin;
		sparse_histogramm hist;
		foreach_warped_region_point(region.lineIntervals.begin(), region.lineIntervals.end(), labelsMatch.cols, cdisparity, [&](cv::Point pt)
		{
			hist.increment(labelsMatch(pt));
		});

		region.other_regions[i].reserve(hist.size());
		double normalizer = 1.0/hist.total();
		for(auto it = hist.begin(); it != hist.end(); ++it)
		{
			double mutual_percent = (double)it->second * normalizer;
			region.other_regions[i].push_back(MutualRegion(it->first, mutual_percent));
		}
	}
}

void labelLRCheck(RegionContainer& base, const RegionContainer& match, int delta)
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
		labelLRCheck(match.labels, base.regions[j], range.first, range.second);
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
		cregion.warped_interval = filtered_region(container.task.base.cols, cregion.lineIntervals, cregion.disparity);
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
