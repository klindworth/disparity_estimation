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

#include "disparity_region.h"
#include "genericfunctions.h"
#include "sparse_counter.h"
#include "disparity_utils.h"

#include <iterator>
#include <segmentation/intervals.h>
#include <segmentation/intervals_algorithms.h>
#include <segmentation/region_descriptor_algorithms.h>

#include "disparity_region_algorithms.h"

std::vector<region_interval> filtered_region(int width, const std::vector<region_interval> &old_region, int d)
{
	std::vector<region_interval> filtered;
	filtered.reserve(old_region.size());

	for(const region_interval& cinterval : old_region)
	{
		int lower = std::max(cinterval.lower+d, 0)-d;
		int upper = std::min(cinterval.upper+d, width)-d;
		if(upper - lower> 0)
			filtered.emplace_back(cinterval.y, lower, upper);
	}

	return filtered;
}

void determine_corresponding_regions(const cv::Mat_<int>& labelsMatch, disparity_region& region, const disparity_range& drange)
{
	const int dispRange = drange.size();
	region.corresponding_regions = std::vector<std::vector<corresponding_region>>(dispRange);
	//TODO segment boxfilter?
	sparse_histogramm hist;
	for(int i = 0; i < dispRange; ++i)
	{
		int cdisparity = i + drange.start();
		hist.reset();
		foreach_warped_region_point(region.lineIntervals.begin(), region.lineIntervals.end(), labelsMatch.cols, cdisparity, [&](cv::Point pt)
		{
			hist.increment(labelsMatch(pt));
		});

		region.corresponding_regions[i].reserve(hist.size());
		double normalizer = hist.total() > 0 ? 1.0/hist.total() : 0.0;
		auto hend = hist.cend();
		for(auto it = hist.cbegin(); it != hend; ++it)
		{
			double mutual_percent = (double)it->second * normalizer;
			region.corresponding_regions[i].push_back(corresponding_region(it->first, mutual_percent));
		}
		/*for(auto it = hist.begin(); it != hist.end(); ++it)
		{
			double mutual_percent = (double)it->second * normalizer;
			region.corresponding_regions[i].push_back(corresponding_region(it->first, mutual_percent));
		}*/
		/*for(const auto& it : hist)
		{
			double mutual_percent = (double)it.second * normalizer;
			region.corresponding_regions[i].push_back(corresponding_region(it.first, mutual_percent));
		}*/
	}
}

void check_corresponding_regions(const region_container& base, const region_container& match)
{
	for(const disparity_region& cregion : base.regions)
	{
		for(const std::vector<corresponding_region>& cdisp : cregion.corresponding_regions)
		{
			for(const corresponding_region& cmutual : cdisp)
			{
				assert(cmutual.index < match.regions.size());
			}
		}
	}
}

void determine_corresponding_regions(region_container& base, const region_container& match, int delta)
{
	parallel_region(base.regions.begin(), base.regions.end(), [&](disparity_region& cregion){
		disparity_range range = task_subrange(base.task, cregion.base_disparity, delta);
		determine_corresponding_regions(match.labels, cregion, range);
	});
}

int reenumerate_segmentation_labels(cv::Mat& labels, int old_count)
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

void replace_neighbor_idx(std::vector<region_descriptor>& regions, std::size_t old_idx, std::size_t new_idx)
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

disparity_map disparity_by_segments(const region_container& container)
{
	return disparity_map(region_descriptors::set_regionwise<short>(container, [](const disparity_region& cregion){return cregion.disparity;}), 1);
	//return regionWiseSet<short>(container.task.base.size(), container.regions, [](const DisparityRegion& cregion){return cregion.disparity;});
}

void refresh_warped_regions(region_container& container)
{
	const std::size_t regions_count = container.regions.size();
	#pragma omp parallel for default(none) shared(container)
	for(std::size_t i = 0; i < regions_count; ++i)
	{
		disparity_region& cregion = container.regions[i];
		cregion.warped_interval.clear();
		cregion.warped_interval.reserve(cregion.lineIntervals.size());
		cregion.warped_interval = filtered_region(container.task.base.cols, cregion.lineIntervals, cregion.disparity);
		for(region_interval& cinterval : cregion.warped_interval)
		{
			cinterval.lower += cregion.disparity;
			cinterval.upper += cregion.disparity;
		}

		//cregion.out_of_image = cregion.size - getSizeOfRegion(cregion.warped_interval);
	}
}

corresponding_region disparity_region::get_corresponding_region(std::size_t idx, std::size_t disparity_idx)
{
	assert(disparity_idx < corresponding_regions.size());
	auto it = std::find_if(corresponding_regions[disparity_idx].begin(), corresponding_regions[disparity_idx].end(), [=](const corresponding_region& creg){return (creg.index == idx);});
	if(it == corresponding_regions[disparity_idx].end())
		return corresponding_region(idx);
	else
		return *it;
}

