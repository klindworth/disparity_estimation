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

#ifndef REGION_H
#define REGION_H

#include <opencv2/core/core.hpp>

#include <vector>
#include <array>
#include "stereotask.h"
#include "costmap_utils.h"
#include <segmentation/region_descriptor.h>
#include <segmentation/segmentation.h>

class RegionInterval;

class MutualRegion
{
public:
	MutualRegion() : index(0), percent(0.0f) {}
	MutualRegion(std::size_t idx) : index(idx), percent(0.0f) {}
	MutualRegion(std::size_t idx, float per) : index(idx), percent(per) {}

	//std::size_t index;
	unsigned int index;
	float percent;
};

class EstimationStep
{
public:
	short searchrange_start, searchrange_end;
	short disparity;
	float costs;
	float optimization_costs;
	short base_disparity;
};

class disparity_region : public region_descriptor
{
public:
	disparity_region();
	std::vector<RegionInterval> warped_interval;
	std::vector<std::vector< MutualRegion >> other_regions;
	cv::Mat_<float> disparity_costs;
	cv::Mat_<float> optimization_energy;

	stat_t stats;
	short disparity;
	short base_disparity;
	short disparity_offset;
	unsigned char dilation;
	char old_dilation;

	//std::vector<EstimationStep> results;
	MutualRegion getMutualRegion(std::size_t idx, std::size_t disparity_idx);
};

struct RegionContainer : public segmentation_image<disparity_region>
{
	StereoSingleTask task;
	//std::vector<short> disparity;
};

cv::Mat disparity_by_segments(const RegionContainer &container);

void fillRegionContainer(std::shared_ptr<RegionContainer>& result, StereoSingleTask& task, std::shared_ptr<segmentation_algorithm>& algorithm);

int reenumerate(cv::Mat& labels, int old_count);
void replace_neighbor_idx(std::vector<region_descriptor>& regions, std::size_t old_idx, std::size_t new_idx);

void generate_stats(std::vector<disparity_region>& regions, const StereoSingleTask& task, const int delta);
void generate_stats(disparity_region& region, const StereoSingleTask& task, int delta);

void labelLRCheck(RegionContainer& base, const RegionContainer& match, int delta);
void refreshWarpedIdx(RegionContainer& container);
std::vector<RegionInterval> filtered_region(int width, const std::vector<RegionInterval> &pixel_idx, int d);

bool checkLabelsIntervalsInvariant(const std::vector<disparity_region>& regions, const cv::Mat_<int>& labels, int segcount);

#endif // REGION_H
