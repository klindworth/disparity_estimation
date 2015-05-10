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
#include "stereotask.h"
#include <segmentation/region_descriptor.h>
#include <segmentation/segmentation.h>

class region_interval;

class corresponding_region
{
public:
	corresponding_region() : index(0), percent(0.0f) {}
	corresponding_region(std::size_t idx) : index(idx), percent(0.0f) {}
	corresponding_region(std::size_t idx, float per) : index(idx), percent(per) {}

	//std::size_t index;
	unsigned int index; ///< Index of the corresponding region
	float percent; ///< Amount of overlapping
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
	std::vector<region_interval> warped_interval;
	std::vector<std::vector< corresponding_region >> corresponding_regions;
	cv::Mat_<float> disparity_costs;
	cv::Mat_<float> optimization_energy;

	short disparity;
	short base_disparity;
	short disparity_offset;

	//std::vector<EstimationStep> results;
	corresponding_region get_corresponding_region(std::size_t idx, std::size_t disparity_idx);
};

struct region_container : public segmentation_image<disparity_region>
{
	region_container(const single_stereo_task& ptask) : task(ptask) {}
	single_stereo_task task;
	//std::vector<short> disparity;
};

cv::Mat_<short> disparity_by_segments(const region_container &container);

void fill_region_container(std::shared_ptr<region_container>& result, single_stereo_task& task, std::shared_ptr<segmentation_algorithm>& algorithm);

int reenumerate_segmentation_labels(cv::Mat& labels, int old_count);
void replace_neighbor_idx(std::vector<region_descriptor>& regions, std::size_t old_idx, std::size_t new_idx);

void determine_corresponding_regions(region_container& base, const region_container& match, int delta);
void refresh_warped_regions(region_container& container);
std::vector<region_interval> filtered_region(int width, const std::vector<region_interval> &pixel_idx, int d);

#endif // REGION_H
