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
#include "region_descriptor.h"

class RegionInterval;

class MutualRegion
{
public:
	MutualRegion() : index(0), percent(0.0f) {}
	MutualRegion(std::size_t idx) : index(idx), percent(0.0f) {}
	MutualRegion(std::size_t idx, float per) : index(idx), percent(per) {}

	std::size_t index;
	float percent;
};

class SegRegion : public RegionDescriptor
{
public:
	SegRegion();
	std::vector<RegionInterval> warped_interval;
	std::vector<std::vector< MutualRegion >> other_regions;
	cv::Mat_<float> disparity_costs;
	cv::Mat_<float> confidence;
	cv::Mat_<float> optimization_energy;


	stat_t stats;
	float entropy;
	short disparity;
	short base_disparity;
	short disparity_offset;
	//short optimization_minimum;
	unsigned char dilation;
	char old_dilation;
	int damping_history;

	//int out_of_image;
	//std::array<int, 5> occlusion;
	//float occ_value;

	float confidence3;

	MutualRegion getMutualRegion(std::size_t idx, std::size_t disparity_idx);
};

class RegionContainer
{
public:
	StereoSingleTask task;
	std::vector<SegRegion> regions;
	cv::Mat labels;
};

template<typename T>
inline void parallel_region(std::vector<SegRegion>& regions, T func)
{
	parallel_region(regions.begin(), regions.end(), func);
}

cv::Mat getDisparityBySegments(const RegionContainer &container);

void refreshBoundingBoxes(const cv::Mat& labels, std::vector<SegRegion>& regions);

int reenumerate(cv::Mat& labels, int old_count);
void replace_neighbor_idx(std::vector<RegionDescriptor>& regions, std::size_t old_idx, std::size_t new_idx);

void generateStats(std::vector<SegRegion>& regions, const StereoSingleTask& task, const int delta);
void generateStats(SegRegion& region, const StereoSingleTask& task, int delta);

void labelLRCheck(const cv::Mat &labelsBase, const cv::Mat &labelsMatch, std::vector<SegRegion>& regions, const short dispMin, const short dispMax);
void labelLRCheck(const cv::Mat& labelsBase, const cv::Mat& labelsMatch, std::vector<SegRegion>& regions, StereoSingleTask& task, int delta);
void refreshWarpedIdx(RegionContainer& container);
float getOtherRegionsAverage(const std::vector<SegRegion> &container, const std::vector<MutualRegion> &cdisp, std::function<float(const SegRegion&)> func);
std::pair<float,float> getOtherRegionsAverageCond(const std::vector<SegRegion>& container, const std::vector<MutualRegion>& cdisp, std::function<float(const SegRegion&)> func, std::function<float(const SegRegion&)> cond_eval);

float getNeighborhoodsAverage(const std::vector<SegRegion>& container, const std::vector<std::pair<std::size_t, std::size_t>>& neighbors, std::function<float(const SegRegion&)> func);
float getWeightedNeighborhoodsAverage(const std::vector<SegRegion>& container, const std::vector<std::pair<std::size_t, std::size_t>>& neighbors, std::function<float(const SegRegion&)> func);
std::pair<float,float> getColorWeightedNeighborhoodsAverage(const cv::Vec3d& base_color, double color_trunc, const std::vector<SegRegion>& container, const std::vector<std::pair<std::size_t, std::size_t>>& neighbors, std::function<float(const SegRegion&)> func);

void calculate_all_average_colors(const cv::Mat &image, std::vector<SegRegion> &regions);

std::vector<RegionInterval> getFilteredPixelIdx(int width, const std::vector<RegionInterval> &pixel_idx, int d);

void setMask(const cv::Mat &mask, std::vector<RegionInterval>& pixel_idx, int py, int px, int height, int width);

bool checkLabelsIntervalsInvariant(const std::vector<SegRegion>& regions, const cv::Mat& labels, int segcount);
#endif // REGION_H
