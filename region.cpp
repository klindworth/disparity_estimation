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
#include "fast_array.h"
#include "slidingEntropy.h"
#include "sparse_counter.h"

#include <iterator>

#include <opencv2/highgui/highgui.hpp>


#include "costmap_creators.h"

#include <cstdlib>

#include "intervals.h"
#include "intervals_algorithms.h"
#include "misc.h"

/*void RegionDescriptor::setMask(cv::Mat mask)
{

}*/

void getRegionAsMatInternal(const cv::Mat& src, const std::vector<RegionInterval> &pixel_idx, int d, cv::Mat& dst, int elemSize)
{
	unsigned char *dst_ptr = dst.data;
	for(const RegionInterval& cinterval : pixel_idx)
	{
		int x = cinterval.lower + d;
		assert(x < src.size[1]);
		assert(x >= 0);
		int length = cinterval.length();
		assert(x + length <= src.size[1]);
		const unsigned char* src_ptr = src.data + cinterval.y * elemSize * src.size[1] + x * elemSize;
		memcpy(dst_ptr, src_ptr, elemSize*length);
		dst_ptr += length*elemSize;
	}
}

cv::Mat getRegionAsMat(const cv::Mat& src, const std::vector<RegionInterval> &pixel_idx, int d)
{
	int length = getSizeOfRegion(pixel_idx);

	int dim3 = src.dims == 2 ? 1 : src.size[2];
	cv::Mat region(length, dim3, src.type());
	getRegionAsMatInternal(src, pixel_idx, d, region, dim3*src.elemSize());

	return region;
}

void setMask(const cv::Mat& mask, std::vector<RegionInterval>& pixel_idx, int py, int px, int height, int width)
{
	pixel_idx.clear();

	int y_max = std::min(std::min(mask.rows, height), height-py);
	int x_max = std::min(std::min(mask.cols, width), width-px);

	int x_min = std::max(0, -px); // x+pxy >= 0
	int y_min = std::max(0, -py);

	assert(y_max >= 0 && x_max >= 0);

	auto factory = [&](int y, int lower, int upper, unsigned char value) {
		if(value == 255)
		{
			assert(y-y_min < height-py && lower+px+x_min >= 0 && y+py+y_min >= 0 && upper+px+x_min > 0 && upper+x_min <= width-px);
			pixel_idx.push_back(RegionInterval(y+py+y_min, lower+px+x_min, upper+px+x_min));
		}
	};

	cv::Mat_<unsigned char> mask2 = mask(cv::Range(y_min, y_max), cv::Range(x_min, x_max));
	intervals::convertDifferential<unsigned char>(mask2, factory);
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

float getOtherRegionsAverage(const std::vector<SegRegion>& container, const std::vector<MutualRegion>& cdisp, std::function<float(const SegRegion&)> func)
{
	float result = 0.0f;
	for(const MutualRegion& cval : cdisp)
	{
		result += cval.percent * func(container[cval.index]);
	}
	return result;
}

std::pair<float,float> getOtherRegionsAverageCond(const std::vector<SegRegion>& container, const std::vector<MutualRegion>& cdisp, std::function<float(const SegRegion&)> func, std::function<float(const SegRegion&)> cond_eval)
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

float getNeighborhoodsAverage(const std::vector<SegRegion>& container, const std::vector<std::pair<std::size_t, std::size_t>>& neighbors, std::function<float(const SegRegion&)> func)
{
	float result = 0.0f;
	for(const std::pair<std::size_t, std::size_t>& cpair : neighbors)
	{
		result += func(container[cpair.first]);
	}
	return result/neighbors.size();
}

float getWeightedNeighborhoodsAverage(const std::vector<SegRegion>& container, const std::vector<std::pair<std::size_t, std::size_t>>& neighbors, std::function<float(const SegRegion&)> func)
{
	float result = 0.0f;
	float sum_weight = 0.0f;
	for(const std::pair<std::size_t, std::size_t>& cpair : neighbors)
	{
		result += cpair.second*func(container[cpair.first]);
		sum_weight += cpair.second;
	}
	return result/sum_weight;
}

std::pair<float,float> getColorWeightedNeighborhoodsAverage(const cv::Vec3d& base_color, double color_trunc, const std::vector<SegRegion>& container, const std::vector<std::pair<std::size_t, std::size_t>>& neighbors, std::function<float(const SegRegion&)> func)
{
	float result = 0.0f;
	float sum_weight = 0.0f;
	for(const std::pair<std::size_t, std::size_t>& cpair : neighbors)
	{
		float diff = color_trunc - std::min(cv::norm(base_color - container[cpair.first].average_color), color_trunc);

		result += diff*func(container[cpair.first]);
		sum_weight += diff;
	}
	sum_weight = std::max(std::numeric_limits<float>::min(), sum_weight);
	return std::make_pair(result/sum_weight, sum_weight);
}

inline void labelLRCheck(const cv::Mat& labelsBase, const cv::Mat& labelsMatch, SegRegion& region, const short dispMin, const short dispMax)
{
	const int dispRange = dispMax-dispMin + 1;
	region.other_regions = std::vector<std::vector<MutualRegion>>(dispRange);
	for(int i = 0; i < dispRange; ++i)
	{
		int cdisparity = i + dispMin;
		sparse_histogramm hist;
		std::vector<RegionInterval> filteredIntervals = getFilteredPixelIdx(labelsBase.cols, region.region.lineIntervals, cdisparity);
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

void labelLRCheck(const cv::Mat& labelsBase, const cv::Mat& labelsMatch, std::vector<SegRegion>& regions, const short dispMin, const short dispMax)
{
	const int dispRange = dispMax-dispMin + 1;
	const std::size_t regions_count = regions.size();
	#pragma omp parallel for default(none) shared(regions, labelsBase, labelsMatch)
	for(std::size_t j = 0; j < regions_count; ++j)
	{
		labelLRCheck(labelsBase, labelsMatch, regions[j], dispMin, dispMax);
	}
}

void labelLRCheck(const cv::Mat& labelsBase, const cv::Mat& labelsMatch, std::vector<SegRegion>& regions, StereoSingleTask& task, int delta)
{
	//const int dispRange = dispMax-dispMin + 1;
	const std::size_t regions_count = regions.size();
	#pragma omp parallel for default(none) shared(regions, labelsBase, labelsMatch, delta, task)
	for(std::size_t j = 0; j < regions_count; ++j)
	{
		std::pair<std::size_t, std::size_t> range;
		if(delta == 0)
			range = std::make_pair(task.dispMin, task.dispMax);
		else
			range = getSubrange(regions[j].base_disparity, delta, task);
		labelLRCheck(labelsBase, labelsMatch, regions[j], range.first, range.second);
	}
}

/**
 * @brief getRegionVector Returns a vector with Region instances, for each label one entry in the vector will be created
 * @param labels cv::Mat with all labels for each specific pixel
 * @param regions_count Total number of different Labels
 * @return Returns a vector with Region instances
 */
std::vector<SegRegion> getRegionVector(const cv::Mat& labels, int regions_count)
{
	std::vector<SegRegion> regions(regions_count);

	auto factory = [&](std::size_t y, std::size_t lower, std::size_t upper, int value) {
		assert(value < regions_count && value >= 0);
		regions[value].region.lineIntervals.push_back(RegionInterval(y, lower, upper));
	};

	const cv::Mat_<int> labels_typed = labels;
	intervals::convertDifferential<int>(labels_typed, factory);

	#pragma omp parallel for default(none) shared(regions, regions_count)
	for(int i = 0; i < regions_count; ++i)
	{
		//compute size
		regions[i].old_dilation = -1;
		regions[i].dilation = 0;
		regions[i].size = getSizeOfRegion(regions[i].region.lineIntervals);

		assert(regions[i].size != 0);
	}

	refreshBoundingBoxes(labels, regions);

	return regions;
}

int getSizeOfRegion(const std::vector<RegionInterval>& intervals)
{
	int length = 0;
	for(const RegionInterval& cinterval : intervals)
		length += cinterval.length();

	return length;
}

void refreshBoundingBoxes(const cv::Mat& labels, std::vector<SegRegion>& regions)
{
	const std::size_t regions_count = regions.size();
	#pragma omp parallel for default(none) shared(labels, regions)
	for(std::size_t i = 0; i < regions_count; ++i)
	{
		regions[i].region.bounding_box.x = labels.cols;
		regions[i].region.bounding_box.y = labels.rows;
		regions[i].region.bounding_box.height = 0;
		regions[i].region.bounding_box.width = 0;
	}

	const int* clabel = labels.ptr<int>(0);
	for(int y = 0; y < labels.rows; ++y)
	{
		for(int x = 0; x < labels.cols; ++x)
		{
			int i = *clabel++;//labels.at<int>(y,x);
			cv::Rect& crect = regions[i].region.bounding_box;

			assert(i < regions.size() && i >= 0);

			crect.x = std::min(x, crect.x);
			crect.y = std::min(y, crect.y);
			crect.width  = std::max(x, crect.width); //save temp the maximal x coordinate
			crect.height = std::max(y, crect.height);
		}
	}

	#pragma omp parallel for default(none) shared(labels, regions)
	for(std::size_t i = 0; i < regions_count; ++i)
	{
		regions[i].region.bounding_box.width  -= regions[i].region.bounding_box.x - 1;
		regions[i].region.bounding_box.height -= regions[i].region.bounding_box.y - 1;
	}
}

inline std::vector<std::pair<std::size_t, std::size_t> >::iterator find_neighbor(std::vector<std::pair<std::size_t, std::size_t> >& container, std::size_t val)
{
	return std::find_if(container.begin(), container.end(), [=](const std::pair<std::size_t, std::size_t>& cpair){return cpair.first == val;});
}

inline void save_neighbors(std::vector<SegRegion>& regions, std::size_t val1, std::size_t val2)
{
	typedef std::vector<std::pair<std::size_t, std::size_t> > neighbor_vector;
	if(val1 != val2)
	{
		neighbor_vector& reg1 = regions[val1].neighbors;
		neighbor_vector& reg2 = regions[val2].neighbors;
		neighbor_vector::iterator it = find_neighbor(reg1, val2);
		if(it == reg1.end())
		{
			reg1.push_back(std::make_pair(val2, 1));
			reg2.push_back(std::make_pair(val1, 1));
		}
		else
		{
			it->second += 1;
			neighbor_vector::iterator it2 = find_neighbor(reg2, val1);
			it2->second += 1;
		}
	}
}

void generate_neighborhood(cv::Mat &labels, std::vector<SegRegion> &regions)
{
	for(SegRegion& cregion : regions)
		cregion.neighbors.clear();

	for(int i = 0; i < labels.rows; ++i)
	{
		for(int j = 1; j < labels.cols; ++j)
			save_neighbors(regions, labels.at<int>(i,j), labels.at<int>(i, j-1));
	}
	for(int i = 1; i < labels.rows; ++i)
	{
		for(int j = 0; j < labels.cols; ++j)
			save_neighbors(regions, labels.at<int>(i,j), labels.at<int>(i-1, j));
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

void replace_neighbor_idx(std::vector<SegRegion>& regions, std::size_t old_idx, std::size_t new_idx)
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

bool checkNeighborhoodInvariant(std::vector<SegRegion>& regions, std::size_t regions_count)
{
	for(std::size_t i = 0; i < regions_count; ++i)
	{
		SegRegion& cregion = regions[i];
		const std::size_t neigh_count = cregion.neighbors.size();

		for(std::size_t j = 0; j < neigh_count; ++j)
		{
			std::size_t c_idx = cregion.neighbors[j].first;

			assert(c_idx < regions_count);

			SegRegion& cneighbor = regions[c_idx];

			bool found = false;
			const std::size_t inner_neigh_count = cneighbor.neighbors.size();
			for(std::size_t k = 0; k < inner_neigh_count; ++k)
			{
				if(cneighbor.neighbors[k].first == i)
				{
					found = true;
					break;
				}
			}

			assert(found);
		}
	}
	return true;
}

bool checkLabelsIntervalsInvariant(const std::vector<SegRegion>& regions, const cv::Mat& labels, int segcount)
{
	int pixelcount = 0;
	for(int i = 0; i < segcount; ++i)
	{
		const SegRegion& cregion = regions[i];
		int segsize = getSizeOfRegion(cregion.region.lineIntervals);
		assert(segsize == cregion.size);
		assert(segsize != 0);
		pixelcount += segsize;
		for(const RegionInterval& cinterval : cregion.region.lineIntervals)
		{
			for(int x = cinterval.lower; x < cinterval.upper; ++x)
			{
				int val = labels.at<int>(cinterval.y, x);
				assert(val == i);
				if(val != i)
					return false;
			}
		}
	}
	assert(pixelcount == labels.rows * labels.cols);
	if(pixelcount != labels.rows * labels.cols)
		return false;
	else
		return true;
}

void generateStats(std::vector<SegRegion>& regions, const StereoSingleTask& task, const int delta)
{
	const std::size_t regions_count = regions.size();
	#pragma omp parallel for default(none) shared(regions, task)
	for(std::size_t i = 0; i < regions_count; ++i)
		generateStats(regions[i], task, delta);
}

void generateStats(SegRegion& region, const StereoSingleTask& task, int delta)
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

void calculate_average_color(SegRegion& region, const cv::Mat& lab_image)
{
	cv::Mat values = getRegionAsMat(lab_image, region.region.lineIntervals, 0);
	cv::Scalar means = cv::mean(values);

	region.average_color[0] = means[0];
	region.average_color[1] = means[1];
	region.average_color[2] = means[2];
}

void calculate_all_average_colors(const cv::Mat& image, std::vector<SegRegion>& regions)
{
	const std::size_t regions_count = regions.size();

	cv::Mat lab_image = bgr_to_lab(image);
	cv::Mat lab_double_image;
	lab_image.convertTo(lab_double_image, CV_64FC3);

	#pragma omp parallel for default(none) shared(regions, lab_double_image)
	for(std::size_t i = 0; i < regions_count; ++i)
		calculate_average_color(regions[i], lab_double_image);
}

cv::Mat getDisparityBySegments(const RegionContainer& container)
{
	return regionWiseSet<short>(container.task, container.regions, [](const SegRegion& cregion){return cregion.disparity;});
}

cv::Mat getDisparityBySegments(const RegionContainer& container, const std::size_t exclude)
{
	return regionWiseSet<short>(container.task, container.regions, exclude, 0, [](const SegRegion& cregion){return cregion.disparity;});
}

void refreshWarpedIdx(RegionContainer& container)
{
	const std::size_t regions_count = container.regions.size();
	#pragma omp parallel for default(none) shared(container)
	for(std::size_t i = 0; i < regions_count; ++i)
	{
		SegRegion& cregion = container.regions[i];
		cregion.warped_interval.clear();
		cregion.warped_interval.reserve(cregion.region.lineIntervals.size());
		cregion.warped_interval = getFilteredPixelIdx(container.task.base.cols, cregion.region.lineIntervals, cregion.disparity);
		for(RegionInterval& cinterval : cregion.warped_interval)
		{
			cinterval.lower += cregion.disparity;
			cinterval.upper += cregion.disparity;
		}

		//cregion.out_of_image = cregion.size - getSizeOfRegion(cregion.warped_interval);
	}
}

MutualRegion SegRegion::getMutualRegion(std::size_t idx, std::size_t disparity_idx)
{
	assert(disparity_idx < other_regions.size());
	auto it = std::find_if(other_regions[disparity_idx].begin(), other_regions[disparity_idx].end(), [=](const MutualRegion& creg){return (creg.index == idx);});
	if(it == other_regions[disparity_idx].end())
		return MutualRegion(idx);
	else
		return *it;
}
