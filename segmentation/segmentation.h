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

#ifndef SEGMENTATION_H
#define SEGMENTATION_H

#include <opencv2/core/core.hpp>
#include <memory>
#include "region_descriptor_algorithms.h"

class RegionContainer;
class fusion_work_data;

class segmentation_settings
{
public:
	std::string algorithm;
	int spatial_var;
	float color_var;
	int superpixel_size;
	float superpixel_compactness;

	bool enable_regionsplit;
	bool enable_fusion;
	bool enable_color_fusion;
};



class segmentation_algorithm {
public:
	/**
	 * @brief operator () Segments an image
	 * @param image Image, which should be segmented
	 * @param labels cv::Mat image with int as datatype. Counting from zero, out parameter
	 * @return Number of segments
	 */
	virtual int operator()(const cv::Mat& image, cv::Mat_<int>& labels) = 0;

	template<typename seg_image_type>
	std::shared_ptr<seg_image_type> getSegmentationImage(cv::Mat image)
	{
		std::shared_ptr<seg_image_type> result = std::make_shared<seg_image_type>();
		result->segment_count = this->operator ()(image, result->labels);
		result->regions = std::vector<typename seg_image_type::regions_type>(result->segment_count);
		result->image_size = image.size();

		fillRegionDescriptors(result->regions.begin(), result->regions.end(), result->labels);
		//refreshBoundingBoxes(result->regions.begin(), result->regions.end(), result->labels);
		generate_neighborhood(result->labels, result->regions);

		return result;
	}

	/**
	 * @brief cacheAllowed Returns, if caching for this segmentation algorithm is allowed
	 * @return
	 */
	virtual bool cacheAllowed() const { return true; }

	/**
	 * @brief cacheName Name of algorithm for filenames
	 * @return
	 */
	virtual std::string cacheName() const = 0;

	/**
	 * @brief refinementPossible Returns if it is possible to obtain a finer segmentation
	 * @return
	 */
	virtual bool refinementPossible() { return false; }

	/**
	 * @brief refine Returns a segmentation with smaller segments. The smaller segments lay in a bigger segments and not in more than one!
	 */
	virtual void refine(RegionContainer&) {}
};

std::shared_ptr<segmentation_algorithm> getSegmentationClass(const segmentation_settings& settings);

class fusion_work_data
{
public:
	fusion_work_data(std::size_t size) :
		visited(std::vector<unsigned char>(size, 0)),
		active(std::vector<unsigned char>(size, 1)),
		fused(std::vector<std::vector<std::size_t>>(size)),
		fused_with(std::vector<std::size_t>(size, 0))
	{
	}

	void visit_reset()
	{
		std::fill(visited.begin(), visited.end(), 0);
	}

	std::vector<unsigned char> visited;
	std::vector<unsigned char> active;
	std::vector<std::vector<std::size_t> > fused;
	std::vector<std::size_t> fused_with;
};

cv::Mat_<cv::Vec3b> getWrongColorSegmentationImage(cv::Mat_<int>& labels, int labelcount);

const cv::FileNode& operator>>(const cv::FileNode& stream, segmentation_settings& config);
cv::FileStorage& operator<<(cv::FileStorage& stream, const segmentation_settings& config);

#endif // SEGMENTATION_H
