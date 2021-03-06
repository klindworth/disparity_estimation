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
#include "segmentation_refinement.h"

class segmentation_settings
{
public:
	std::string algorithm;
	int spatial_var;
	float color_var;
	int superpixel_size;
	float superpixel_compactness;
};


/**
 * @brief The segmentation_algorithm class provides a common interface for several segmentation algorithms.
 *
 * You can call a free function create_segmentation_instance for creating a instance of this class via string (e.g. "superpixel" for SLIC).
 * If you've an instance of this class, you can call segmentation_image to get a structure instance with all region descriptors etc.
 */
class segmentation_algorithm {
public:
	/**
	 * @brief operator () Segments an image
	 * @param image Image, which should be segmented
	 * @param labels cv::Mat image with int as datatype. Counting from zero, out parameter
	 * @return Number of segments
	 */
	virtual int operator()(const cv::Mat& image, cv::Mat_<int>& labels) = 0;

	template<typename seg_image_type, typename... arg_types>
	std::shared_ptr<seg_image_type> segmentation_image(cv::Mat image, arg_types... params)
	{
		std::shared_ptr<seg_image_type> result = std::make_shared<seg_image_type>(params...);
		result->segment_count = this->operator ()(image, result->labels);
		result->regions = std::vector<typename seg_image_type::regions_type>(result->segment_count);
		result->image_size = image.size();

		region_descriptors::fill(result->regions.begin(), result->regions.end(), result->labels);
		region_descriptors::generate_neighborhood(result->labels, result->regions);

		return result;
	}

	template<typename region_type>
	void refinement(region_type& container)
	{
		container.labels = segmentation_iteration(container.regions, container.task.base.size());

		region_descriptors::generate_neighborhood(container.labels, container.regions);
	}

	/**
	 * @brief name Name of the algorithm
	 * @return Name of the algorithm
	 */
	virtual std::string name() const = 0;
};

std::shared_ptr<segmentation_algorithm> create_segmentation_instance(const segmentation_settings& settings);

cv::Mat_<cv::Vec3b> getWrongColorSegmentationImage(cv::Mat_<int>& labels, int labelcount);

const cv::FileNode& operator>>(const cv::FileNode& stream, segmentation_settings& config);
cv::FileStorage& operator<<(cv::FileStorage& stream, const segmentation_settings& config);




#endif // SEGMENTATION_H
