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

#ifndef REGION_DESCRIPTOR_H
#define REGION_DESCRIPTOR_H

#include <opencv2/core/core.hpp>

class region_interval;
typedef std::vector<std::pair<std::size_t, std::size_t> > neighbor_vector;

/**
 * @brief The RegionDescriptor class saves the shape of an region. It holds a vector of line intervalls which shapes the region
 * Additionally it saves a bounding box of the region, the size (in pixel).
 */
class region_descriptor
{
public:
	std::vector<region_interval> lineIntervals;
	neighbor_vector neighbors;
	cv::Vec3d average_color;
	cv::Rect bounding_box;
	cv::Point avg_point;

	//! Returns a cv::Mat with a binary image of the region. The image can be made bigger than actually nedded by setting a margin > 0
	cv::Mat mask(int margin) const;

	//! Returns all pixels of the region as 1D cv::Mat. You can move the region horizontally (by setting d != 0), where the pixels will be extracted.
	cv::Mat as_mat(const cv::Mat& src, int d) const;

	//! Returns the size of the region in pixel (the actual region, not the bounding box)
	int size() const;

	int m_size;
};

/**
 * @brief region_as_mat Returns a region as 1d matrix
 * @param src Image with values, from which the values for the new matrix will be taken from
 * @param pixel_idx Container with regions
 * @param d Applies a move in x-direction before taking values
 * @return Newly created cv::Mat with the size of the region and the values from src
 */
cv::Mat region_as_mat(const cv::Mat& src, const std::vector<region_interval> &pixel_idx, int d);

int size_of_region(const std::vector<region_interval>& intervals);
void calculate_average_color(region_descriptor& region, const cv::Mat& lab_image);
std::vector<region_interval> dilated_region(region_descriptor &cregion, unsigned int dilate_grow, cv::Mat base);

void create_region_from_mask(std::vector<region_interval>& region, const cv::Mat &mask, int py, int px, int height, int width);

cv::Mat lab_to_bgr(const cv::Mat& src);
cv::Mat bgr_to_lab(const cv::Mat& src);

#endif // REGION_DESCRIPTOR_H
