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

class RegionInterval;
typedef std::vector<std::pair<std::size_t, std::size_t> > neighbor_vector;

/**
 * @brief The RegionDescriptor class saves the shape of an region. It holds a vector of line intervalls which shapes the region
 * Additionally it saves a bounding box of the region, the size (in pixel).
 */
class RegionDescriptor
{
public:
	std::vector<RegionInterval> lineIntervals;
	cv::Rect bounding_box;

	/*! Returns a cv::Mat with a binary image of the region. The image can be made bigger than actually nedded by setting a margin > 0*/
	cv::Mat getMask(int margin) const;
	/*! Returns all pixels of the region as 1D cv::Mat. You can move the region horizontally (by setting d != 0), where the pixels will be extracted. */
	cv::Mat getAsMat(const cv::Mat& src, int d) const;
	/*! Returns the size of the region in pixel (the actual region, not the bounding box) */
	int size() const;

	int m_size;

	neighbor_vector neighbors;

	cv::Vec3d average_color;


};

/**
 * @brief region_as_mat Returns a region as 1d matrix
 * @param src Image with values, from which the values for the new matrix will be taken from
 * @param pixel_idx Container with regions
 * @param d Applies a move in x-direction before taking values
 * @return Newly created cv::Mat with the size of the region and the values from src
 */
cv::Mat region_as_mat(const cv::Mat& src, const std::vector<RegionInterval> &pixel_idx, int d);

int getSizeOfRegion(const std::vector<RegionInterval>& intervals);
void calculate_average_color(RegionDescriptor& region, const cv::Mat& lab_image);
std::vector<RegionInterval> getDilatedRegion(RegionDescriptor &cregion, unsigned int dilate_grow, cv::Mat base);
void setMask(const cv::Mat &mask, std::vector<RegionInterval>& pixel_idx, int py, int px, int height, int width);

cv::Mat lab_to_bgr(const cv::Mat& src);
cv::Mat bgr_to_lab(const cv::Mat& src);

#endif // REGION_DESCRIPTOR_H
