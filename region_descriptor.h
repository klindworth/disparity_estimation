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

class RegionDescriptor
{
public:
	std::vector<RegionInterval> lineIntervals;
	cv::Rect bounding_box;

	cv::Mat getMask(int margin);
	cv::Mat getAsMat(const cv::Mat& src, int d);
	int size() const;

	int m_size;

	std::vector<std::pair<std::size_t, std::size_t>> neighbors;

	cv::Vec3d average_color;

	cv::Mat getRegionMask(int margin) const;
};

cv::Mat getRegionAsMat(const cv::Mat& src, const std::vector<RegionInterval> &pixel_idx, int d);
int getSizeOfRegion(const std::vector<RegionInterval>& intervals);
void calculate_average_color(RegionDescriptor& region, const cv::Mat& lab_image);
std::vector<RegionInterval> getDilatedRegion(RegionDescriptor &cregion, unsigned int dilate_grow, cv::Mat base);

#endif // REGION_DESCRIPTOR_H
