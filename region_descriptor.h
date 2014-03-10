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
#include "intervals.h"
#include "intervals_algorithms.h"

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

template<typename Iterator, typename T>
inline void parallel_region(Iterator begin, Iterator end, T func)
{
	const std::size_t regions_count = std::distance(begin, end);
	#pragma omp parallel for default(none) shared(begin, func)
	for(std::size_t i = 0; i < regions_count; ++i)
		func(*(begin + i));
}

template<typename T, typename reg_type>
cv::Mat_<T> regionWiseSet(cv::Size size, const std::vector<reg_type>& regions, std::function<T(const reg_type& region)> func)
{
	cv::Mat_<T> result(size, 0);

	parallel_region(regions.begin(), regions.end(), [&](const reg_type& region) {
		intervals::setRegionValue<T>(result, region.lineIntervals, func(region));
	});

	return result;
}

cv::Mat getRegionAsMat(const cv::Mat& src, const std::vector<RegionInterval> &pixel_idx, int d);
int getSizeOfRegion(const std::vector<RegionInterval>& intervals);
void calculate_average_color(RegionDescriptor& region, const cv::Mat& lab_image);

#endif // REGION_DESCRIPTOR_H
