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

#ifndef MISC_H
#define MISC_H

#include "genericfunctions.h"
#include "region_descriptor.h"

template<typename T>
cv::Mat regionWiseSet(const StereoSingleTask& task, const std::vector<SegRegion>& regions, std::function<T(const SegRegion& region)> func)
{
	return regionWiseSet<T>(task.base.size(), regions, func);
}

template<typename T>
cv::Mat regionWiseSet(const StereoSingleTask& task, const std::vector<SegRegion>& regions, const std::size_t exclude, T defaultValue, std::function<T(const SegRegion& region)> func)
{
	cv::Mat_<T> result = cv::Mat_<T>(task.base.size(), defaultValue);

	const std::size_t regions_count = regions.size();
	#pragma omp parallel for default(none) shared(result, regions, func)
	for(std::size_t i = 0; i < regions_count; ++i)
	{
		if(i != exclude)
			intervals::setRegionValue<T>(result, regions[i].lineIntervals, func(regions[i]));
	}

	return result;
}

template<typename T>
cv::Mat regionWiseImage(StereoSingleTask& task, std::vector<SegRegion>& regions, std::function<T(const SegRegion& region)> func)
{
	return getValueScaledImage<T, unsigned char>(regionWiseSet<T>(task, regions, func));
}

template<typename T, typename reg_type>
cv::Mat regionWiseImage(cv::Size size, std::vector<reg_type>& regions, std::function<T(const reg_type& region)> func)
{
	return getValueScaledImage<T, unsigned char>(regionWiseSet<T, reg_type>(size, regions, func));
}

#endif // MISC_H
