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

#include "region_descriptor.h"

#include "region.h"

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

cv::Mat RegionDescriptor::getMask(int margin)
{
	assert(bounding_box.width > 0 && bounding_box.height > 0);

	cv::Mat mask = cv::Mat::zeros(bounding_box.height + 2*margin, bounding_box.width + 2*margin, CV_8UC1);
	for(const RegionInterval& cinterval : lineIntervals)
	{
		assert(cinterval.y + margin - bounding_box.y >= 0 && cinterval.y + margin - bounding_box.y < bounding_box.height + 2*margin);
		assert(cinterval.lower + margin - bounding_box.x >= 0 && cinterval.lower + margin - bounding_box.x + cinterval.length() < bounding_box.width + 2*margin);

		memset(mask.ptr<unsigned char>(cinterval.y + margin - bounding_box.y, cinterval.lower + margin - bounding_box.x), 255, cinterval.length());
	}
	return mask;
}

cv::Mat RegionDescriptor::getAsMat(const cv::Mat& src, int d)
{
	return getRegionAsMat(src, lineIntervals, d);
}


int getSizeOfRegion(const std::vector<RegionInterval>& intervals)
{
	int length = 0;
	for(const RegionInterval& cinterval : intervals)
		length += cinterval.length();

	return length;
}

int RegionDescriptor::size() const
{
	return getSizeOfRegion(lineIntervals);
}

void calculate_average_color(RegionDescriptor& region, const cv::Mat& lab_image)
{
	cv::Mat values = getRegionAsMat(lab_image, region.lineIntervals, 0);
	cv::Scalar means = cv::mean(values);

	region.average_color[0] = means[0];
	region.average_color[1] = means[1];
	region.average_color[2] = means[2];
}
