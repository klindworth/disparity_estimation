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

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "intervals_algorithms.h"


void region_as_mat_internal(const cv::Mat& src, const std::vector<RegionInterval> &pixel_idx, int d, cv::Mat& dst, int elemSize)
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

cv::Mat region_as_mat(const cv::Mat& src, const std::vector<RegionInterval> &pixel_idx, int d)
{
	int length = size_of_region(pixel_idx);

	int dim3 = src.dims == 2 ? 1 : src.size[2];
	cv::Mat region(length, dim3, src.type());
	region_as_mat_internal(src, pixel_idx, d, region, dim3*src.elemSize());

	return region;
}

cv::Mat region_descriptor::mask(int margin) const
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

void create_region_from_mask(std::vector<RegionInterval>& region, const cv::Mat& mask, int py, int px, int height, int width)
{
	region.clear();

	int y_max = std::min(std::min(mask.rows, height), height-py);
	int x_max = std::min(std::min(mask.cols, width), width-px);

	int x_min = std::max(0, -px); // x+pxy >= 0
	int y_min = std::max(0, -py);

	assert(y_max >= 0 && x_max >= 0);

	auto factory = [&](int y, int lower, int upper, unsigned char value) {
		if(value == 255)
		{
			assert(y-y_min < height-py && lower+px+x_min >= 0 && y+py+y_min >= 0 && upper+px+x_min > 0 && upper+x_min <= width-px);
			region.push_back(RegionInterval(y+py+y_min, lower+px+x_min, upper+px+x_min));
		}
	};

	cv::Mat_<unsigned char> mask2 = mask(cv::Range(y_min, y_max), cv::Range(x_min, x_max));
	intervals::convert_differential<unsigned char>(mask2, factory);
}

std::vector<RegionInterval> dilated_region(region_descriptor& cregion, unsigned int dilate_grow, cv::Mat base)
{
	if(dilate_grow > 0)
	{
		std::vector<RegionInterval> dil_pixel_idx;
		cv::Mat strucElem = cv::Mat(1+2*dilate_grow,1+2*dilate_grow, CV_8UC1, cv::Scalar(1));
		cv::Mat mask = cregion.mask(dilate_grow);
		cv::dilate(mask, mask, strucElem);

		create_region_from_mask(dil_pixel_idx, mask, cregion.bounding_box.y - dilate_grow, cregion.bounding_box.x - dilate_grow, base.size[0], base.size[1]);
		//TODO: refresh bounding boxes
		return dil_pixel_idx;
	}
	else
		return cregion.lineIntervals;
}

cv::Mat region_descriptor::as_mat(const cv::Mat& src, int d) const
{
	return region_as_mat(src, lineIntervals, d);
}


int size_of_region(const std::vector<RegionInterval>& intervals)
{
	int length = 0;
	for(const RegionInterval& cinterval : intervals)
		length += cinterval.length();

	return length;
}

int region_descriptor::size() const
{
	return size_of_region(lineIntervals);
}

void calculate_average_color(region_descriptor& region, const cv::Mat& lab_image)
{
	cv::Mat values = region_as_mat(lab_image, region.lineIntervals, 0);
	cv::Scalar means = cv::mean(values);

	region.average_color[0] = means[0];
	region.average_color[1] = means[1];
	region.average_color[2] = means[2];
}

cv::Mat lab_to_bgr(const cv::Mat& src)
{
	cv::Mat temp = src;
	if(src.type() == CV_64FC3)
		src.convertTo(temp, CV_32FC3);
	cv::Mat bgr_float_image;
	cv::cvtColor(temp, bgr_float_image, CV_Lab2BGR);
	cv::Mat result;
	bgr_float_image.convertTo(result, CV_8UC3, 255);

	return result;
}

cv::Mat bgr_to_lab(const cv::Mat& src)
{
	cv::Mat bgr_float_image;
	src.convertTo(bgr_float_image, CV_32FC3, 1/255.0);

	cv::Mat result;
	cv::cvtColor(bgr_float_image,result, CV_BGR2Lab);
	//showMatTable(result);

	return result;
}
