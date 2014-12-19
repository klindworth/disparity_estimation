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

#ifndef DISPARITY_UTILS_H
#define DISPARITY_UTILS_H

#include <opencv2/core/core.hpp>

#include "stereotask.h"

namespace disparity {

/**
 * Calls for every pixel a function with the coordinates of the original pixel and the warped pixel and passes additionally the disparity.
 * The function is only called, if the coordinate in the warped space is still valid, that means within the bodunaries of the image
 * @param disparity Matrix with the disparity
 * @param scaling Scaling factor of the passed disparity matrix. In most cases the disparity matrix is scaled due to subsampling, while the matrix remains integer.
 * That means, if you have a subsampling factor of four, your disparity matrix is scaled by four.
 * @param func Function that will be called for every pixel
 */
template<typename disparity_type, typename T>
void foreach_warped_pixel(const cv::Mat& disparity, float scaling, T func)
{
	#pragma omp parallel for
	for(int y = 0; y < disparity.rows; ++y)
	{
		const disparity_type* disp_ptr = disparity.ptr<disparity_type>(y,0);

		for(int j = 0; j < disparity.cols; ++j)
		{
			disparity_type cdisp = *disp_ptr++;
			int x = j + cdisp * scaling;

			if(x >= 0 && x < disparity.cols)
				func(cv::Point(j,y), cv::Point(x,y), cdisp);
		}
	}
}

/**
 * Creates a matrix, in which every positions contains the number of correspondences or explained differently: The number of pixels which warp to that position
 * @param disparity Matrix with the disparity
 * @param scaling Scaling factor of the passed disparity matrix. In most cases the disparity matrix is scaled due to subsampling, while the matrix remains integer.
 * That means, if you have a subsampling factor of four, your disparity matrix is scaled by four.
 */
template<typename disparity_type>
cv::Mat occlusion_stat(const cv::Mat& disparity, float scaling = 1.0f)
{
	cv::Mat stat_image(disparity.size(), CV_8UC1, cv::Scalar(0));

	foreach_warped_pixel<disparity_type>(disparity, scaling, [&](cv::Point, cv::Point warped_pos, disparity_type){
		stat_image.at<unsigned char>(warped_pos)++;
		assert(stat_image.at<unsigned char>(warped_pos) < 50);

	});

	return stat_image;
}

template<typename disparity_type>
cv::Mat occlusion_map(const cv::Mat& disparity, const cv::Mat& warped, float scaling = 1.0f)
{
	cv::Mat occ_image(disparity.size(), CV_8UC1, cv::Scalar(0));

	foreach_warped_pixel<disparity_type>(disparity, scaling, [&](cv::Point pos, cv::Point warped_pos, disparity_type disp){
		if(warped.at<disparity_type>(warped_pos) != disp )
			occ_image.at<unsigned char>(pos) = 255;
	});

	return occ_image;
}

///warps an image
template<typename image_type, typename disparity_type>
cv::Mat warp_image(const cv::Mat& image, const cv::Mat& disparity, float scaling = 1.0f)
{
	cv::Mat warpedImage(image.size(), image.type(), cv::Scalar(0));

	foreach_warped_pixel<disparity_type>(disparity, scaling, [&](cv::Point pos, cv::Point warped_pos, disparity_type){
		warpedImage.at<image_type>(warped_pos) = image.ptr<image_type>(pos);
	});

	return warpedImage;
}

template<typename T>
inline T absmax(const T& v1, const T& v2)
{
	if(std::abs(v1) > std::abs(v2))
		return v1;
	else
		return v2;
}

template<typename disparity_type>
cv::Mat warp_disparity(const cv::Mat& disparity, float scaling = 1.0f)
{
	cv::Mat warpedImage(disparity.size(), disparity.type(), cv::Scalar(0));

	foreach_warped_pixel<disparity_type>(disparity, scaling, [&](cv::Point, cv::Point warped_pos, disparity_type disp){
		warpedImage.at<disparity_type>(warped_pos) = -absmax(warpedImage.at<disparity_type>(warped_pos), disp);
	});

	return warpedImage;
}

template<typename cost_class, typename data_type>
cv::Mat_<short> wta_disparity(cv::Mat base, data_type data, int dispMin, int dispMax)
{
	cv::Mat_<short> result = cv::Mat_<short>(base.size(), 0);

	cost_class cost_agg(base, data, dispMin);

	#pragma omp parallel for
	for(int y = 0; y< base.size[0]; ++y)
	{
		for(int x = 0; x < base.size[1]; ++x)
		{
			int disp_start = std::min(std::max(x+dispMin, 0), base.size[1]-1) - x;
			int disp_end   = std::max(std::min(x+dispMax, base.size[1]-1), 0) - x;

			assert(disp_start-dispMin >= 0);
			assert(disp_start-dispMin < dispMax - dispMin + 1);
			assert(disp_end-disp_start < dispMax - dispMin + 1);
			assert(disp_end-disp_start >= 0);

			short cdisp = 0;
			float min_cost = std::numeric_limits<float>::max();

			for(int d = disp_start; d <= disp_end; ++d)
			{
				float cost = cost_agg(y,x,d);
				if(cost < min_cost)
				{
					min_cost = cost;
					cdisp = d;
				}
			}
			result(y,x) = cdisp;
		}
	}

	return result;
}

cv::Mat create_from_costmap(const cv::Mat &cost_map_org, int dispMin, int subsample);
cv::Mat create_image(const cv::Mat &disparity);

}

inline std::pair<short,short> getSubrange(short baseDisparity, short delta, const single_stereo_task& task)
{
	if(delta == 0)
		return std::make_pair(task.dispMin, task.dispMax);
	else
	{
		short start = std::max(baseDisparity - delta, task.dispMin);
		short end   = std::min(baseDisparity + delta, task.dispMax);

		return std::make_pair(start, end);
	}
}

inline std::pair<short,short> getSubrangeIdx(short baseDisparity, short delta, const single_stereo_task& task)
{
	auto range = getSubrange(baseDisparity, delta, task);
	range.first -= task.dispMin;
	range.second -= task.dispMin;
	return range;
	/*short start = std::max(baseDisparity - delta, task.dispMin) - task.dispMin;
	short end   = std::min(baseDisparity + delta, task.dispMax) - task.dispMin;

	return std::make_pair(start, end);*/
}

inline bool gotDisparity(short disparity, short baseDisparity, short delta, const single_stereo_task& task)
{
	auto range = getSubrange(baseDisparity, delta, task);
	return disparity >= range.first && range.second <= range.second;
}



#endif // DISPARITY_UTILS_H
