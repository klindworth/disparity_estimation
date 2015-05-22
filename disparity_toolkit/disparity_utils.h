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
#include "disparity_range.h"
#include <omp.h>

namespace disparity {

/**
 * Returns the value with the bigger absolute value. It returns the original value, not it's absolute one.
 * E.g. absmax(-4, 3) returns -4
 */
template<typename T>
inline T absmax(const T& v1, const T& v2)
{
	if(std::abs(v1) > std::abs(v2))
		return v1;
	else
		return v2;
}

/**
 * Calls for every pixel a function with the coordinates of the original pixel and the warped pixel and passes additionally the disparity.
 * The function is only called, if the coordinate in the warped space is still valid, that means within the boundaries of the image
 * @param disparity Matrix with the disparity
 * @param scaling Scaling factor of the passed disparity matrix. In most cases the disparity matrix is scaled due to subsampling, while the matrix remains integer.
 * That means, if you have a subsampling factor of four, your disparity matrix is scaled by four.
 * @param func Function that will be called for every pixel (pos, warped_pos, disparity)
 */
template<typename disparity_type, typename T>
void foreach_warped_pixel(const cv::Mat_<disparity_type>& disparity, float scaling, T func)
{
	#pragma omp parallel for
	for(int y = 0; y < disparity.rows; ++y)
	{
		const disparity_type* disp_ptr = disparity[y];

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
 * Calls for every pixel a function with the coordinates of the original pixel and the warped pixel and passes additionally the disparity.
 * The function is only called, if the coordinate in the warped space is still valid, that means within the boundaries of the image.
 * Compared to the non-unique version, this function ensures, that the lambda is only called once for every position
 * @param disparity Matrix with the disparity
 * @param scaling Scaling factor of the passed disparity matrix. In most cases the disparity matrix is scaled due to subsampling, while the matrix remains integer.
 * That means, if you have a subsampling factor of four, your disparity matrix is scaled by four.
 * @param func Function that will be called for every pixel
 */
template<typename disparity_type, typename T>
void foreach_warped_pixel_unique(const cv::Mat_<disparity_type>& disparity, float scaling, T func)
{
	#pragma omp parallel for
	for(int y = 0; y < disparity.rows; ++y)
	{
		const disparity_type* disp_ptr = disparity[y];
		std::vector<disparity_type> warp_buf(disparity.cols, 0);

		for(int j = 0; j < disparity.cols; ++j)
		{
			disparity_type cdisp = *disp_ptr++;
			int x = j + cdisp * scaling;

			if(x >= 0 && x < disparity.cols)
				warp_buf[x] = absmax(warp_buf[x], cdisp);
		}

		disp_ptr = disparity[y];
		for(int j = 0; j < disparity.cols; ++j)
		{
			disparity_type cdisp = *disp_ptr++;
			int x = j + cdisp * scaling;

			if(x >= 0 && x < disparity.cols)
			{
				if(warp_buf[x] == cdisp)
					func(cv::Point(j,y), cv::Point(x,y), cdisp);
			}
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
cv::Mat_<unsigned char> occlusion_stat(const cv::Mat_<disparity_type>& disparity, float scaling = 1.0f)
{
	cv::Mat_<unsigned char> stat_image(disparity.size(), static_cast<unsigned char>(0));

	foreach_warped_pixel<disparity_type>(disparity, scaling, [&](cv::Point, cv::Point warped_pos, disparity_type){
		stat_image(warped_pos)++;
	});

	return stat_image;
}

template<typename disparity_type>
cv::Mat_<unsigned char> occlusion_map(const cv::Mat_<disparity_type>& disparity, const cv::Mat_<disparity_type>& warped, float scaling = 1.0f)
{
	cv::Mat_<unsigned char> occ_image(disparity.size(), static_cast<unsigned char>(0));

	foreach_warped_pixel<disparity_type>(disparity, scaling, [&](cv::Point pos, cv::Point warped_pos, disparity_type disp){
		if(warped(warped_pos) != disp )
			occ_image(pos) = 255;
	});

	return occ_image;
}



/**
 * @brief Warps an image.
 * This function uses a disparity map to simulate a different perspective of an image
 * @param image Image that will be warped
 * @param disparity Disparity map that determines the amount of pixels that a pixel will be moved to simulate the different perspective
 * @param init Value for undefinied values (undefinied/not existing in the simulated perspective). Default: 0
 * @param scaling The scaling factor will be applied on the disparity values. This is usefull for over/undersampled integer disparity maps.
 * If a disparity value e.g. of four means one pixel, you pass 0.25 as scaling factor. Default: 1
 * @return Warped image
 */
template<typename image_type, typename disparity_type>
cv::Mat_<image_type> warp_image(const cv::Mat_<image_type>& image, const cv::Mat_<disparity_type>& disparity, image_type init = 0, float scaling = 1.0f)
{
	cv::Mat_<image_type> warpedImage(image.size(), init);

	foreach_warped_pixel_unique<disparity_type>(disparity, scaling, [&](cv::Point pos, cv::Point warped_pos, disparity_type){
		warpedImage(warped_pos) = image(pos);
	});

	return warpedImage;
}

/**
 * @brief Warps a disparity with itself and adjust it's sign for backtransformation
 * @param disparity Disparity map that will be wapred and that determines the amount of pixels that a disparity value will be moved to simulate the different perspective
 * @param scaling The scaling factor will be applied on the disparity values. This is usefull for over/undersampled integer disparity maps.
 * If a disparity value e.g. of four means one pixel, you pass 0.25 as scaling factor. Default: 1
 * @return Warped disparity
 */
template<typename disparity_type>
cv::Mat_<disparity_type> warp_disparity(const cv::Mat_<disparity_type>& disparity, float scaling = 1.0f)
{
	cv::Mat_<disparity_type> warpedImage(disparity.size(), static_cast<disparity_type>(0));

	foreach_warped_pixel<disparity_type>(disparity, scaling, [&](cv::Point, cv::Point warped_pos, disparity_type disp){
		warpedImage(warped_pos) = absmax(warpedImage(warped_pos), static_cast<disparity_type>(-disp));
	});

	return warpedImage;
}

template<typename disparity_type>
disparity_map warp_disparity(const disparity_map& disparity)
{
	return disparity_map(warp_disparity(disparity, 1.0f/disparity.sampling), disparity.sampling);
}

template<typename cost_class, typename data_type>
disparity_map wta_disparity(cv::Mat base, data_type data, const disparity_range range)
{
	cv::Mat_<short> result = cv::Mat_<short>(base.size(), 0);

	cost_class cost_agg(base, data, range.start());
	using cost_type = typename cost_class::result_type;

	#pragma omp parallel for
	for(int y = 0; y< base.size[0]; ++y)
	{
		for(int x = 0; x < base.size[1]; ++x)
		{
			const disparity_range crange = range.restrict_to_image(x, base.size[1]);

			short cdisp = 0;
			cost_type min_cost = std::numeric_limits<cost_type>::max();

			for(int d = crange.start(); d <= crange.end(); ++d)
			{
				cost_type cost = cost_agg(y,x,d);
				if(cost < min_cost)
				{
					min_cost = cost;
					cdisp = d;
				}
			}
			result(y,x) = cdisp;
		}
	}

	return disparity_map(result, 1);
}

template<typename T>
T disparity_interpolate(const T* cost_ptr, std::size_t min_d, std::size_t range)
{
	auto interpolation = [](T d0, T d1, T d2) {
		return 0.5*(d0-d2)/(d0-2.0f*d1+d2);
	};

	//T ndisp;
	if(min_d > 0 && min_d < range-2 && //check range
		(cost_ptr[min_d-1]-2.0f*cost_ptr[min_d]+cost_ptr[min_d+1]) > 0)// avoid division by zero (that part is taken from the denominator)
	{
		return min_d + interpolation(cost_ptr[min_d-1], cost_ptr[min_d], cost_ptr[min_d+1]);
	}
	else
		return min_d;
}

template<typename cost_class, typename data_type>
disparity_map wta_disparity_sampling(cv::Mat base, data_type data, const disparity_range range, int sampling)
{
	cv::Mat_<short> result = cv::Mat_<short>(base.size(), 0);

	cost_class cost_agg(base, data, range.start());
	using cost_type = typename cost_class::result_type;

	std::vector<std::vector<cost_type>> cost_temp(omp_get_max_threads(), std::vector<cost_type>(range.size(), std::numeric_limits<cost_type>::max()));

	#pragma omp parallel for
	for(int y = 0; y< base.size[0]; ++y)
	{
		for(int x = 0; x < base.size[1]; ++x)
		{
			const disparity_range crange = range.restrict_to_image(x, base.size[1]);

			short cdisp = 0;
			cost_type min_cost = std::numeric_limits<cost_type>::max();

			for(int d = crange.start(); d <= crange.end(); ++d)
			{
				cost_type cost = cost_agg(y,x,d);
				cost_temp[omp_get_thread_num()][crange.index(d)] = cost;
				if(cost < min_cost)
				{
					min_cost = cost;
					cdisp = d;
				}
			}

			result(y,x) = disparity_interpolate(cost_temp[omp_get_thread_num()].data(), crange.index(cdisp), crange.size()) * sampling + range.start()*sampling;
		}
	}

	return disparity_map(result, sampling);
}

template<typename T>
std::size_t minimal_cost_disparity(const T* cost_ptr, int range, int dispMin)
{
	T min_cost = std::numeric_limits<T>::max();
	std::size_t min_d = 0;
	for(int d = 0; d < range; ++d)
	{
		if((cost_ptr[d] < min_cost && d+dispMin > 0) || (cost_ptr[d] <= min_cost && d+dispMin <= 0))
		{
			min_cost = cost_ptr[d];
			min_d = d;
		}
	}
	return min_d;
}

template<typename T>
short minimal_cost_disparity_with_interpolation(const T* cost_ptr, disparity_range drange, int sampling = 1)
{
	std::size_t min_d = minimal_cost_disparity(cost_ptr, drange.size(), drange.start());

	return sampling > 1 ? std::round(disparity_interpolate(cost_ptr, min_d, drange.size())*sampling+drange.start()*sampling) : min_d;
}

disparity_map create_from_costmap(const cv::Mat &cost_map_org, int dispMin, int sampling);
cv::Mat create_image(const cv::Mat &disparity);

}

inline disparity_range task_subrange(const single_stereo_task& task, int pivot, int delta)
{
	if(delta == 0)
		return task.range;
	else
		return task.range.subrange(pivot, delta);
}

#endif // DISPARITY_UTILS_H
