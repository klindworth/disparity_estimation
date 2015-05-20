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

#include "disparity_utils.h"

#include "genericfunctions.h"

namespace disparity
{

typedef float costmap_type;

disparity_map create_from_costmap(const cv::Mat& cost_map, int dispMin, int sampling)
{
	cv::Mat_<short> result = cv::Mat::zeros(cost_map.size[0], cost_map.size[1], CV_16SC1);
	disparity_range drange(dispMin, dispMin + cost_map.size[2] - 1);

	for(int i = 0; i < cost_map.size[0]; ++i)
	{
		for(int j = 0; j < cost_map.size[1]; ++j)
		{
			const costmap_type *cost_ptr = cost_map.ptr<costmap_type>(i,j);
			result(i,j) = minimal_cost_disparity_with_interpolation(cost_ptr, drange, sampling);
		}
	}

	return disparity_map(result, sampling);
}

cv::Mat create_image(const cv::Mat& disparity)
{
	assert(disparity.type() == CV_16SC1);

	double mind;
	double maxd;
	cv::minMaxIdx(disparity, &mind, &maxd);

	short mins = mind;
	short maxs = maxd;
	float scale = 255.0f/(maxs-mins);

	cv::Mat temp;
	if(mins >= 0 && maxs >= 0)
		temp = (disparity - mins)*scale;
	else
		temp = (disparity - mins)*-scale+255;

	cv::Mat result;
	temp.convertTo(result, CV_8U);

	return result;
}

}
