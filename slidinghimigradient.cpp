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

#include "slidinghimigradient.h"

#include "slidingGradient.h"
#include "it_metrics.h"
#include "debugmatstore.h"
#include "stereotask.h"

#include <iostream>

cv::Mat onestepSlidingInfoGradient(StereoSingleTask task, std::function<cv::Mat(StereoSingleTask)> func, int windowsize)
{
	cv::Mat cost_gradient = slidingGradient(task, windowsize);
	//cv::Mat cost_gradient = slidingGradient<windowsizeGrad>(task);
	filterGradientCostmap(cost_gradient, 10);
	cv::Mat invert_grad = -cost_gradient;
	matstore.addMat(task, invert_grad, "inv_gradient", windowsize);
	double min, max;
	cv::minMaxIdx(cost_gradient, &min, &max);
	cost_gradient /= (float)max;
	cost_gradient = 1-cost_gradient;

	//cv::Mat invert_grad = -cost_gradient;
	matstore.addMat(task, cost_gradient, "gradient", windowsize);

	cv::Mat cost_map = func(task);
	matstore.addMat(task, cost_map, "info-func", 11); //TODO: replace 11

	cv::Mat comb = cost_map.mul(cost_gradient);
	matstore.addMat(task, comb, "comb", 15);

	return comb;
}

std::function<cv::Mat(StereoSingleTask)> gradient_enhancer_bind(std::function<cv::Mat(StereoSingleTask)> func, int windowsize)
{
	using namespace std::placeholders;
	return std::bind(onestepSlidingInfoGradient, _1, func, windowsize);
}

void scaleUpCostMap(cv::Mat& cost_map_src, cv::Mat& cost_map_dst, float oldscale)
{
	/*float scalex = cost_map_src.size[1]/cost_map_dst.size[1];
	float scaley = cost_map_src.size[0]/cost_map_dst.size[0];

	assert(std::abs(scalex - scaley) < 0.001f); //check for same scaling in x and y direction
	oldscale = scalex;*/

	if(std::abs(oldscale - 1.0f) < 0.001)
	{
		cost_map_dst = cost_map_src;
		return;
	}
	for(int i = 0; i < cost_map_dst.size[0]; ++i)
	{
		for(int j = 0; j < cost_map_dst.size[1]; ++j)
		{
			for(int k = 0; k < cost_map_dst.size[2]; ++k)
				cost_map_dst.at<float>(i, j, k) = cost_map_src.at<float>(i*oldscale, j*oldscale, k*oldscale);
		}
	}
}

cv::Mat genericScaledProcessing(StereoSingleTask task, int border, std::function<cv::Mat(StereoSingleTask)> func)
{
	int disparityRange = task.dispMax - task.dispMin + 1;
	int sz[] = {task.base.size[0], task.base.size[1], disparityRange};
	cv::Mat cost_complete = cv::Mat(3, sz, CV_32FC1);
	memset(cost_complete.data, 0, sizeof(float)*cost_complete.size[0]*cost_complete.size[1]*disparityRange);

	//{scale, weight)
	std::vector<std::pair<float, float>> steps = {{0.5f, 0.25f}, {0.75f, 0.25f}, {1.0f, 0.5f}};

	for(auto& cstep : steps)
	{
		std::cout << "processing scale: " << cstep.first << std::endl;

		//downscale images
		cv::Mat baseScaled;
		cv::resize(task.baseGray, baseScaled, cv::Size(), cstep.first, cstep.first, cv::INTER_LANCZOS4);

		cv::Mat matchScaled;
		cv::resize(task.matchGray, matchScaled, cv::Size(), cstep.first, cstep.first, cv::INTER_LANCZOS4);

		//run the algorithm
		StereoSingleTask scaledTask;
		scaledTask.baseGray = baseScaled;
		scaledTask.matchGray = matchScaled;
		scaledTask.dispMin = task.dispMin * cstep.first;
		scaledTask.dispMax = task.dispMax * cstep.first;

		cv::Mat scaledCostmap = func(scaledTask);
		matstore.addMat(scaledTask, scaledCostmap, "scale", border);

		resetBorder<float>(scaledCostmap, border, 0);
		//upscale the costmap
		int sz[] = {cost_complete.size[0], cost_complete.size[1], cost_complete.size[2]};
		cv::Mat costUpScaled(3, sz, cost_complete.type());
		scaleUpCostMap(scaledCostmap, costUpScaled, cstep.first);

		cost_complete += costUpScaled*cstep.second;
	}

	return cost_complete;
}


