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

#ifndef REFINEMENT_H
#define REFINEMENT_H


#include "window_size.h"
#include "slidingEntropy.h"
#include "costmap_creators.h"
#include "stereotask.h"
#include "disparity_region.h"

#include "debugmatstore.h"

#include <iostream>

class RefinementConfig
{
public:
	int min_windowsize;
	int max_windowsize;

	int deltaDisp;
	float entropy_threshold;

	int subsampling;
};

template<typename charT, typename traits>
inline std::basic_ostream<charT, traits>& operator<<(std::basic_ostream<charT, traits>& stream, const RefinementConfig& config)
{
	stream << "min_windowsize: " << config.min_windowsize << "\nmax_windowsize: " << config.max_windowsize << "\ndeltaDisp: " << config.deltaDisp;
	return stream;
}

cv::FileStorage& operator<<(cv::FileStorage& stream, const RefinementConfig& config);
cv::FileStorage& operator>>(cv::FileStorage& stream, RefinementConfig& config);

typedef std::function<cv::Mat(cv::Mat& initial_disparity, single_stereo_task& task, cv::Mat base_quant, cv::Mat match_quant, region_container& container, const RefinementConfig& config)> refinement_func_type;


template<typename cost_func, int quantizer>
cv::Mat refine_initial_disparity(cv::Mat& initial_disparity, single_stereo_task& task, cv::Mat base, cv::Mat match, region_container& container, const RefinementConfig& config)
{
	std::cout << "windows" << std::endl;

	cv::Mat_<cv::Vec2b> sizes  = adaptive_window_size(base, container.labels, config.entropy_threshold, config.min_windowsize, config.max_windowsize, container.regions);

	std::cout << "windows finished" << std::endl;

	showWindowSizes(sizes);

	long long start = cv::getCPUTickCount();
	cv::Mat costmap  = costmap_creators::slidingParametricJointWindowFlex<cost_func>(base, match, sizes, initial_disparity, config.deltaDisp, task.dispMin, task.dispMax, config.min_windowsize, config.max_windowsize);
	//cv::Mat costmap = scaleDisparity<cost_func,quantizer>(task, task.baseGray, task.matchGray, config, sizes, initial_disparity);
	start = cv::getCPUTickCount() - start;
	std::cout << "varwindow: " << start << std::endl;
	return costmap;
}

#endif // REFINEMENT_H
