#ifndef CONFIGRUN_H
#define CONFIGRUN_H

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

#include "disparity_toolkit/disparity_map.h"
#include <opencv2/core/core.hpp>
#include <functional>

#include "optimizer/region_optimizer.h"
#include <segmentation/segmentation.h>

class stereo_task;
class disparity_region;
class region_container;
class TaskTestSet;
class classic_search_config;
class refinement_config;
class task_collection;

class disparity_estimator_algo
{
public:
	disparity_estimator_algo() {}
	virtual ~disparity_estimator_algo() {}
	virtual std::pair<disparity_map, disparity_map> operator()(stereo_task& task) = 0;
	virtual void writeConfig(cv::FileStorage& fs) = 0;
};

std::pair<disparity_map, disparity_map> single_logged_run(stereo_task& task, disparity_estimator_algo &disparity_estimator, cv::FileStorage& fs, const std::string& filename);
void logged_run(stereo_task& task, initial_disparity_config& config, refinement_config& refconfig);
void logged_run(task_collection& testset, initial_disparity_config& config, refinement_config& refconfig);
void classicLoggedRun(task_collection& taskset, classic_search_config& config);

#endif // CONFIGRUN_H
