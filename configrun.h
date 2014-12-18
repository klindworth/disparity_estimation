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

#include <opencv2/core/core.hpp>
#include <functional>

#include "region_optimizer.h"
#include <segmentation/segmentation.h>

class StereoTask;
class disparity_region;
class RegionContainer;
class TaskTestSet;
class ClassicSearchConfig;
class RefinementConfig;
class TaskCollection;

class disparity_estimator_algo
{
public:
	disparity_estimator_algo() {}
	virtual ~disparity_estimator_algo() {}
	virtual std::pair<cv::Mat, cv::Mat> operator()(StereoTask& task) = 0;
	virtual void writeConfig(cv::FileStorage& fs) = 0;
};

cv::FileStorage& operator<<(cv::FileStorage& stream, const InitialDisparityConfig& config);
cv::FileStorage& operator>>(cv::FileStorage& stream, InitialDisparityConfig& config);

std::pair<cv::Mat, cv::Mat> singleLoggedRun(StereoTask& task, disparity_estimator_algo &disparity_estimator, cv::FileStorage& fs, const std::string& filename);
void loggedRun(StereoTask& task, InitialDisparityConfig& config, RefinementConfig& refconfig);
void loggedRun(TaskCollection& testset, InitialDisparityConfig& config, RefinementConfig& refconfig);
void classicLoggedRun(TaskCollection& taskset, ClassicSearchConfig& config);

#endif // CONFIGRUN_H
