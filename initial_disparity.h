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

#ifndef INITIAL_DISPARITY_H
#define INITIAL_DISPARITY_H

#include <vector>
#include <memory>
#include "configrun.h"
#include "refinement.h"

namespace cv {
	class Mat;
}

class stereo_task;
class single_stereo_task;
class disparity_region;
class region_container;
class segmentation_algorithm;
class region_interval;
class region_descriptor;
class region_optimizer;

void generate_region_information(region_container& left, region_container& right);

std::pair<cv::Mat, cv::Mat> segment_based_disparity_it(stereo_task& task, const initial_disparity_config& config, const RefinementConfig& refconfig, int subsampling, region_optimizer& optimizer);

class initial_disparity_config
{
public:
	std::string name;
	std::string metric_type;
	unsigned int dilate;
	int dilate_step;
	bool dilate_grow;
	bool enable_refinement;
	int region_refinement_delta;
	int region_refinement_rounds;

	segmentation_settings segmentation;
	optimizer_settings optimizer;
	bool verbose;
};

class initial_disparity_algo : public disparity_estimator_algo
{
public:
	initial_disparity_algo(initial_disparity_config& config, RefinementConfig& refconfig, std::shared_ptr<region_optimizer>& optimizer);
	virtual std::pair<cv::Mat, cv::Mat> operator()(stereo_task& task);
	virtual void writeConfig(cv::FileStorage& fs);
	void train(std::vector<stereo_task>& tasks);

private:
	std::shared_ptr<region_optimizer> m_optimizer;
	initial_disparity_config m_config;
	RefinementConfig m_refconfig;
};

#endif // INITIAL_DISPARITY_H
