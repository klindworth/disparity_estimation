/*
Copyright (c) 2014, Kai Klindworth
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

#include "manual_region_optimizer.h"

#include "disparity_region.h"
#include "genericfunctions.h"
#include "disparity_utils.h"

void manual_region_optimizer::optimize(std::vector<unsigned char>& damping_history, std::vector<std::vector<float>>& optimization_vectors_base, std::vector<std::vector<float>>& optimization_vectors_match, RegionContainer& base, RegionContainer& match, const disparity_hypothesis_weight_vector& base_eval, std::function<float(const DisparityRegion&, const RegionContainer&, const RegionContainer&, int)> prop_eval, int delta)
{
	//std::cout << "base" << std::endl;
	refreshOptimizationBaseValues(optimization_vectors_base, base, match, base_eval, delta);
	refreshOptimizationBaseValues(optimization_vectors_match, match, base, base_eval, delta);
	//std::cout << "optimize" << std::endl;

	std::random_device random_dev;
	std::mt19937 random_gen(random_dev()); //FIXME each threads needs a own copy
	std::uniform_int_distribution<> random_dist(0, 1);

	const int dispMin = base.task.dispMin;
	const int crange = base.task.range_size();
	cv::Mat_<float> temp_results(crange, 1, 100.0f);

	const std::size_t regions_count = base.regions.size();
	//pragma omp parallel for default(none) shared(base, match, prop_eval, delta) private(random_dist, random_gen, temp_results)
	for(std::size_t j = 0; j < regions_count; ++j)
	{
		DisparityRegion& baseRegion = base.regions[j];
		temp_results = cv::Mat_<float>(crange, 1, 5500.0f); //TODO: try another mt safe with less memory allocations...
		auto range = getSubrange(baseRegion.base_disparity, delta, base.task);

		for(short d = range.first; d <= range.second; ++d)
		{
			if(!baseRegion.other_regions[d-dispMin].empty())
				temp_results(d-dispMin) = prop_eval(baseRegion, base, match, d);
		}

		short ndisparity = min_idx(temp_results, baseRegion.disparity - dispMin) + dispMin;

		//damping
		if(ndisparity != baseRegion.disparity)
			damping_history[j] += 1;
		if(damping_history[j] < 2)
		{
			baseRegion.disparity = ndisparity;
			/*EstimationStep step;
			step.disparity = baseRegion.disparity;
			baseRegion.results.push_back(step);*/
		}
		else {
			if(random_dist(random_gen) == 1)
			{
				baseRegion.disparity = ndisparity;
				/*EstimationStep step;
				step.disparity = baseRegion.disparity;
				baseRegion.results.push_back(step);*/
			}
			else
				damping_history[j] = 0;
		}
	}
}

void manual_region_optimizer::training()
{
	 std::cout << "training" << std::endl;
}

void manual_region_optimizer::reset(const RegionContainer &left, const RegionContainer &right)
{
	damping_history_left.resize(left.regions.size());
	std::fill(damping_history_left.begin(), damping_history_left.end(), 0);

	damping_history_right.resize(right.regions.size());
	std::fill(damping_history_right.begin(), damping_history_right.end(), 0);

	optimization_vectors_left.resize(left.regions.size());
	optimization_vectors_right.resize(right.regions.size());
}

void manual_region_optimizer::run(RegionContainer &left, RegionContainer &right, const optimizer_settings &config, int refinement)
{
	reset(left, right);

	for(int i = 0; i < config.rounds; ++i)
	{
		std::cout << "optimization round" << std::endl;
		optimize(damping_history_left, optimization_vectors_left, optimization_vectors_right, left, right, config.base_eval, config.prop_eval, refinement);
		refreshWarpedIdx(left);
		optimize(damping_history_right, optimization_vectors_right, optimization_vectors_left, right, left, config.base_eval, config.prop_eval, refinement);
		refreshWarpedIdx(right);
	}
	for(int i = 0; i < config.rounds; ++i)
	{
		std::cout << "optimization round2" << std::endl;
		optimize(damping_history_left, optimization_vectors_left, optimization_vectors_right, left, right, config.base_eval2, config.prop_eval2, refinement);
		refreshWarpedIdx(left);
		optimize(damping_history_right, optimization_vectors_right, optimization_vectors_left, right, left, config.base_eval2, config.prop_eval2, refinement);
		refreshWarpedIdx(right);
	}
	if(config.rounds == 0)
	{
		refreshOptimizationBaseValues(optimization_vectors_left, left, right, config.base_eval, refinement);
		refreshOptimizationBaseValues(optimization_vectors_right, right, left, config.base_eval, refinement);
	}
}
