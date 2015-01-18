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

#include <omp.h>

#include "disparity_region.h"
#include "genericfunctions.h"
#include "disparity_utils.h"

void manual_region_optimizer::optimize(std::vector<unsigned char>& damping_history, region_container& base, region_container& match, const disparity_hypothesis_weight_vector& base_eval, std::function<float(const disparity_region&, const region_container&, const region_container&, int)> prop_eval, int delta)
{
	//std::cout << "base" << std::endl;
	refreshOptimizationBaseValues(base, match, base_eval, delta);
	refreshOptimizationBaseValues(match, base, base_eval, delta);
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
		disparity_region& baseRegion = base.regions[j];
		temp_results = cv::Mat_<float>(crange, 1, 5500.0f); //TODO: try another mt safe with less memory allocations...
		disparity_range drange = task_subrange(base.task, baseRegion.base_disparity, delta);

		for(short d = drange.start(); d <= drange.end(); ++d)
		{
			if(!baseRegion.corresponding_regions[d-dispMin].empty())
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

void manual_region_optimizer::reset(const region_container &left, const region_container &right)
{
	damping_history_left.resize(left.regions.size());
	std::fill(damping_history_left.begin(), damping_history_left.end(), 0);

	damping_history_right.resize(right.regions.size());
	std::fill(damping_history_right.begin(), damping_history_right.end(), 0);
}

void manual_region_optimizer::run(region_container &left, region_container &right, const optimizer_settings &config, int refinement)
{
	reset(left, right);

	for(int i = 0; i < config.rounds; ++i)
	{
		std::cout << "optimization round" << std::endl;
		optimize(damping_history_left, left, right, config.base_eval, config.prop_eval, refinement);
		refresh_warped_regions(left);
		optimize(damping_history_right, right, left, config.base_eval, config.prop_eval, refinement);
		refresh_warped_regions(right);
	}
	for(int i = 0; i < config.rounds; ++i)
	{
		std::cout << "optimization round2" << std::endl;
		optimize(damping_history_left, left, right, config.base_eval2, config.prop_eval2, refinement);
		refresh_warped_regions(left);
		optimize(damping_history_right, right, left, config.base_eval2, config.prop_eval2, refinement);
		refresh_warped_regions(right);
	}
	if(config.rounds == 0)
	{
		refreshOptimizationBaseValues(left, right, config.base_eval, refinement);
		refreshOptimizationBaseValues(right, left, config.base_eval, refinement);
	}
}

void manual_optimizer_feature_calculator::update(cv::Mat_<unsigned char>& occmap, short pot_trunc, const disparity_region& baseRegion, const disparity_range& drange)
{
	const int range = drange.size();

	cost_values.resize(range);

	update_occ_avg(occmap, baseRegion, pot_trunc, drange);
	update_average_neighbor_values(baseRegion, pot_trunc, drange);
	update_lr_pot(baseRegion, pot_trunc, drange);
	update_warp_costs(baseRegion, drange);

	for(int i = 0; i < range; ++i)
		cost_values[i] = baseRegion.disparity_costs((drange.start()+i)-baseRegion.disparity_offset);
}

void refreshOptimizationBaseValues(region_container& base, const region_container& match, const disparity_hypothesis_weight_vector& stat_eval, int delta)
{
	cv::Mat disp = disparity_by_segments(base);
	cv::Mat occmap = disparity::occlusion_stat<short>(disp, 1.0);
	int pot_trunc = 15;

	const short dispMin = base.task.dispMin;
	const short dispRange = base.task.dispMax - base.task.dispMin + 1;

	std::vector<manual_optimizer_feature_calculator> hyp_vec(omp_get_max_threads(), manual_optimizer_feature_calculator(base, match));
	std::vector<cv::Mat_<unsigned char>> occmaps(omp_get_max_threads());
	for(std::size_t i = 0; i < occmaps.size(); ++i)
	{
		occmaps[i] = occmap.clone();
		//hyp_vec.emplace_back(base.regions, match.regions);
	}

	std::size_t regions_count = base.regions.size();


	#pragma omp parallel for
	for(std::size_t i = 0; i < regions_count; ++i)
	{
		disparity_region& baseRegion = base.regions[i];
		int thread_idx = omp_get_thread_num();

		baseRegion.optimization_energy = cv::Mat_<float>(dispRange, 1, 100.0f);

		disparity_range drange = task_subrange(base.task, baseRegion.base_disparity, delta);

		hyp_vec[thread_idx].update(occmaps[thread_idx], pot_trunc, baseRegion, drange);
		for(short d = drange.start(); d <= drange.end(); ++d)
		{
			std::vector<corresponding_region>& cregionvec = baseRegion.corresponding_regions[d-dispMin];
			if(!cregionvec.empty())
				baseRegion.optimization_energy(d-dispMin) = stat_eval.evaluate_hypthesis(hyp_vec[thread_idx].get_disparity_hypothesis((d - drange.start())));
		}
	}
}

