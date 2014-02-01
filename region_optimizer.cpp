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

#include "region_optimizer.h"

#include "region.h"
#include "initial_disparity.h"
#include "debugmatstore.h"
#include "intervals.h"
#include "intervals_algorithms.h"
#include "disparity_utils.h"
#include "misc.h"

#include <iostream>
#include <iterator>
#include <functional>
#include <random>

disparity_hypothesis::disparity_hypothesis(cv::Mat& occmap, const SegRegion& baseRegion, short disparity, const std::vector<SegRegion>& left_regions, const std::vector<SegRegion>& right_regions, int pot_trunc, int dispMin)
{
	//occ
	std::vector<RegionInterval> filtered = getFilteredPixelIdx(occmap.cols, baseRegion.region.lineIntervals, disparity);
	cv::Mat occ_region = getRegionAsMat(occmap, filtered, disparity);
	int occ_sum = 0;
	for(int j = 0; j < occ_region.size[0]; ++j)
		occ_sum += occ_region.at<unsigned char>(j);

	if(occ_region.total() > 0)
		occ_avg = (float)occ_sum/occ_region.total();
	else
		occ_avg = 1; //TODO: find a better solution to this

	//neighbors
	neighbor_pot = getNeighborhoodsAverage(left_regions, baseRegion.neighbors, [&](const SegRegion& cregion){return (float) abs_pott((int)cregion.disparity, -disparity, pot_trunc);});

	neighbor_color_pot = getColorWeightedNeighborhoodsAverage(baseRegion.average_color, 15.0f, left_regions, baseRegion.neighbors, [&](const SegRegion& cregion){return (float) abs_pott((int)cregion.disparity, (int)disparity, pot_trunc);}).first;

	//lr
	lr_pot = getOtherRegionsAverage(right_regions, baseRegion.other_regions[disparity-dispMin], [&](const SegRegion& cregion){return (float)abs_pott((int)disparity, -cregion.disparity, pot_trunc);});

	//misc
	assert(disparity-dispMin >= 0);
	costs = baseRegion.disparity_costs(disparity-baseRegion.disparity_offset);
}

disparity_hypothesis disparity_hypothesis::delta(const disparity_hypothesis& base) const
{
	disparity_hypothesis hyp;
	hyp.costs   = costs - base.costs;
	hyp.occ_avg = occ_avg - base.occ_avg;
	hyp.neighbor_pot = neighbor_pot - base.neighbor_pot;

	return hyp;
}

disparity_hypothesis disparity_hypothesis::abs_delta(const disparity_hypothesis& base) const
{
	disparity_hypothesis hyp;
	hyp.costs   = std::abs(costs - base.costs);
	hyp.occ_avg = std::abs(occ_avg - base.occ_avg);
	hyp.neighbor_pot = std::abs(neighbor_pot - base.neighbor_pot);

	return hyp;
}

template<typename T>
std::size_t min_idx(const cv::Mat_<T>& src, std::size_t preferred = 0)
{
	std::size_t idx = preferred;
	T value = src(preferred);

	const T* ptr = src[0];
	std::size_t size = src.total();

	for(std::size_t i = 0; i < size; ++i)
	{
		if(*ptr < value)
		{
			value = *ptr;
			idx = i;
		}
		++ptr;
	}

	return idx;
}

void refreshOptimizationBaseValues(RegionContainer& base, RegionContainer& match, std::function<float(const disparity_hypothesis&)> stat_eval, int delta)
{
	cv::Mat disp = getDisparityBySegments(base);
	cv::Mat occmap = occlusionStat<short>(disp, 1.0);
	int pot_trunc = 10;

	const short dispMin = base.task.dispMin;
	const short dispRange = base.task.dispMax - base.task.dispMin + 1;

	for(SegRegion& baseRegion : base.regions)
	{
		auto range = getSubrange(baseRegion.base_disparity, delta, base.task);

		intervals::substractRegionValue<unsigned char>(occmap, baseRegion.warped_interval, 1);

		baseRegion.optimization_energy = cv::Mat_<float>(dispRange, 1, 100.0f);

		#pragma omp parallel for
		for(short d = range.first; d <= range.second; ++d)
		{
			std::vector<MutualRegion>& cregionvec = baseRegion.other_regions[d-dispMin];
			if(!cregionvec.empty())
			{
				disparity_hypothesis hyp(occmap, baseRegion, d, base.regions, match.regions, pot_trunc, dispMin);
				baseRegion.optimization_energy(d-dispMin) = stat_eval(hyp);
			}
		}

		//baseRegion.optimization_minimum = min_idx(baseRegion.optimization_energy);
		//float opt_val = baseRegion.optimization_energy(baseRegion.optimization_minimum);
		//std::transform(baseRegion.optimization_energy.begin(), baseRegion.optimization_energy.end(), baseRegion.optimization_energy.begin(), [=](const float& cval){return cval/opt_val;});
		//for(int i = 0; i < baseRegion.optimization_energy.rows; ++i)
			//baseRegion.optimization_energy.at<float>(i) /= opt_val;
		intervals::addRegionValue<unsigned char>(occmap, baseRegion.warped_interval, 1);
	}
}

void optimize(RegionContainer& base, RegionContainer& match, std::function<float(const disparity_hypothesis&)> base_eval, std::function<float(const SegRegion&, const RegionContainer&, const RegionContainer&, int)> prop_eval, int delta)
{
	refreshOptimizationBaseValues(base, match, base_eval, delta);
	refreshOptimizationBaseValues(match, base, base_eval, delta);

	std::random_device random_dev;
	std::mt19937 random_gen(random_dev());
	std::uniform_int_distribution<> random_dist(0, 1);

	const int dispMin = base.task.dispMin;
	const int crange = base.task.dispMax - base.task.dispMin + 1;
	cv::Mat_<float> temp_results(crange, 1, 100.0f);

	const std::size_t regions_count = base.regions.size();
	#pragma omp parallel for default(none) shared(base, match, prop_eval, delta) private(random_dist, random_gen, temp_results)
	for(std::size_t j = 0; j < regions_count; ++j)
	{
		SegRegion& baseRegion = base.regions[j];
		temp_results = cv::Mat_<float>(crange, 1, 5500.0f);
		auto range = getSubrange(baseRegion.base_disparity, delta, base.task);

		for(short d = range.first; d < range.second; ++d)
		{
			if(!baseRegion.other_regions[d-dispMin].empty())
				temp_results(d-dispMin) = prop_eval(baseRegion, base, match, d);
		}

		short ndisparity = min_idx(temp_results, baseRegion.disparity - dispMin) + dispMin;

		//damping
		if(ndisparity != baseRegion.disparity)
			++baseRegion.damping_history;
		if(baseRegion.damping_history < 2)
			baseRegion.disparity = ndisparity;
		else {
			if(random_dist(random_gen) == 1)
				baseRegion.disparity = ndisparity;
			else
				baseRegion.damping_history = 0;
		}
	}
}

void run_optimization(StereoTask& task, RegionContainer& left, RegionContainer& right, const optimizer_settings& config, int refinement)
{
	for(SegRegion& cregion : left.regions)
		cregion.damping_history = 0;
	for(SegRegion& cregion : right.regions)
		cregion.damping_history = 0;

	//optimize L/R
	if(refinement == 0)
	{
		for(int i = 0; i < config.rounds; ++i)
		{
			std::cout << "optimization round" << std::endl;
			optimize(left, right, config.base_eval, config.prop_eval, refinement);
			refreshWarpedIdx(left);
			optimize(right, left, config.base_eval, config.prop_eval, refinement);
			refreshWarpedIdx(right);
		}
		for(int i = 0; i < config.rounds; ++i)
		{
			std::cout << "optimization round2" << std::endl;
			optimize(left, right, config.base_eval2, config.prop_eval2, refinement);
			refreshWarpedIdx(left);
			optimize(right, left, config.base_eval2, config.prop_eval2, refinement);
			refreshWarpedIdx(right);
		}
		if(config.rounds == 0)
		{
			refreshOptimizationBaseValues(left, right, config.base_eval, refinement);
			refreshOptimizationBaseValues(right, left, config.base_eval, refinement);
		}
	}
	else
	{
		for(int i = 0; i < config.rounds; ++i)
		{
			std::cout << "optimization round-refine" << std::endl;
			optimize(left, right, config.base_eval, config.prop_eval_refine, refinement);
			refreshWarpedIdx(left);
			optimize(right, left, config.base_eval, config.prop_eval_refine, refinement);
			refreshWarpedIdx(right);
		}
	}
}

cv::FileStorage& operator<<(cv::FileStorage& stream, const config_term& config)
{
	stream << "{:";
	stream << "cost" << config.cost;
	stream << "color_disp" << config.color_disp;
	stream << "lr_pot" << config.lr_pot;
	stream << "occ" << config.occ;
	stream << "}";

	return stream;
}

const cv::FileNode& operator>>(const cv::FileNode& node, config_term& config)
{
	node["cost"] >> config.cost;
	node["color_disp"] >> config.color_disp;
	node["lr_pot"] >> config.lr_pot;
	node["occ"] >> config.occ;

	return node;
}

cv::FileStorage& operator<<(cv::FileStorage& stream, const optimizer_settings& config)
{
	stream << "optimization_rounds" << config.rounds;
	stream << "enable_damping" << config.enable_damping;

	stream << "base_config" << config.base;

	return stream;
}

const cv::FileNode& operator>>(const cv::FileNode& stream, optimizer_settings& config)
{
	stream["optimization_rounds"] >> config.rounds;
	stream["enable_damping"] >> config.enable_damping;

	//cv::FileNode node = stream["base_configs"];
	//for(cv::FileNodeIterator it = node.begin(); it != node.end(); ++it)
		//*it >> config.base;
	stream["base_config"] >> config.base;

	return stream;
}

