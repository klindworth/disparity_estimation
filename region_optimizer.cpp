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

#include <omp.h>

template<typename sum_type, typename T>
void segment_boxfilter(std::vector<std::pair<int, sum_type> >& result, const cv::Mat_<T>& src, const std::vector<RegionInterval>& region, int dx_min, int dx_max)
{
	assert(dx_max >= dx_min);
	assert((int)result.size() == dx_max - dx_min + 1);

	std::vector<RegionInterval> old_region = region;
	move_x_region(old_region.begin(), old_region.end(), dx_min, src.cols);

	sum_type sum = 0;
	int count = 0;
	intervals::foreach_region_point(old_region.begin(), old_region.end(), [&](cv::Point pt) {
		sum += src(pt);
		++count;
	});
	result[0] = std::make_pair(count, sum);

	for(int dx = dx_min+1; dx < dx_max; ++dx)
	{
		for(int i = 0; i < region.size(); ++i)
		{
			//std::cout << "check" << std::endl;
			RegionInterval hyp_interval = region[i];
			hyp_interval.move(dx, src.cols);
			RegionInterval old_interval = old_region[i];


			//std::cout << "dx: " << dx << ", " << hyp_interval << ", " << old_interval << std::endl;

			if(hyp_interval.upper != old_interval.upper)
			{
				sum += src(hyp_interval.y, hyp_interval.upper - 1);
				++count;
			}
			if(hyp_interval.lower != old_interval.lower)
			{
				//if(std::abs(hyp_interval.lower - old_interval.lower) != 1)
					//std::cout << hyp_interval.lower << " vs " << old_interval.lower << std::endl;
				//else
					//std::cout << "ok" << std::endl;

				sum -= src(old_interval.y, old_interval.lower);
				--count;
			}

			old_region[i] = hyp_interval;
		}
		result[dx - dx_min] = std::make_pair(count, sum);
	}
}

/*disparity_hypothesis_vector::disparity_hypothesis_vector(int dispRange) : dispRange(dispRange), occ_temp(dispRange), occ_avg_values(dispRange), neighbor_pot_values(dispRange), neighbor_color_pot_values(dispRange), lr_pot_values(dispRange), cost_values(dispRange)
{

}*/

void disparity_hypothesis_vector::operator()(const cv::Mat_<unsigned char>& occmap, const DisparityRegion& baseRegion, const std::vector<DisparityRegion>& left_regions, const std::vector<DisparityRegion>& right_regions, short pot_trunc, int dispMin, int dispStart, int dispEnd)
{
	//resizing



	this->dispStart = dispStart;
	const int range = dispEnd - dispStart + 1;

	occ_temp.resize(range);
	occ_avg_values.resize(range);
	neighbor_pot_values.resize(range);
	neighbor_color_pot_values.resize(range);
	lr_pot_values.resize(range);
	cost_values.resize(range);

	//assert(dispRange == range);
	//occ_avg
	segment_boxfilter(occ_temp, occmap, baseRegion.lineIntervals, dispStart, dispEnd);

	for(int i = 0; i < range; ++i)
		occ_avg_values[i] = (occ_temp[i].first != 0) ? (float)occ_temp[i].second / occ_temp[i].first : 1;

	//neighbor pot
	gather_neighbor_values(neighbor_disparities, left_regions, baseRegion.neighbors, [](const DisparityRegion& cregion) {
		return cregion.disparity;
	});


	float divider = 1.0f/neighbor_disparities.size();
	for(short i = 0; i < range; ++i)
	{
		short pot_sum = 0;
		short disp = i + dispStart;
		for(short cdisp : neighbor_disparities)
			pot_sum += abs_pott(cdisp, disp, pot_trunc);

		neighbor_pot_values[i] = pot_sum * divider;
	}

	//color neighbor pot
	float weight_sum = gather_neighbor_color_weights(neighbor_color_weights, baseRegion.average_color, 15.0f, left_regions, baseRegion.neighbors);
	weight_sum = 1.0f/weight_sum;

	for(short i = 0; i < range; ++i)
	{
		float pot_sum = 0;
		short disp = i + dispStart;
		for(std::size_t j = 0; j < neighbor_disparities.size(); ++j)
			pot_sum += abs_pott(neighbor_disparities[j], disp, pot_trunc) * neighbor_color_weights[j];

		neighbor_color_pot_values[i] = pot_sum * weight_sum;
	}

	//lr_pot
	for(short cdisp = dispStart; cdisp <= dispEnd; ++cdisp)
		lr_pot_values[cdisp - dispStart] = getOtherRegionsAverage(right_regions, baseRegion.other_regions[cdisp-dispMin], [&](const DisparityRegion& cregion){return (float)abs_pott(cdisp, (short)-cregion.disparity, pot_trunc);});

	for(int i = 0; i < range; ++i)
		cost_values[i] = baseRegion.disparity_costs((dispStart+i)-baseRegion.disparity_offset);
}

disparity_hypothesis disparity_hypothesis_vector::operator()(int disp) const
{
	std::size_t idx = disp - dispStart;
	assert(idx < dispRange);

	disparity_hypothesis hyp;
	hyp.occ_avg = occ_avg_values[idx];
	hyp.neighbor_color_pot = neighbor_color_pot_values[idx];
	hyp.neighbor_pot = neighbor_pot_values[idx];
	hyp.lr_pot = lr_pot_values[idx];
	hyp.costs = cost_values[idx];

	return hyp;
}



float calculate_occ_avg(const cv::Mat_<unsigned char>& occmap, const DisparityRegion& baseRegion, short disparity)
{
	//occ
	int occ_sum = 0;
	int count = 0;
	//foreach_warped_region_point(baseRegion.lineIntervals, occmap.cols, disparity, [&](cv::Point pt){
	foreach_warped_region_point(baseRegion.lineIntervals.begin(), baseRegion.lineIntervals.end(), occmap.cols, disparity, [&](cv::Point pt){
		occ_sum += occmap(pt);
		++count;
	});
	return (count > 0 ? (float)occ_sum/count : 1.0f);
}

disparity_hypothesis::disparity_hypothesis(const cv::Mat_<unsigned char>& occmap, const DisparityRegion& baseRegion, short disparity, const std::vector<DisparityRegion>& left_regions, const std::vector<DisparityRegion>& right_regions, short pot_trunc, int dispMin)
{
	//occ
	/*int occ_sum = 0;
	int count = 0;
	foreach_warped_region_point(baseRegion.lineIntervals, occmap.cols, disparity, [&](cv::Point pt){
		occ_sum += occmap.at<unsigned char>(pt);
		++count;
	});
	occ_avg = count > 0 ? (float)occ_sum/count : 1;*/
	occ_avg = calculate_occ_avg(occmap, baseRegion, disparity);

	//neighbors
	neighbor_pot = getNeighborhoodsAverage(left_regions, baseRegion.neighbors, [&](const DisparityRegion& cregion){return (float) abs_pott(cregion.disparity, disparity, pot_trunc);});

	neighbor_color_pot = getColorWeightedNeighborhoodsAverage(baseRegion.average_color, 15.0f, left_regions, baseRegion.neighbors, [&](const DisparityRegion& cregion){return (float) abs_pott((int)cregion.disparity, (int)disparity, (int)pot_trunc);}).first;

	//lr
	lr_pot = getOtherRegionsAverage(right_regions, baseRegion.other_regions[disparity-dispMin], [&](const DisparityRegion& cregion){return (float)abs_pott((int)disparity, -cregion.disparity, (int)pot_trunc);});

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

	std::vector<cv::Mat_<unsigned char>> occmaps(omp_get_max_threads());
	for(std::size_t i = 0; i < occmaps.size(); ++i)
		occmaps[i] = occmap.clone();

	std::size_t regions_count = base.regions.size();

	disparity_hypothesis_vector hyp_vec;
	#pragma omp parallel for private(hyp_vec)
	//for(DisparityRegion& baseRegion : base.regions)
	for(std::size_t i = 0; i < regions_count; ++i)
	{
		DisparityRegion& baseRegion = base.regions[i];
		int thread_idx = omp_get_thread_num();
		//int thread_idx = 0;
		auto range = getSubrange(baseRegion.base_disparity, delta, base.task);

		intervals::substractRegionValue<unsigned char>(occmaps[thread_idx], baseRegion.warped_interval, 1);

		baseRegion.optimization_energy = cv::Mat_<float>(dispRange, 1, 100.0f);

		//disparity_hypothesis_vector hyp_vec(range.second - range.first + 1);
		hyp_vec(occmaps[thread_idx], baseRegion, base.regions, match.regions, pot_trunc, dispMin, range.first, range.second);
		for(short d = range.first; d <= range.second; ++d)
		{
			std::vector<MutualRegion>& cregionvec = baseRegion.other_regions[d-dispMin];
			if(!cregionvec.empty())
			{
				//std::cout << d << std::endl;
				/*disparity_hypothesis hyp(occmaps[thread_idx], baseRegion, d, base.regions, match.regions, pot_trunc, dispMin);

				disparity_hypothesis hyp_cmp = hyp_vec(d);

				float eps = 0.000001f;
				if(std::abs(hyp.costs - hyp_cmp.costs) > eps)
					std::cout << "cost fail" << std::endl;
				if(std::abs(hyp.lr_pot - hyp_cmp.lr_pot) > eps)
					std::cout << "lr_pot fail" << std::endl;
				if(std::abs(hyp.neighbor_pot - hyp_cmp.neighbor_pot) > eps)
				{
					std::cout << "neighbor_pot fail" << std::endl;
					std::cout << hyp.neighbor_pot << " vs " << hyp_cmp.neighbor_pot << std::endl;
				}
				if(std::abs(hyp.occ_avg - hyp_cmp.occ_avg) > eps)
				{
					std::cout << "occ fail" << std::endl;
					std::cout << hyp.occ_avg << " vs " << hyp_cmp.occ_avg << std::endl;
				}
				if(std::abs(hyp.neighbor_color_pot - hyp_cmp.neighbor_color_pot) > eps)
				{
					std::cout << "color fail" << std::endl;
					std::cout << hyp.neighbor_color_pot << " vs " << hyp_cmp.neighbor_color_pot << std::endl;
				}*/

				//baseRegion.optimization_energy(d-dispMin) = stat_eval(hyp);
				baseRegion.optimization_energy(d-dispMin) = stat_eval(hyp_vec(d));
			}
		}

		//baseRegion.optimization_minimum = min_idx(baseRegion.optimization_energy);
		//float opt_val = baseRegion.optimization_energy(baseRegion.optimization_minimum);
		//std::transform(baseRegion.optimization_energy.begin(), baseRegion.optimization_energy.end(), baseRegion.optimization_energy.begin(), [=](const float& cval){return cval/opt_val;});
		//for(int i = 0; i < baseRegion.optimization_energy.rows; ++i)
			//baseRegion.optimization_energy.at<float>(i) /= opt_val;
		intervals::addRegionValue<unsigned char>(occmaps[thread_idx], baseRegion.warped_interval, 1);
	}
}

void optimize(RegionContainer& base, RegionContainer& match, std::function<float(const disparity_hypothesis&)> base_eval, std::function<float(const DisparityRegion&, const RegionContainer&, const RegionContainer&, int)> prop_eval, int delta)
{
	std::cout << "base" << std::endl;
	refreshOptimizationBaseValues(base, match, base_eval, delta);
	refreshOptimizationBaseValues(match, base, base_eval, delta);
	std::cout << "optimize" << std::endl;

	std::random_device random_dev;
	std::mt19937 random_gen(random_dev()); //FIXME each threads needs a own copy
	std::uniform_int_distribution<> random_dist(0, 1);

	const int dispMin = base.task.dispMin;
	const int crange = base.task.dispMax - base.task.dispMin + 1;
	cv::Mat_<float> temp_results(crange, 1, 100.0f);

	const std::size_t regions_count = base.regions.size();
	#pragma omp parallel for default(none) shared(base, match, prop_eval, delta) private(random_dist, random_gen, temp_results)
	for(std::size_t j = 0; j < regions_count; ++j)
	{
		DisparityRegion& baseRegion = base.regions[j];
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
	for(DisparityRegion& cregion : left.regions)
		cregion.damping_history = 0;
	for(DisparityRegion& cregion : right.regions)
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

