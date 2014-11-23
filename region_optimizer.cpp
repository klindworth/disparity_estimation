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
#include <segmentation/intervals.h>
#include <segmentation/intervals_algorithms.h>
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
			RegionInterval hyp_interval = region[i];
			hyp_interval.move(dx, src.cols);
			RegionInterval old_interval = old_region[i];

			if(hyp_interval.lower != old_interval.lower)
			{
				sum -= src(old_interval.y, old_interval.lower);
				--count;
			}
			if(hyp_interval.upper != old_interval.upper)
			{
				sum += src(old_interval.y, old_interval.upper);
				++count;
			}

			old_region[i] = hyp_interval;
		}
		result[dx - dx_min] = std::make_pair(count, sum);
	}
}

disparity_hypothesis_vector::disparity_hypothesis_vector(const std::vector<DisparityRegion>& base_regions, const std::vector<DisparityRegion>& match_regions) : base_disparities_cache(base_regions.size()), match_disparities_cache(match_regions.size()), color_cache(base_regions.size())
{
	for(std::size_t i = 0; i < match_regions.size(); ++i)
		match_disparities_cache[i] = match_regions[i].disparity;
	for(std::size_t i = 0; i < base_regions.size(); ++i)
		base_disparities_cache[i] = base_regions[i].disparity;

	for(std::size_t i = 0; i < base_regions.size(); ++i)
		color_cache[i] = base_regions[i].average_color;
}

void disparity_hypothesis_vector::operator()(const cv::Mat_<unsigned char>& occmap, const DisparityRegion& baseRegion, short pot_trunc, int dispMin, int dispStart, int dispEnd, std::vector<float>& result_vector)
{
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
	/*gather_neighbor_values(neighbor_disparities, left_regions, baseRegion.neighbors, [](const DisparityRegion& cregion) {
		return cregion.disparity;
	});*/

	gather_neighbor_values_idx(neighbor_disparities, baseRegion.neighbors, [&](std::size_t idx){
		assert(base_disparities_cache.size() > idx);
		return base_disparities_cache[idx];
	});

	/*gather_neighbor_values_idx(neighbor_disparities, left_regions, baseRegion.neighbors, [](std::size_t idx) {
		return left_regions.
	});*/


	float divider = 1.0f/neighbor_disparities.size();
	for(short i = 0; i < range; ++i)
	{
		short pot_sum = 0;
		short disp = i + dispStart;
		for(short cdisp : neighbor_disparities)
			pot_sum += abs_pott(cdisp, disp, pot_trunc);

		neighbor_pot_values[i] = pot_sum * divider;
	}

	//TODO: stays constant during iterations -> dont recalculate
	float weight_sum = gather_neighbor_color_weights_from_cache(neighbor_color_weights, baseRegion.average_color, 15.0f, color_cache, baseRegion.neighbors);
	weight_sum = 1.0f/weight_sum;

	//color neighbor pot
	for(short i = 0; i < range; ++i)
	{
		float pot_sum = 0;
		short disp = i + dispStart;
		for(std::size_t j = 0; j < neighbor_disparities.size(); ++j)
			pot_sum += abs_pott(neighbor_disparities[j], disp, pot_trunc) * neighbor_color_weights[j];

		neighbor_color_pot_values[i] = pot_sum * weight_sum;
	}

	//lr_pot
	assert(baseRegion.other_regions.size() >= range);
	for(short cdisp = dispStart; cdisp <= dispEnd; ++cdisp)
		lr_pot_values[cdisp - dispStart] = other_regions_average_by_index(baseRegion.other_regions[cdisp-dispMin],
				[&](std::size_t idx){
			return (float)abs_pott(cdisp, (short)-match_disparities_cache[idx], pot_trunc);
		});
		//lr_pot_values[cdisp - dispStart] = getOtherRegionsAverage(right_regions, baseRegion.other_regions[cdisp-dispMin], [&](const DisparityRegion& cregion){return (float)abs_pott(cdisp, (short)-cregion.disparity, pot_trunc);});

	for(int i = 0; i < range; ++i)
		cost_values[i] = baseRegion.disparity_costs((dispStart+i)-baseRegion.disparity_offset);

	//	float costs, occ_avg, neighbor_pot, lr_pot ,neighbor_color_pot;
	result_vector.resize(range*5);
	float *result_ptr = result_vector.data();
	for(int i = 0; i < range; ++i)
	{
		*result_ptr++ = cost_values[i];
		*result_ptr++ = occ_avg_values[i];
		*result_ptr++ = neighbor_pot_values[i];
		*result_ptr++ = lr_pot_values[i];
		*result_ptr++ = neighbor_color_pot_values[i];
	}
}

disparity_hypothesis::disparity_hypothesis(const std::vector<float>& optimization_vector, int dispIdx)
{
	const float *ptr = optimization_vector.data() + dispIdx * 5;
	costs = *ptr++;
	occ_avg = *ptr++;
	neighbor_pot = *ptr++;
	lr_pot = *ptr++;
	neighbor_color_pot = *ptr++;
}

float calculate_end_result(const float *raw_results, const disparity_hypothesis_weight_vector& wv)
{
	return raw_results[0] * wv.costs + raw_results[1] * wv.occ_avg + raw_results[2] * wv.neighbor_pot + raw_results[3] * wv.lr_pot + raw_results[4] * wv.neighbor_color_pot;
}

float calculate_end_result(int disp_idx, const float *raw_results, const disparity_hypothesis_weight_vector &wv)
{
	std::size_t idx_offset = disp_idx*5;
	const float *result_ptr = raw_results + idx_offset;

	return calculate_end_result(result_ptr, wv);
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

void refreshOptimizationBaseValues(std::vector<std::vector<float>>& optimization_vectors, RegionContainer& base, const RegionContainer& match, const disparity_hypothesis_weight_vector& stat_eval, int delta)
{
	cv::Mat disp = getDisparityBySegments(base);
	cv::Mat occmap = occlusionStat<short>(disp, 1.0);
	int pot_trunc = 10;

	const short dispMin = base.task.dispMin;
	const short dispRange = base.task.dispMax - base.task.dispMin + 1;

	std::vector<disparity_hypothesis_vector> hyp_vec(omp_get_max_threads(), disparity_hypothesis_vector(base.regions, match.regions));
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
		DisparityRegion& baseRegion = base.regions[i];
		int thread_idx = omp_get_thread_num();
		auto range = getSubrange(baseRegion.base_disparity, delta, base.task);

		intervals::substractRegionValue<unsigned char>(occmaps[thread_idx], baseRegion.warped_interval, 1);

		baseRegion.optimization_energy = cv::Mat_<float>(dispRange, 1, 100.0f);

		hyp_vec[thread_idx](occmaps[thread_idx], baseRegion, pot_trunc, dispMin, range.first, range.second, optimization_vectors[i]);
		for(short d = range.first; d <= range.second; ++d)
		{
			std::vector<MutualRegion>& cregionvec = baseRegion.other_regions[d-dispMin];
			if(!cregionvec.empty())
				baseRegion.optimization_energy(d-dispMin) = calculate_end_result((d - range.first), optimization_vectors[i].data(), stat_eval);
		}

		intervals::addRegionValue<unsigned char>(occmaps[thread_idx], baseRegion.warped_interval, 1);
	}
}

const cv::FileNode& operator>>(const cv::FileNode& node, disparity_hypothesis_weight_vector& config)
{
	node["cost"] >> config.costs;
	node["neighbor_pot"] >> config.neighbor_pot;
	node["neighbor_color_pot"] >> config.neighbor_color_pot;
	node["lr_pot"] >> config.lr_pot;
	node["occ_avg"] >> config.occ_avg;

	return node;
}

cv::FileStorage& operator<<(cv::FileStorage& stream, const disparity_hypothesis_weight_vector& config)
{
	stream << "{:";
	stream << "cost" << config.costs;
	stream << "neighbor_color_pot" << config.neighbor_color_pot;
	stream << "neighbor_pot" << config.neighbor_pot;
	stream << "lr_pot" << config.lr_pot;
	stream << "occ_avg" << config.occ_avg;
	stream << "}";

	return stream;
}

cv::FileStorage& operator<<(cv::FileStorage& stream, const optimizer_settings& config)
{
	stream << "optimization_rounds" << config.rounds;
	stream << "enable_damping" << config.enable_damping;

	stream << "base_eval" << config.base_eval;
	stream << "base_eval2" << config.base_eval2;

	return stream;
}

const cv::FileNode& operator>>(const cv::FileNode& stream, optimizer_settings& config)
{
	stream["optimization_rounds"] >> config.rounds;
	stream["enable_damping"] >> config.enable_damping;

	//cv::FileNode node = stream["base_configs"];
	//for(cv::FileNodeIterator it = node.begin(); it != node.end(); ++it)
		//*it >> config.base;
	stream["base_eval"] >> config.base_eval;
	stream["base_eval2"] >> config.base_eval2;

	return stream;
}

