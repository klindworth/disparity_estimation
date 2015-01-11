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

#include "disparity_region.h"
#include "debugmatstore.h"
#include <segmentation/intervals.h>
#include <segmentation/intervals_algorithms.h>
#include "disparity_utils.h"
#include "disparity_region_algorithms.h"

#include <iostream>
#include <iterator>
#include <random>
#include <omp.h>

disparity_hypothesis_vector::disparity_hypothesis_vector(const region_container& base, const region_container& match) : base_avg_cache(base.regions.size()), base_disparities_cache(base.regions.size()), match_disparities_cache(match.regions.size()), color_cache(base.regions.size())
{
	for(std::size_t i = 0; i < match.regions.size(); ++i)
		match_disparities_cache[i] = match.regions[i].disparity;
	for(std::size_t i = 0; i < base.regions.size(); ++i)
	{
		base_disparities_cache[i] = base.regions[i].disparity;
		color_cache[i] = base.regions[i].average_color;
		base_avg_cache[i] = base.regions[i].avg_point;
	}

	int dispMin = base.task.dispMin;
	cv::Mat_<float> min_costs = region_descriptors::set_regionwise<float>(base, [=](const disparity_region& region) {
		return region.disparity_costs(region.disparity-dispMin);
	});

	warp_costs = cv::Mat_<float>(base.image_size, 1.0f);
	cv::Mat_<short> disp = disparity_by_segments(base);
	disparity::foreach_warped_pixel(disp, 1.0f, [&](cv::Point pos, cv::Point warped_pos, short) {
		warp_costs(warped_pos) = std::min(min_costs(pos), warp_costs(warped_pos));
	});
}

void disparity_hypothesis_vector::update_average_neighbor_values(const disparity_region& baseRegion, short pot_trunc, const disparity_range& drange)
{
	const int range = drange.size();
	neighbor_pot_values.resize(range);
	neighbor_color_pot_values.resize(range);

	region_descriptors::gather_neighbor_values_idx(neighbor_disparities, baseRegion.neighbors, [&](std::size_t idx){
		assert(base_disparities_cache.size() > idx);
		return base_disparities_cache[idx];
	});

	float divider = 1.0f/neighbor_disparities.size();
	for(short i = 0; i < range; ++i)
	{
		short pot_sum = 0;
		short disp = i + drange.start();
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
		short disp = i + drange.start();
		for(std::size_t j = 0; j < neighbor_disparities.size(); ++j)
			pot_sum += abs_pott(neighbor_disparities[j], disp, pot_trunc) * neighbor_color_weights[j];

		neighbor_color_pot_values[i] = pot_sum * weight_sum;
	}
}

void disparity_hypothesis_vector::update_lr_pot(const disparity_region& baseRegion, short pot_trunc, const disparity_range& drange)
{
	const int range = drange.size();
	const int dispStart = drange.start();
	const int dispEnd = drange.end();
	const int dispMin = drange.offset();
	lr_pot_values.resize(range);
	//lr_pot
	assert((int)baseRegion.corresponding_regions.size() >= range);
	for(short cdisp = dispStart; cdisp <= dispEnd; ++cdisp)
	{
		lr_pot_values[cdisp - dispStart] = corresponding_regions_average_by_index(baseRegion.corresponding_regions[cdisp-dispMin], [&](std::size_t idx){
			return (float)abs_pott(cdisp, (short)-match_disparities_cache[idx], pot_trunc);
		});
	}
}

float create_min_version(std::vector<float>::iterator start, std::vector<float>::iterator end, std::vector<float>::iterator ins)
{
	float min_value = *(std::min_element(start, end));

	std::transform(start, end, ins, [min_value](float val){
		return val - min_value;
	});

	return min_value;
}

void disparity_hypothesis_vector::update_occ_avg(const cv::Mat_<unsigned char>& occmap, const disparity_region& baseRegion, short /*pot_trunc*/, const disparity_range& drange)
{
	const int range = drange.size();
	occ_temp.resize(range);
	occ_avg_values.resize(range);

	region_descriptors::segment_boxfilter(occ_temp, occmap, baseRegion.lineIntervals, drange.start(), drange.end());

	for(int i = 0; i < range; ++i)
		occ_avg_values[i] = (occ_temp[i].first != 0) ? (float)occ_temp[i].second / occ_temp[i].first : -1;
}

void disparity_hypothesis_vector::operator()(const cv::Mat_<unsigned char>& occmap, const disparity_region& baseRegion, short pot_trunc, const disparity_range& drange, std::vector<float>& result_vector)
{
	const int range = drange.size();

	cost_values.resize(range);
	rel_cost_values.resize(range);
	cost_temp.resize(range);
	disp_costs.resize(range);

	//cost
	region_descriptors::segment_boxfilter(cost_temp, warp_costs, baseRegion.lineIntervals, drange.start(), drange.end());
	for(int i = 0; i < range; ++i)
		disp_costs[i] = (cost_temp[i].first != 0) ? baseRegion.disparity_costs(i)/(float)cost_temp[i].second * cost_temp[i].first : 0;

	update_occ_avg(occmap, baseRegion, pot_trunc, drange);
	update_average_neighbor_values(baseRegion, pot_trunc, drange);
	update_lr_pot(baseRegion, pot_trunc, drange);

	for(int i = 0; i < range; ++i)
		cost_values[i] = baseRegion.disparity_costs((drange.start()+i)-baseRegion.disparity_offset);

	create_min_version(cost_values.begin(), cost_values.end(), rel_cost_values.begin());

	update_result_vector(result_vector, baseRegion, drange);
}

void disparity_hypothesis_vector::update_result_vector(std::vector<float>& result_vector, const disparity_region& baseRegion, const disparity_range& drange)
{
	const int range = drange.size();
	const int dispMin = drange.offset();

	int cidx_left = -1;
	int cidx_right = -1;
	int cidx_top = -1;
	int cidx_bottom = -1;
	//extract neighbors
	{
		assert(baseRegion.neighbors.size() > 0);

		int y_diff_left = std::numeric_limits<int>::max();
		int y_diff_right = std::numeric_limits<int>::max();
		int x_diff_top = std::numeric_limits<int>::max();
		int x_diff_bottom = std::numeric_limits<int>::max();

		for(const std::pair<std::size_t, std::size_t>& cneigh : baseRegion.neighbors)
		{
			int cy_diff = std::abs(baseRegion.avg_point.y - base_avg_cache[cneigh.first].y);
			if( (baseRegion.avg_point.x > base_avg_cache[cneigh.first].x) && (cy_diff < y_diff_left) )
			{
				y_diff_left = cy_diff;
				cidx_left = cneigh.first;
			}
			else if( (baseRegion.avg_point.x < base_avg_cache[cneigh.first].x) && (cy_diff < y_diff_right) )
			{
				y_diff_right = cy_diff;
				cidx_right = cneigh.first;
			}

			int cx_diff = std::abs(baseRegion.avg_point.x - base_avg_cache[cneigh.first].x);
			if( (baseRegion.avg_point.y < base_avg_cache[cneigh.first].y) && (cx_diff < x_diff_top) )
			{
				x_diff_top = cx_diff;
				cidx_top = cneigh.first;
			}
			else if( (baseRegion.avg_point.y > base_avg_cache[cneigh.first].y) && (cx_diff < x_diff_bottom) )
			{
				x_diff_bottom = cy_diff;
				cidx_bottom = cneigh.first;
			}
		}
	}

	short left_neighbor_disp  = cidx_left >= 0 ? base_disparities_cache[cidx_left] : baseRegion.disparity;
	short right_neighbor_disp = cidx_right >= 0 ? base_disparities_cache[cidx_right] : baseRegion.disparity;
	short top_neighbor_disp  = cidx_top >= 0 ? base_disparities_cache[cidx_top] : baseRegion.disparity;
	short bottom_neighbor_disp = cidx_bottom >= 0 ? base_disparities_cache[cidx_bottom] : baseRegion.disparity;
	float left_color_dev = cidx_left >= 0 ? cv::norm(color_cache[cidx_left] - baseRegion.average_color) : 0;
	float right_color_dev = cidx_right >= 0 ? cv::norm(color_cache[cidx_right] - baseRegion.average_color) : 0;
	float top_color_dev = cidx_top >= 0 ? cv::norm(color_cache[cidx_top] - baseRegion.average_color) : 0;
	float bottom_color_dev = cidx_bottom >= 0 ? cv::norm(color_cache[cidx_bottom] - baseRegion.average_color) : 0;

	if(dispMin < 0)
	{
		std::swap(left_neighbor_disp, right_neighbor_disp);
		std::swap(left_color_dev, right_color_dev);
	}

	//	float costs, occ_avg, neighbor_pot, lr_pot ,neighbor_color_pot;
	result_vector.resize(range*vector_size_per_disp+vector_size);
	float org_size = baseRegion.size();
	float *result_ptr = result_vector.data();
	for(int i = 0; i < range; ++i)
	{
		*result_ptr++ = cost_values[i];
		*result_ptr++ = occ_avg_values[i];
		*result_ptr++ = neighbor_pot_values[i];
		*result_ptr++ = lr_pot_values[i];
		*result_ptr++ = neighbor_color_pot_values[i];
		*result_ptr++ = (float)occ_temp[i].first / org_size;
		//*result_ptr++ = rel_cost_values[i];
		int hyp_disp = dispMin + i;
		*result_ptr++ = left_neighbor_disp - hyp_disp;
		*result_ptr++ = right_neighbor_disp - hyp_disp;
		//*result_ptr++ = top_neighbor_disp - hyp_disp;
		//*result_ptr++ = bottom_neighbor_disp - hyp_disp;
		*result_ptr++ = disp_costs[i];
	}
	//*result_ptr = baseRegion.disparity;
	*result_ptr++ = *std::min_element(cost_values.begin(), cost_values.end());
	*result_ptr++ = left_color_dev;
	*result_ptr++ = right_color_dev;
	//*result_ptr++ = top_color_dev;
	//*result_ptr++ = bottom_color_dev;
}

disparity_hypothesis::disparity_hypothesis(const std::vector<float>& optimization_vector, int dispIdx)
{
	const float *ptr = optimization_vector.data() + dispIdx * disparity_hypothesis_vector::vector_size_per_disp;
	costs = *ptr++;
	occ_avg = *ptr++;
	neighbor_pot = *ptr++;
	lr_pot = *ptr++;
	neighbor_color_pot = *ptr++;

}

float calculate_end_result(const float *raw_results, const disparity_hypothesis_weight_vector& wv)
{
	return raw_results[0] * wv.costs + raw_results[1] * wv.occ_avg + raw_results[2] * wv.neighbor_pot + raw_results[3] * wv.lr_pot + raw_results[4] * wv.neighbor_color_pot;
	//return raw_results[0] * wv.costs + raw_results[1] * wv.occ_avg + raw_results[2] * wv.lr_pot + raw_results[3] * wv.neighbor_color_pot;
}

float calculate_end_result(int disp_idx, const float *raw_results, const disparity_hypothesis_weight_vector &wv)
{
	std::size_t idx_offset = disp_idx*disparity_hypothesis_vector::vector_size_per_disp;
	const float *result_ptr = raw_results + idx_offset;

	return calculate_end_result(result_ptr, wv);
}

void refreshOptimizationBaseValues(std::vector<std::vector<float>>& optimization_vectors, region_container& base, const region_container& match, const disparity_hypothesis_weight_vector& stat_eval, int delta)
{
	cv::Mat disp = disparity_by_segments(base);
	cv::Mat occmap = disparity::occlusion_stat<short>(disp, 1.0);
	int pot_trunc = 10;

	const short dispMin = base.task.dispMin;
	const short dispRange = base.task.dispMax - base.task.dispMin + 1;

	std::vector<disparity_hypothesis_vector> hyp_vec(omp_get_max_threads(), disparity_hypothesis_vector(base, match));
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

		intervals::substract_region_value<unsigned char>(occmaps[thread_idx], baseRegion.warped_interval, 1);

		baseRegion.optimization_energy = cv::Mat_<float>(dispRange, 1, 100.0f);

		disparity_range drange = task_subrange(base.task, baseRegion.base_disparity, delta);

		hyp_vec[thread_idx](occmaps[thread_idx], baseRegion, pot_trunc, drange , optimization_vectors[i]);
		for(short d = drange.start(); d <= drange.end(); ++d)
		{
			std::vector<corresponding_region>& cregionvec = baseRegion.corresponding_regions[d-dispMin];
			if(!cregionvec.empty())
				baseRegion.optimization_energy(d-dispMin) = calculate_end_result((d - drange.start()), optimization_vectors[i].data(), stat_eval);
		}

		intervals::add_region_value<unsigned char>(occmaps[thread_idx], baseRegion.warped_interval, 1);
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
	stream << "optimizer_type" << config.optimizer_type;

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
	stream["optimizer_type"] >> config.optimizer_type;

	return stream;
}

