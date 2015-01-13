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

single_neighbor_values::single_neighbor_values(const std::vector<short>& disparities, const std::vector<cv::Vec3d>& colors, const disparity_region& baseRegion, int neigh_idx) {
	disparity = neigh_idx >= 0 ? disparities[neigh_idx] : baseRegion.disparity;
	color_dev = neigh_idx >= 0 ? cv::norm(colors[neigh_idx] - baseRegion.average_color) : 0;
}

disparity_features_calculator::disparity_features_calculator(const region_container& base, const region_container& match) : base_avg_cache(base.regions.size()), base_disparities_cache(base.regions.size()), match_disparities_cache(match.regions.size()), color_cache(base.regions.size())
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

void disparity_features_calculator::update_average_neighbor_values(const disparity_region& baseRegion, short pot_trunc, const disparity_range& drange)
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

void disparity_features_calculator::update_lr_pot(const disparity_region& baseRegion, short pot_trunc, const disparity_range& drange)
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

void disparity_features_calculator::update_occ_avg(const cv::Mat_<unsigned char>& occmap, const disparity_region& baseRegion, short /*pot_trunc*/, const disparity_range& drange)
{
	const int range = drange.size();
	occ_temp.resize(range);
	occ_avg_values.resize(range);

	region_descriptors::segment_boxfilter(occ_temp, occmap, baseRegion.lineIntervals, drange.start(), drange.end());

	for(int i = 0; i < range; ++i)
		occ_avg_values[i] = (occ_temp[i].first != 0) ? (float)occ_temp[i].second / occ_temp[i].first : -1;
}

void disparity_features_calculator::update_warp_costs(const disparity_region& baseRegion, const disparity_range& drange)
{
	const int range = drange.size();

	warp_costs_values.resize(range);
	cost_temp.resize(range);

	region_descriptors::segment_boxfilter(cost_temp, warp_costs, baseRegion.lineIntervals, drange.start(), drange.end());
	for(int i = 0; i < range; ++i)
		warp_costs_values[i] = (cost_temp[i].first != 0) ? baseRegion.disparity_costs(i)/(float)cost_temp[i].second * cost_temp[i].first : 0;
}

neighbor_values disparity_features_calculator::get_neighbor_values(const disparity_region& baseRegion, const disparity_range& drange)
{
	int cidx_left = -1;
	int cidx_right = -1;
	int cidx_top = -1;
	int cidx_bottom = -1;

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

	if(drange.start() < 0)
		std::swap(cidx_left, cidx_right);

	return neighbor_values(base_disparities_cache, color_cache, baseRegion, cidx_left, cidx_right, cidx_top, cidx_bottom);
}



disparity_hypothesis::disparity_hypothesis(const std::vector<float>& optimization_vector, int dispIdx)
{
	const float *ptr = optimization_vector.data() + dispIdx * disparity_features_calculator::vector_size_per_disp;
	costs = *ptr++;
	occ_avg = *ptr++;
	neighbor_pot = *ptr++;
	lr_pot = *ptr++;
	neighbor_color_pot = *ptr++;

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

