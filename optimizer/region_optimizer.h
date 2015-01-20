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

#ifndef REGION_OPTIMIZER_H
#define REGION_OPTIMIZER_H

#include "disparity_range.h"

#include <vector>
#include <functional>
#include <memory>

#include <opencv2/core/core.hpp>

template<typename T>
inline T abs_pott(const T& v1, const T& v2, const T& trunc)
{
	return std::min((T)std::abs(v1 - v2), trunc);
}

class region_container;
class disparity_region;
class initial_disparity_config;
class stereo_task;
class stat_t;

/*namespace cv {
	class Mat;
	template<typename T>
	class Mat_;
	class FileStorage;
	class FileNode;
}*/

struct single_neighbor_values
{
	single_neighbor_values(const std::vector<short>& disparities, const std::vector<cv::Vec3d>& colors, const disparity_region& baseRegion, int neigh_idx);

	float color_dev;
	short disparity;
};

struct neighbor_values
{
	neighbor_values(const std::vector<short>& disparities, const std::vector<cv::Vec3d>& colors, const disparity_region& baseRegion, int idx_left, int idx_right, int idx_top, int idx_bottom)
		: left(disparities, colors, baseRegion, idx_left), right(disparities, colors, baseRegion, idx_right), top(disparities, colors, baseRegion, idx_top), bottom(disparities, colors, baseRegion, idx_bottom)
	{}
	single_neighbor_values left, right, top, bottom;
};

struct disparity_hypothesis
{
	float costs, occ_avg, neighbor_pot, lr_pot ,neighbor_color_pot, warp_costs;
};

struct disparity_hypothesis_weight_vector
{
	float costs, occ_avg, neighbor_pot, lr_pot ,neighbor_color_pot, warp_costs;

	inline float evaluate_hypthesis(const disparity_hypothesis& hyp) const
	{
		return hyp.costs * costs + hyp.occ_avg * occ_avg + hyp.neighbor_pot * neighbor_pot + hyp.lr_pot * lr_pot + hyp.neighbor_color_pot * neighbor_color_pot;
	}
};

class shared_features_cache
{
public:
	shared_features_cache(const region_container& left_regions, const region_container& right_regions);

	std::vector<cv::Point> base_avg_cache;
	std::vector<short> base_disparities_cache, match_disparities_cache;
	std::vector<cv::Vec3d> color_cache;
	cv::Mat_<float> warp_costs;
};

class disparity_features_calculator
{
protected:
	//construction-time calculated
	const std::shared_ptr<const shared_features_cache> cache;


	cv::Mat_<unsigned char> occmap;
	//temps
	std::vector<std::pair<int, int> > occ_temp;
	std::vector<short> neighbor_disparities;
	std::vector<float> neighbor_color_weights;
	std::vector<std::pair<int, float> > cost_temp;

	//end results
	std::vector<float> occ_avg_values, neighbor_pot_values, neighbor_color_pot_values, lr_pot_values, cost_values, rel_cost_values, warp_costs_values;

	void update_average_neighbor_values(const disparity_region& baseRegion, short pot_trunc, const disparity_range& drange);
	void update_lr_pot(const disparity_region& baseRegion, short pot_trunc, const disparity_range& drange);
	void update_occ_avg(const disparity_region& baseRegion, short pot_trunc, const disparity_range& drange);
	void update_warp_costs(const disparity_region& baseRegion, const disparity_range& drange);
	neighbor_values get_neighbor_values(const disparity_region& baseRegion, const disparity_range& drange);

public:
	disparity_features_calculator(const region_container& left_regions, const region_container& right_regions);
	disparity_features_calculator(const disparity_features_calculator& org);
};

class optimizer_settings
{
public:
	int rounds;
	bool enable_damping;
	std::string optimizer_type;

	std::function<float(const disparity_region&, const region_container&, const region_container&, int, const stat_t&)> prop_eval, prop_eval2;
	disparity_hypothesis_weight_vector base_eval, base_eval2;
};

class region_optimizer
{
public:
	region_optimizer() {training_mode = false;}
	virtual void run(region_container& left, region_container& right, const optimizer_settings& config, int refinement= 0) = 0;
	//void optimize(std::vector<unsigned char>& damping_history, std::vector<std::vector<float>>& optimization_vectors_base, std::vector<std::vector<float>>& optimization_vectors_match, RegionContainer& base, RegionContainer& match, const disparity_hypothesis_weight_vector& stat_eval, std::function<float(const DisparityRegion&, const RegionContainer&, const RegionContainer&, int)> prop_eval, int delta);
	virtual void reset(const region_container& left, const region_container& right) = 0;

	virtual void set_training_mode(bool training) {training_mode = training;}
	virtual void training() = 0;

protected:
	bool training_mode;
};

cv::FileStorage& operator<<(cv::FileStorage& stream, const optimizer_settings& config);
const cv::FileNode& operator>>(const cv::FileNode& stream, optimizer_settings& config);

#endif // REGION_OPTIMIZER_H
