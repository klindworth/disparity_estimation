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

#ifndef MANUAL_REGION_OPTIMIZER_H
#define MANUAL_REGION_OPTIMIZER_H

#include "region_optimizer.h"

class manual_optimizer_feature_calculator : public disparity_features_calculator
{
public:
	manual_optimizer_feature_calculator(const region_container& left_regions, const region_container& right_regions) : disparity_features_calculator(left_regions, right_regions)
	{
	}

	void update(cv::Mat_<unsigned char>& occmap, short pot_trunc, const disparity_region& baseRegion, const disparity_range& drange);

	disparity_hypothesis get_disparity_hypothesis(int disp_idx) const
	{
		disparity_hypothesis result;
		result.costs = cost_values[disp_idx];
		result.lr_pot = lr_pot_values[disp_idx];
		result.neighbor_color_pot = neighbor_color_pot_values[disp_idx];
		result.neighbor_pot = neighbor_pot_values[disp_idx];
		result.occ_avg = occ_avg_values[disp_idx];
		result.warp_costs = warp_costs_values[disp_idx];

		return result;
	}
};

class manual_region_optimizer : public region_optimizer
{
public:
	void run(region_container& left, region_container& right, const optimizer_settings& config, int refinement= 0) override;
	void optimize(std::vector<unsigned char>& damping_history, region_container& base, region_container& match, const disparity_hypothesis_weight_vector& stat_eval, std::function<float(const disparity_region&, const region_container&, const region_container&, int)> prop_eval, int delta);
	void reset(const region_container& left, const region_container& right) override;

	void training() override;

private:
	std::vector<unsigned char> damping_history_left, damping_history_right;
};

void refreshOptimizationBaseValues(region_container& left, const region_container& match, const disparity_hypothesis_weight_vector& base_eval, int delta);

#endif
