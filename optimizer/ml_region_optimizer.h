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

#ifndef ML_REGION_OPTIMIZER_H
#define ML_REGION_OPTIMIZER_H

#include "region_optimizer.h"
#include "region_ground_truth.h"
#include <memory>

class single_stereo_task;

namespace neural_network
{
template<typename T>
class network;
}

class ml_feature_calculator : public disparity_features_calculator
{
public:
	ml_feature_calculator(const region_container& left_regions, const region_container& right_regions) : disparity_features_calculator(left_regions, right_regions) {}
	void operator()(const disparity_region& baseRegion, short pot_trunc, const disparity_range& drange, std::vector<float>& result_vector);
	void update_result_vector(std::vector<float>& result_vector, const disparity_region& baseRegion, const disparity_range& drange);
};

class ml_region_optimizer : public region_optimizer
{
public:
	ml_region_optimizer();
	~ml_region_optimizer();
	void run(region_container& left, region_container& right, const optimizer_settings& config, int refinement= 0) override;
	void reset(const region_container& left, const region_container& right) override;

	void training() override;

	const static int vector_size_per_disp = 11;
	const static int vector_size = 5;
	const static int normalizer_size = vector_size+vector_size_per_disp;

private:
	void refresh_base_optimization_vector(const region_container& base, const region_container& match, int delta);
	void prepare_training_sample(std::vector<short>& dst_gt, std::vector<std::vector<double>>& dst_data, const std::vector<std::vector<float>>& base_optimization_vectors, const std::vector<std::vector<float>>& match_optimization_vectors, const region_container& base, const region_container& match, int delta);
	void optimize_ml(region_container& base, const region_container& match, std::vector<std::vector<float>>& optimization_vectors_base, std::vector<std::vector<float>>& optimization_vectors_match, int delta, const std::string& filename);
	void reset_internal();
	void prepare_per_disp_training_sample(std::vector<short>& dst_gt, std::vector<std::vector<double>>& dst_data, const std::vector<std::vector<float>>& base_optimization_vectors, const std::vector<std::vector<float>>& match_optimization_vectors, const region_container& base, const region_container& match, int delta);
	void optimize_ml_disp(region_container& base, const region_container& match, std::vector<std::vector<float>>& optimization_vectors_base, std::vector<std::vector<float>>& optimization_vectors_match, int delta, const std::string& filename);
	std::vector<std::vector<float>> optimization_vectors_left, optimization_vectors_right;

	std::vector<std::vector<double>> samples_left, samples_right;
	std::vector<short> samples_gt_left, samples_gt_right;
	std::vector<std::vector<double>> per_disp_samples_left, per_disp_samples_right;
	std::vector<short> per_disp_samples_gt_left, per_disp_samples_gt_right;

	int training_iteration;
	std::string filename_left_prefix, filename_right_prefix;

	std::unique_ptr<neural_network::network<double>> nnet;
	result_eps_calculator total_diff_calc;
};

#endif
