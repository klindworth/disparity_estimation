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
#include <memory>

class StereoSingleTask;

template<typename T>
class neural_network;

class result_eps_calculator
{
public:
	result_eps_calculator()
	{
		std::fill(counters.begin(), counters.end(), 0);
		total = 0;
	}

	void operator()(short gt, short estimated)
	{
		++total;
		unsigned int diff = std::abs(std::abs(estimated) - std::abs(gt));

		for(unsigned int i = diff; i < 11; ++i)
			++counters[i];
	}

	float epsilon_result(unsigned int eps) const
	{
		if(eps < 11)
			return (float)counters[eps]/total;
		else
			return 1.0f;
	}

	void print_to_stream(std::ostream& stream) const
	{
		stream << "correct: " << (float)counters[0]/total << ", approx5: " << (float)counters[4]/total << ", approx10: " << (float)counters[9]/total;
	}

private:
	std::array<unsigned int, 11> counters;
	unsigned int total = 0;
};

class ml_region_optimizer : public region_optimizer
{
public:
	ml_region_optimizer();
	~ml_region_optimizer();
	void run(region_container& left, region_container& right, const optimizer_settings& config, int refinement= 0) override;
	void reset(const region_container& left, const region_container& right) override;

	void training() override;

	const static int vector_size_per_disp = 7;
	const static int vector_size = 1;
	const static int normalizer_size = vector_size+vector_size_per_disp;

private:
	void refresh_base_optimization_vector(const region_container& base, const region_container& match, int delta);
	void prepare_training_sample(std::vector<short>& dst_gt, std::vector<std::vector<double>>& dst_data, const std::vector<std::vector<float>>& base_optimization_vectors, const std::vector<std::vector<float>>& match_optimization_vectors, const region_container& base, const region_container& match, int delta);
	void optimize_ml(region_container& base, const region_container& match, std::vector<std::vector<float>>& optimization_vectors_base, std::vector<std::vector<float>>& optimization_vectors_match, int delta, const std::string& filename);
	void reset_internal();
	std::vector<std::vector<float>> optimization_vectors_left, optimization_vectors_right;

	std::vector<std::vector<double>> samples_left, samples_right;
	std::vector<short> samples_gt_left, samples_gt_right;

	int training_iteration;
	std::string filename_left_prefix, filename_right_prefix;

	std::unique_ptr<neural_network<double>> nnet;
	result_eps_calculator total_diff_calc;
};

//void gather_region_optimization_vector(double *dst_ptr, const DisparityRegion& baseRegion, const std::vector<float>& optimization_vector_base, const std::vector<std::vector<float>>& optimization_vectors_match, const RegionContainer& match, int delta, const StereoSingleTask& task);

#endif
