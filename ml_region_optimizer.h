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

class StereoSingleTask;

class ml_region_optimizer : public region_optimizer
{
public:
	ml_region_optimizer();
	void run(RegionContainer& left, RegionContainer& right, const optimizer_settings& config, int refinement= 0) override;
	void reset(const RegionContainer& left, const RegionContainer& right) override;

	void training() override;

	const static int vector_size_per_disp = 6;
	const static int vector_size = 1;
	const static int normalizer_size = vector_size+vector_size_per_disp;

private:
	void refresh_base_optimization_vector(const RegionContainer& base, const RegionContainer& match, int delta);
	void prepare_training_sample(std::vector<std::vector<float>>& dst, const std::vector<std::vector<float>>& base_optimization_vectors, const std::vector<std::vector<float>>& match_optimization_vectors, const RegionContainer& base, const RegionContainer& match, int delta);
	void optimize_ml(RegionContainer& base, RegionContainer& match, std::vector<std::vector<float>>& optimization_vectors_base, std::vector<std::vector<float>>& optimization_vectors_match, int delta);
	void gather_region_optimization_vector(float *dst_ptr, const DisparityRegion& baseRegion, const std::vector<float>& optimization_vector_base, const std::vector<std::vector<float>>& optimization_vectors_match, const RegionContainer& match, int delta, const StereoSingleTask& task, const std::vector<float>& mean_normalization_vector, const std::vector<float>& stddev_normalization_vector);
	void reset_internal();
	std::vector<std::vector<float>> optimization_vectors_left, optimization_vectors_right;

	std::vector<std::vector<float>> samples_left, samples_right;
	std::vector<unsigned char> samples_gt_left;
};

#endif
