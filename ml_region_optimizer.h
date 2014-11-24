#ifndef ML_REGION_OPTIMIZER_H
#define ML_REGION_OPTIMIZER_H

#include "region_optimizer.h"

class StereoSingleTask;

class ml_region_optimizer : public region_optimizer
{
public:
	void run(RegionContainer& left, RegionContainer& right, const optimizer_settings& config, int refinement= 0) override;
	void reset(const RegionContainer& left, const RegionContainer& right) override;

	void training() override;

	const static int vector_size = 6;
private:
	void refresh_base_optimization_vector(const RegionContainer& base, const RegionContainer& match, int delta);
	void prepare_training(std::vector<float>& dst, const std::vector<std::vector<float>>& base_optimization_vectors, const std::vector<std::vector<float>>& match_optimization_vectors, const RegionContainer& base, const RegionContainer& match, int delta);
	void optimize_ml(RegionContainer& base, RegionContainer& match, std::vector<std::vector<float>>& optimization_vectors_base, std::vector<std::vector<float>>& optimization_vectors_match, int delta);
	void gather_region_optimization_vector(float *dst_ptr, const DisparityRegion& baseRegion, const std::vector<float>& optimization_vector_base, const std::vector<std::vector<float>>& optimization_vectors_match, const RegionContainer& match, int delta, const StereoSingleTask& task, const std::vector<float>& normalization_vector);

	std::vector<std::vector<float>> optimization_vectors_left, optimization_vectors_right;

	std::vector<std::vector<float>> samples_left, samples_right;
	std::vector<int> samples_gt_left, samples_gt_right;
};

#endif
