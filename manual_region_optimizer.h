#ifndef MANUAL_REGION_OPTIMIZER_H
#define MANUAL_REGION_OPTIMIZER_H

#include "region_optimizer.h"

class manual_region_optimizer : public region_optimizer
{
public:
	void run(RegionContainer& left, RegionContainer& right, const optimizer_settings& config, int refinement= 0) override;
	void optimize(std::vector<unsigned char>& damping_history, std::vector<std::vector<float>>& optimization_vectors_base, std::vector<std::vector<float>>& optimization_vectors_match, RegionContainer& base, RegionContainer& match, const disparity_hypothesis_weight_vector& stat_eval, std::function<float(const DisparityRegion&, const RegionContainer&, const RegionContainer&, int)> prop_eval, int delta);
	void reset(const RegionContainer& left, const RegionContainer& right) override;

	void training() override;

private:
	std::vector<unsigned char> damping_history_left, damping_history_right;
	std::vector<std::vector<float>> optimization_vectors_left, optimization_vectors_right;
};

#endif
