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

#include <vector>
#include <cmath>
#include <algorithm>
#include <functional>

#include <opencv2/core/core.hpp>

template<typename T>
inline T abs_pott(const T& v1, const T& v2, const T& trunc)
{
	return std::min((T)std::abs(v1 - v2), trunc);
}

class RegionContainer;
class DisparityRegion;
class InitialDisparityConfig;
class StereoTask;

namespace cv {
	class Mat;
	template<typename T>
	class Mat_;
	class FileStorage;
	class FileNode;
}

class disparity_hypothesis
{
public:
	disparity_hypothesis() {}
	disparity_hypothesis(const std::vector<float>& optimization_vector, int dispIdx);

	float costs, occ_avg, neighbor_pot, lr_pot ,neighbor_color_pot;
};

struct disparity_hypothesis_weight_vector
{
	float costs, occ_avg, neighbor_pot, lr_pot ,neighbor_color_pot;
};

class disparity_hypothesis_vector
{
	int dispRange, dispStart;
	//temps
	std::vector<std::pair<int, int> > occ_temp;
	std::vector<short> neighbor_disparities;
	std::vector<float> neighbor_color_weights;

	std::vector<short> base_disparities_cache, match_disparities_cache;
	std::vector<cv::Vec3d> color_cache;

	//end results
	std::vector<float> occ_avg_values, neighbor_pot_values, neighbor_color_pot_values, lr_pot_values, cost_values;

public:
	disparity_hypothesis_vector(const std::vector<DisparityRegion>& left_regions, const std::vector<DisparityRegion>& right_regions);
	void operator()(const cv::Mat_<unsigned char>& occmap, const DisparityRegion& baseRegion, short pot_trunc, int dispMin, int dispStart, int dispEnd, std::vector<float>& result_vector);
};

class optimizer_settings
{
public:
	int rounds;
	bool enable_damping;

	std::function<float(const DisparityRegion&, const RegionContainer&, const RegionContainer&, int)> prop_eval, prop_eval2, prop_eval_refine;
	disparity_hypothesis_weight_vector base_eval, base_eval2, base_eval_refine;
};

std::vector<std::size_t> regionSplitUp(RegionContainer& base, RegionContainer& match);
void refreshOptimizationBaseValues(std::vector<std::vector<float>>& optimization_vectors, RegionContainer& left, const RegionContainer& match, const disparity_hypothesis_weight_vector& base_eval, int delta);

cv::FileStorage& operator<<(cv::FileStorage& stream, const optimizer_settings& config);
const cv::FileNode& operator>>(const cv::FileNode& stream, optimizer_settings& config);

#endif // REGION_OPTIMIZER_H
