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

template<typename T>
inline T abs_pott(const T& v1, const T& v2, const T& trunc)
{
	return std::min(std::abs(v1 - v2), trunc);
}

class RegionContainer;
class SegRegion;
class InitialDisparityConfig;
class StereoTask;

namespace cv {
	class Mat;
	class FileStorage;
	class FileNode;
}

class disparity_hypothesis
{
public:
	disparity_hypothesis() {}
	disparity_hypothesis(cv::Mat& occmap, const SegRegion& baseRegion, short disparity, const std::vector<SegRegion>& left_regions, const std::vector<SegRegion> &right_regions, int pot_trunc, int dispMin);
	disparity_hypothesis abs_delta(const disparity_hypothesis& base) const;
	disparity_hypothesis delta(const disparity_hypothesis& base) const;

	float costs, occ_avg, neighbor_pot, lr_pot ,neighbor_color_pot;
};

class config_term
{
public:
	double cost, color_disp, lr_pot, occ;
};

class optimizer_settings
{
public:
	int rounds;
	bool enable_damping;

	config_term base;

	std::function<float(const disparity_hypothesis&)> base_eval;
	std::function<float(const SegRegion&, const RegionContainer&, const RegionContainer&, int)> prop_eval;

	std::function<float(const disparity_hypothesis&)> base_eval2;
	std::function<float(const SegRegion&, const RegionContainer&, const RegionContainer&, int)> prop_eval2;

	std::function<float(const disparity_hypothesis&)> base_eval_refine;
	std::function<float(const SegRegion&, const RegionContainer&, const RegionContainer&, int)> prop_eval_refine;

};

std::vector<std::size_t> regionSplitUp(RegionContainer& base, RegionContainer& match);
void optimize(RegionContainer& base, RegionContainer& match, std::function<float(const disparity_hypothesis&)> stat_eval, std::function<float(const SegRegion&, const RegionContainer&, const RegionContainer&, int)> prop_eval, int delta);
void refreshOptimizationBaseValues(RegionContainer& left, RegionContainer& match, std::function<float(const disparity_hypothesis&)> base_eval, int delta);
void run_optimization(StereoTask& task, RegionContainer& left, RegionContainer& right, const optimizer_settings& config, int refinement= 0);

cv::FileStorage& operator<<(cv::FileStorage& stream, const optimizer_settings& config);
const cv::FileNode& operator>>(const cv::FileNode& stream, optimizer_settings& config);

#endif // REGION_OPTIMIZER_H
