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

#ifndef COSTMAP_UTILS_H
#define COSTMAP_UTILS_H

#include <opencv2/core/core.hpp>

class RegionInterval;
class DisparityRegion;

//saves the statistics for an pixel in a cost map
typedef struct statistics {
	float mean;
	float stddev;
	float min;
	float max;
	float confidence;
	float confidence2;
	float confidence_range;
	float confidence_variance;
	float occ_val;
	int disparity_idx;
	std::vector<int> minima;
	std::vector<int> bad_minima;
	std::vector<RegionInterval> minima_ranges;
} stat_t;

cv::Mat deriveCostmap(const cv::Mat &cost_map);
void derivePartialCostmap(const float *cost_map, float *result, int len);
void analyzeDisparityRange(stat_t& cstat, const float *src_ptr, const float *derived_ptr, int range);
void analyzeDisparityRange2(DisparityRegion& region);

//sums all values up in a window and saves the sum in the middlepoint (like boxfilter - but here for 3D matrix)
cv::Mat windowSum(cv::Mat& cost_map_old, int windowsize);

#endif // COSTMAP_UTILS_H
