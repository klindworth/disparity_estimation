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

#ifndef REGION_METRICS_H
#define REGION_METRICS_H

#include "sliding_entropy.h"
#include <segmentation/intervals.h>
#include "costmap_creators.h"


template<int quantizer>
class region_info_thread
{
public:
	static const int bins = 256/quantizer;
	fast_array2d<unsigned int, bins+2, bins+2> counter_joint;
	fast_array<unsigned int, bins+2> counter_array;
};

template<typename T, int bins, typename counter, typename joint_counter, typename entropy_type>
std::tuple<T, T, T> entropies_calculator(joint_counter& counter_joint, counter& counter_array, const entropy_type& entropy_table, const cv::Mat& base_region, const cv::Mat& match_region)
{
	using namespace costmap_creators::entropy;
	calculate_joint_soft_histogramm(counter_joint, base_region.data, match_region.data, base_region.total());
	T joint_entropy;
	if(base_region.total()*6 > bins*bins)
		joint_entropy = calculate_joint_entropy_unnormalized<T>(counter_joint, entropy_table, bins);
	else
		joint_entropy = calculate_joint_entropy_unnormalized_sparse<T>(counter_joint, entropy_table, bins, base_region.total(), base_region.data, match_region.data);

	calculate_soft_histogramm(counter_array, base_region.data, base_region.total());
	T base_entropy = calculate_entropy_unnormalized<T>(counter_array, entropy_table, bins);

	calculate_soft_histogramm(counter_array, match_region.data, match_region.total());
	T match_entropy = calculate_entropy_unnormalized<T>(counter_array, entropy_table, bins);

	return std::make_tuple(joint_entropy, base_entropy, match_entropy);
}

template<typename cost_calc, int quantizer>
class region_info_disparity
{
public:
	typedef region_info_thread<quantizer> thread_type;
	static const int bins = 256/quantizer;

	cv::Mat_<float> entropy_table;

	cost_calc calculator;
	cv::Mat m_base, m_match;

	region_info_disparity(const cv::Mat& base, const cv::Mat& match, int size) : m_base(base), m_match(match)
	{
		costmap_creators::entropy::fill_entropytable_unnormalized(entropy_table, size*9);
	}

	static float normalization_value()
	{
		return cost_calc::upper_bound();
	}

	float operator()(thread_type& thread, int d, const std::vector<region_interval>& region)
	{
		cv::Mat base_region  = region_as_mat(m_base,  region, 0);
		cv::Mat match_region = region_as_mat(m_match, region, d);

		assert(base_region.total() == match_region.total());

		auto result = entropies_calculator<float, bins>(thread.counter_joint, thread.counter_array, entropy_table, base_region, match_region);

		return calculator(std::get<0>(result), std::get<1>(result), std::get<2>(result));
	}
};

template<typename cost_type>
void region_disparity_internal(std::vector<region_interval>& actual_region, cost_type& cost_agg, typename cost_type::thread_type& thread, disparity_region& cregion, const cv::Mat&, const cv::Mat& match, int dispMin, int dispMax)
{
	int length = size_of_region(actual_region);

	//cregion.confidence = cv::Mat(dispMax-dispMin+1, 1, CV_32FC1, cv::Scalar(0));
	cregion.disparity_costs = cv::Mat(dispMax-dispMin+1, 1, CV_32FC1, cv::Scalar(cost_agg.normalization_value()));

	float minCost = std::numeric_limits<float>::max();
	short minD = cregion.base_disparity;

	for(int d = dispMin; d <= dispMax; ++d)
	{
		std::vector<region_interval> filtered = filtered_region(match.size[1], actual_region, d);
		int filtered_length = size_of_region(filtered);

		if((float)filtered_length/(float)length > 0.6f && filtered_length > 10) //0.4
		{
			float cost = cost_agg(thread, d, filtered);

			assert(d-dispMin >= 0 && d-dispMin < cregion.disparity_costs.size[0]);

			cregion.disparity_costs(d-dispMin) = cost;

			if(minCost > cost)
			{
				minD = d;
				minCost = cost;
			}
		}
	}
	cregion.disparity = minD;
}

#endif // REGION_METRICS_H
