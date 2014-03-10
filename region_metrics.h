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

#include "slidingEntropy.h"
#include "intervals.h"
#include "costmap_creators.h"


template<int quantizer>
class RegionInfoThread
{
public:
	static const int bins = 256/quantizer;
	fast_array2d<unsigned int, bins+2, bins+2> counter_joint;
	fast_array<unsigned int, bins+2> counter_array;
};

template<typename T, int bins, typename counter, typename joint_counter, typename entropy_type>
std::tuple<T, T, T> entropies_calculator(joint_counter& counter_joint, counter& counter_array, const entropy_type& entropy_table, const cv::Mat& base_region, const cv::Mat& match_region)
{
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

/*class NormalMetricDisparityWiseCalculator
{
public:

	cv::Mat operator()(const cv::Mat& base, const cv::Mat& match, d)
	{
		cv::Mat pbase = prepare_base(base, d);
		cv::Mat pmatch = prepare_match(match, d);

		return cv::absdiff(pbase, pmatch, CV_L1);
	}
};

class RegionDisparityWiseCalculator
{
public:

	cv::Mat m_base, m_match;

	RegionDisparityWiseCalculator(const cv::Mat& base, const cv::Mat& match, int) : m_base(base), m_match(match)
	{
		auto window_sum = [](const cv::Mat& input) { return input; };
		disparitywise_calculator()
	}
};

template<typename cost_type, typename window_type>
cv::Mat region_disparitywise_calculator(cost_type cost_func, window_type window_sum, cv::Mat base, int dispMin, int dispMax)
{
	int sz[] = {base.rows, base.cols, dispMax - dispMin + 1};
	cv::Mat_<float> result = cv::Mat(3, sz, CV_32FC1, cv::Scalar(-std::numeric_limits<float>::max()));

	#pragma omp parallel for
	for(int d = dispMin; d <= dispMax; ++d)
	{
		cv::Mat temp_result = cost_func(d);

		for(int y = 0; y < base.rows; ++y)
		{
			if(d < 0)
			{
				for(int x = -d; x < base.cols; ++x)
					result(y,x,d-dispMin) = temp_result.at<float>(y, x+d);
			}
			else
			{
				int max_x = base.cols - d;
				for(int x = 0; x < max_x; ++x)
					result(y,x,d-dispMin) = temp_result.at<float>(y, x);
			}
		}
	}
	return result;
}*/

template<typename cost_calc, int quantizer>
class RegionInfoDisparity
{
public:
	typedef RegionInfoThread<quantizer> thread_type;
	static const int bins = 256/quantizer;

	cv::Mat_<float> entropy_table;

	cost_calc calculator;
	cv::Mat m_base, m_match;

	RegionInfoDisparity(const cv::Mat& base, const cv::Mat& match, int size) : m_base(base), m_match(match)
	{
		fill_entropytable_unnormalized(entropy_table, size*9);
	}

	static float normalizationValue()
	{
		return cost_calc::upper_bound();
	}

	float operator()(thread_type& thread, int d, const std::vector<RegionInterval>& region)
	{
		cv::Mat base_region  = getRegionAsMat(m_base,  region, 0);
		cv::Mat match_region = getRegionAsMat(m_match, region, d);

		assert(base_region.total() == match_region.total());
		/*calculate_joint_soft_histogramm(thread.counter_joint, base_region.data, match_region.data, base_region.total());
		float joint_entropy = calculate_joint_entropy_unnormalized<float>(thread.counter_joint, entropy_table, bins);

		calculate_soft_histogramm(thread.counter_array, base_region.data, base_region.total());
		float base_entropy = calculate_entropy_unnormalized<float>(thread.counter_array, entropy_table, bins);

		calculate_soft_histogramm(thread.counter_array, match_region.data, match_region.total());
		float match_entropy = calculate_entropy_unnormalized<float>(thread.counter_array, entropy_table, bins);

		return calculator(joint_entropy, base_entropy, match_entropy);*/

		auto result = entropies_calculator<float, bins>(thread.counter_joint, thread.counter_array, entropy_table, base_region, match_region);

		return calculator(std::get<0>(result), std::get<1>(result), std::get<2>(result));
	}
};

template<typename cost_calc, int quantizer>
class RegionInfoDisparityConf
{
public:
	typedef RegionInfoThread<quantizer> thread_type;
	static const int bins = 256/quantizer;

	cv::Mat_<float> entropy_table;

	cost_calc calculator;

	cv::Mat m_base, m_match;

	RegionInfoDisparityConf(const cv::Mat& base, const cv::Mat& match, int size) : m_base(base), m_match(match)
	{
		fill_entropytable_unnormalized(entropy_table, size*9);
	}

	static float normalizationValue()
	{
		return cost_calc::upper_bound();
	}

	std::pair<float,float> operator()(thread_type& thread, int d, const std::vector<RegionInterval>& region)
	{
		cv::Mat base_region  = getRegionAsMat(m_base,  region, 0);
		cv::Mat match_region = getRegionAsMat(m_match, region, d);

		assert(base_region.total() == match_region.total());

		/*calculate_joint_soft_histogramm(thread.counter_joint, base_region.data, match_region.data, base_region.total());
		float joint_entropy;
		if(base_region.total()*6 > bins*bins)
			joint_entropy = calculate_joint_entropy_unnormalized<float>(thread.counter_joint, entropy_table, bins);
		else
			joint_entropy = calculate_joint_entropy_unnormalized_sparse<float>(thread.counter_joint, entropy_table, bins, base_region.total(), base_region.data, match_region.data);

		calculate_soft_histogramm(thread.counter_array, base_region.data, base_region.total());
		float base_entropy = calculate_entropy_unnormalized<float>(thread.counter_array, entropy_table, bins);

		calculate_soft_histogramm(thread.counter_array, match_region.data, match_region.total());
		float match_entropy = calculate_entropy_unnormalized<float>(thread.counter_array, entropy_table, bins);*/

		auto result = entropies_calculator<float, bins>(thread.counter_joint, thread.counter_array, entropy_table, base_region, match_region);

		return std::make_pair(calculator(std::get<0>(result), std::get<1>(result), std::get<2>(result)), std::get<1>(result)+ std::get<2>(result)-std::get<0>(result));
		//return std::make_pair(calculator(joint_entropy, base_entropy, match_entropy), base_entropy+match_entropy-joint_entropy);
	}
};

//template<typename cost_type>
//std::pair<float, float> getRegionDisparityInternal2(std::vector<)

inline std::vector<RegionInterval> filter_region(std::vector<RegionInterval>& actual_region, int d, const std::vector<RegionInterval>& occ, int width)
{
	std::vector<RegionInterval> filtered_pixel_idx = getFilteredPixelIdx(width, actual_region, d);

	if(!occ.empty())
	{
		std::vector<RegionInterval> occ_filtered;
		intervals::difference(filtered_pixel_idx.begin(), filtered_pixel_idx.end(), occ.begin(), occ.end(), d, std::back_inserter(occ_filtered));
		return occ_filtered;
	}
	else
		return filtered_pixel_idx;
}

template<typename cost_type>
void getRegionDisparityInternal(std::vector<RegionInterval>& actual_region, cost_type& cost_agg, typename cost_type::thread_type& thread, DisparityRegion &cregion, const cv::Mat& base, const cv::Mat& match, int dispMin, int dispMax, const std::vector<RegionInterval>& occ)
{
	int length = getSizeOfRegion(actual_region);

	cregion.confidence = cv::Mat(dispMax-dispMin+1, 1, CV_32FC1, cv::Scalar(0));
	cregion.disparity_costs = cv::Mat(dispMax-dispMin+1, 1, CV_32FC1, cv::Scalar(cost_agg.normalizationValue()));

	float minCost = std::numeric_limits<float>::max();
	short minD = cregion.base_disparity;

	for(int d = dispMin; d <= dispMax; ++d)
	{
		std::vector<RegionInterval> filtered = filter_region(actual_region, d, occ, match.size[1]);
		int filtered_length = getSizeOfRegion(filtered);

		if((float)filtered_length/(float)length > 0.6f && filtered_length > 10) //0.4
		{
			std::pair<float, float> cost = cost_agg(thread, d, filtered);

			assert(d-dispMin >= 0 && d-dispMin < cregion.disparity_costs.size[0]);

			cregion.disparity_costs(d-dispMin) = cost.first;
			cregion.confidence(d-dispMin) = cost.second;

			if(minCost > cost.first)
			{
				minD = d;
				minCost = cost.first;
			}
		}
	}
	cregion.disparity = minD;
}

/**
 * @brief getRegionDisparity
 * @param cost_map the costmap that will be filled. Memory must be allocated before calling this function!
 * @param base Image
 * @param match Image
 * @param pixel_idx Vector of all pixel indices of a region
 * @param dispMin
 * @param dispMax
 */
template<typename cost_type>
void getRegionDisparity(cost_type& cost_agg, typename cost_type::thread_type& thread, DisparityRegion &cregion, const cv::Mat& base, const cv::Mat& match, int dispMin, int dispMax, unsigned int dilate_grow, const std::vector<RegionInterval>& occ)
{
	int dilate = cregion.dilation;
	if(dilate == cregion.old_dilation)
		return;

	std::vector<RegionInterval> actual_pixel_idx = getDilatedRegion(cregion, dilate, base);

	getRegionDisparityInternal(actual_pixel_idx, cost_agg, thread, cregion, base, match, dispMin, dispMax, occ);

	cregion.old_dilation = dilate;
}

/*template<typename cost_type>
void getRegionDisparity(SegRegion &cregion, const cv::Mat& base, const cv::Mat& match, int dispMin, int dispMax, unsigned int dilate_grow, const std::vector<RegionInterval>& occ)
{
	for(unsigned int dilate = 0; dilate <= dilate_grow; dilate+=4)
	{
		cregion.unprocessed = true;
		std::vector<RegionInterval> actual_pixel_idx = getDilatedRegion(cregion, dilate, base);

		int length = getSizeOfRegion(actual_pixel_idx);

		cost_type cost_agg(length);

		cregion.confidence = cv::Mat(dispMax-dispMin+1, 1, CV_32FC1, cv::Scalar(0));
		cregion.disparity_costs = cv::Mat(dispMax-dispMin+1, 1, CV_32FC1, cv::Scalar(cost_agg.normalizationValue()));
		cregion.unprocessed = cv::Mat_<unsigned char>(dispMax-dispMin+1, 1, 1);

		float minCost = std::numeric_limits<float>::max();
		short minD = dispMin;
		for(int d = dispMin; d <= dispMax; ++d)
		{
			std::vector<RegionInterval> filtered_pixel_idx = getFilteredPixelIdx(match.size[1], actual_pixel_idx, d);

			std::vector<RegionInterval> *actual_filtered = &filtered_pixel_idx;
			std::vector<RegionInterval> occ_filtered;
			if(!occ.empty())
			{
				intervals::difference(filtered_pixel_idx.begin(), filtered_pixel_idx.end(), occ.begin(), occ.end(), d, std::back_inserter(occ_filtered));
				actual_filtered = &occ_filtered;
			}
			int filtered_length = getSizeOfRegion(*actual_filtered);

			if((float)filtered_length/(float)length > 0.6f && filtered_length > 10) //0.4
			{
				cregion.unprocessed(d-dispMin) = 0;
				cv::Mat base_region  = getRegionAsMat(base,  *actual_filtered, 0);
				cv::Mat match_region = getRegionAsMat(match, *actual_filtered, d);
				assert(base_region.total() == match_region.total());

				std::pair<float,float> cost = cost_agg(base_region, match_region);

				assert(d-dispMin >= 0 && d-dispMin < cregion.disparity_costs.size[0]);

				cregion.disparity_costs(d-dispMin) = cost.first;
				cregion.confidence(d-dispMin) = cost.second;

				if(minCost > cost.first)
				{
					minD = d;
					minCost = cost.first;
				}
			}
		}
		cregion.disparity = minD;
		cregion.dilation = dilate;

		//analyze for further dilation rounds
		int len = cregion.disparity_costs.size[0];
		float *derived = new float[len-1];
		const float *costs = cregion.disparity_costs[0];
		derivePartialCostmap(costs, derived, len);
		analyzeDisparityRange(cregion.stats, costs, derived, len);

		if((cregion.stats.minima.size() < 2) && !(cregion.stats.minima.size() == 0 && cregion.stats.bad_minima.size() > 2))
			return;
	}
}*/

/*template<typename cost_type>
void getRegionDisparity(SegRegion &cregion, const cv::Mat& base, const cv::Mat& match, int dispMin, int dispMax, unsigned int dilate_grow, const std::vector<RegionInterval>& occ)
{
	for(unsigned int dilate = 0; dilate <= dilate_grow; dilate+=4)
	{
		cregion.unprocessed = true;
		std::vector<RegionInterval> actual_pixel_idx = getDilatedRegion(cregion, dilate, base);

		int length = getSizeOfRegion(actual_pixel_idx);

		cost_type cost_agg(length);

		cregion.disparity_costs = cv::Mat(dispMax-dispMin+1, 1, CV_32FC1, cv::Scalar(cost_agg.normalizationValue()));
		cregion.unprocessed = cv::Mat_<unsigned char>(dispMax-dispMin+1, 1, 1);

		float minCost = std::numeric_limits<float>::max();
		short minD = dispMin;
		for(int d = dispMin; d <= dispMax; ++d)
		{
			std::vector<RegionInterval> filtered_pixel_idx = getFilteredPixelIdx(match.size[1], actual_pixel_idx, d);

			std::vector<RegionInterval> *actual_filtered = &filtered_pixel_idx;
			std::vector<RegionInterval> occ_filtered;
			if(!occ.empty())
			{
				intervals::difference(filtered_pixel_idx.begin(), filtered_pixel_idx.end(), occ.begin(), occ.end(), d, std::back_inserter(occ_filtered));
				actual_filtered = &occ_filtered;
			}
			int filtered_length = getSizeOfRegion(*actual_filtered);

			if((float)filtered_length/(float)length > 0.6f && filtered_length > 10) //0.4
			{
				cregion.unprocessed(d-dispMin) = 0;
				cv::Mat base_region  = getRegionAsMat(base,  *actual_filtered, 0);
				cv::Mat match_region = getRegionAsMat(match, *actual_filtered, d);
				assert(base_region.total() == match_region.total());

				float cost = cost_agg(base_region, match_region);

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
		cregion.dilation = dilate;

		//analyze for further dilation rounds
		int len = cregion.disparity_costs.size[0];
		float *derived = new float[len-1];
		const float *costs = cregion.disparity_costs[0];
		derivePartialCostmap(costs, derived, len);
		analyzeDisparityRange(cregion.stats, costs, derived, len);

		if((cregion.stats.minima.size() < 2) && !(cregion.stats.minima.size() == 0 && cregion.stats.bad_minima.size() > 2))
			return;
	}
}*/

#endif // REGION_METRICS_H
