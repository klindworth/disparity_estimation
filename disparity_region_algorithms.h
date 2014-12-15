#ifndef DISPARITY_REGION_ALGORITHMS_H
#define DISPARITY_REGION_ALGORITHMS_H

#include "disparity_region.h"

template<typename T>
inline void parallel_region(std::vector<DisparityRegion>& regions, T func)
{
	parallel_region(regions.begin(), regions.end(), func);
}

template<typename lambda_type>
void foreach_filtered_interval_point(const RegionInterval& interval, int width, int d, lambda_type func)
{
	int lower = std::max(interval.lower+d, 0)-d;
	int upper = std::min(interval.upper+d, width)-d;

	for(int x = lower; x < upper; ++x)
		func(cv::Point(x, interval.y));
}

template<typename lambda_type>
void foreach_warped_interval_point(const RegionInterval& interval, int width, int d, lambda_type func)
{
	int lower = std::max(interval.lower+d, 0);
	int upper = std::min(interval.upper+d, width);

	for(int x = lower; x < upper; ++x)
		func(cv::Point(x, interval.y));
}

template<typename lambda_type>
void foreach_filtered_region_point(const std::vector<RegionInterval> &pixel_idx, int width, int d, lambda_type func)
{
	for(const RegionInterval& cinterval : pixel_idx)
		foreach_filtered_interval_point(cinterval, width, d, func);
}

template<typename lambda_type>
void foreach_warped_region_point(const std::vector<RegionInterval> &pixel_idx, int width, int d, lambda_type func)
{
	for(const RegionInterval& cinterval : pixel_idx)
		foreach_warped_interval_point(cinterval, width, d, func);
}

template<typename Iterator, typename lambda_type>
void foreach_warped_region_point(Iterator it, Iterator end, int width, int d, lambda_type func)
{
	//for(const RegionInterval& cinterval : pixel_idx)
	for(; it != end; ++it)
		foreach_warped_interval_point(*it, width, d, func);
}

template<typename lambda_type>
float corresponding_regions_average(const std::vector<DisparityRegion>& container, const std::vector<MutualRegion>& cdisp, lambda_type func)
{
	float result = 0.0f;
	for(const MutualRegion& cval : cdisp)
	{
		result += cval.percent * func(container[cval.index]);
	}
	return result;
}

template<typename lambda_type>
float corresponding_regions_average_by_index(const std::vector<MutualRegion>& cdisp, lambda_type func)
{
	float result = 0.0f;
	for(const MutualRegion& cval : cdisp)
	{
		result += cval.percent * func(cval.index);
	}
	return result;
}

template<typename lambda_type>
void foreach_corresponding_region(const std::vector<MutualRegion>& cdisp, lambda_type func)
{
	for(const MutualRegion& cval : cdisp)
		func(cval.index, cval.percent);
}

/*template<typename cache_type, typename lambda_type>
void gather_other_regions_values(std::vector<cache_type>& cache, const std::vector<DisparityRegion>& container, const std::vector<MutualRegion>& cdisp, lambda_type func)
{
	std::size_t nsize = cdisp.size();
	cache.resize(nsize);

	for(std::size_t i = 0; i < nsize; ++i)
		cache[i] = func(container[cdisp[i].index]);
}

inline void gather_other_regions_weights(std::vector<float>& weights, const std::vector<MutualRegion>& cdisp)
{
	std::size_t nsize = cdisp.size();
	weights.resize(nsize);

	for(std::size_t i = 0; i < nsize; ++i)
		weights[i] = cdisp[i].percent;
}*/

template<typename lambda_type>
float getNeighborhoodsAverage(const std::vector<DisparityRegion>& container, const std::vector<std::pair<std::size_t, std::size_t>>& neighbors, lambda_type func)
{
	return getNeighborhoodsAverage(container, neighbors, 0.0f, func);
}

template<typename lambda_type>
float getWeightedNeighborhoodsAverage(const std::vector<DisparityRegion>& container, const std::vector<std::pair<std::size_t, std::size_t>>& neighbors, lambda_type func)
{
	return getWeightedNeighborhoodsAverage(container, neighbors, 0.0f, func);
}

template<typename lambda_type>
std::pair<float,float> getColorWeightedNeighborhoodsAverage(const cv::Vec3d& base_color, double color_trunc, const std::vector<DisparityRegion>& container, const std::vector<std::pair<std::size_t, std::size_t>>& neighbors, lambda_type func)
{
	float result = 0.0f;
	float sum_weight = 0.0f;
	for(const auto& cpair : neighbors)
	{
		float diff = color_trunc - std::min(cv::norm(base_color - container[cpair.first].average_color), color_trunc);

		result += diff*func(container[cpair.first]);
		sum_weight += diff;
	}
	sum_weight = std::max(std::numeric_limits<float>::min(), sum_weight);
	return std::make_pair(result/sum_weight, sum_weight);
}

inline float gather_neighbor_color_weights(std::vector<float>& weights, const cv::Vec3d& base_color, double color_trunc, const std::vector<DisparityRegion>& container, const std::vector<std::pair<std::size_t, std::size_t>>& neighbors)
{
	std::size_t nsize = neighbors.size();
	weights.resize(nsize);

	float sum_weight = 0.0f;
	for(std::size_t i = 0; i < nsize; ++i)
	{
		float diff = color_trunc - std::min(cv::norm(base_color - container[neighbors[i].first].average_color), color_trunc);

		weights[i] = diff;
		sum_weight += diff;
	}

	return std::max(std::numeric_limits<float>::min(), sum_weight);
}

inline float gather_neighbor_color_weights_from_cache(std::vector<float>& weights, const cv::Vec3d& base_color, double color_trunc, const std::vector<cv::Vec3d>& color_cache, const std::vector<std::pair<std::size_t, std::size_t>>& neighbors)
{
	std::size_t nsize = neighbors.size();
	weights.resize(nsize);

	float sum_weight = 0.0f;
	for(std::size_t i = 0; i < nsize; ++i)
	{
		float diff = color_trunc - std::min(cv::norm(base_color - color_cache[neighbors[i].first]), color_trunc);

		weights[i] = diff;
		sum_weight += diff;
	}

	return std::max(std::numeric_limits<float>::min(), sum_weight);
}

#endif //DISPARITY_REGION_ALGORITHMS_H
