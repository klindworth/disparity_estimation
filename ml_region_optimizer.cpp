#include "ml_region_optimizer.h"

#include <omp.h>
#include "region.h"
#include "segmentation/intervals.h"
#include "segmentation/intervals_algorithms.h"
#include "disparity_utils.h"
#include "genericfunctions.h"

void refresh_optimization_vector(std::vector<std::vector<float>>& optimization_vectors, RegionContainer& base, const RegionContainer& match, int delta)
{
	cv::Mat disp = getDisparityBySegments(base);
	cv::Mat occmap = occlusionStat<short>(disp, 1.0);
	int pot_trunc = 10;

	const short dispMin = base.task.dispMin;

	std::vector<disparity_hypothesis_vector> hyp_vec(omp_get_max_threads(), disparity_hypothesis_vector(base.regions, match.regions));
	std::vector<cv::Mat_<unsigned char>> occmaps(omp_get_max_threads());
	for(std::size_t i = 0; i < occmaps.size(); ++i)
	{
		occmaps[i] = occmap.clone();
	}

	std::size_t regions_count = base.regions.size();


	#pragma omp parallel for
	for(std::size_t i = 0; i < regions_count; ++i)
	{
		DisparityRegion& baseRegion = base.regions[i];
		int thread_idx = omp_get_thread_num();
		//int thread_idx = 0;
		auto range = getSubrange(baseRegion.base_disparity, delta, base.task);

		intervals::substractRegionValue<unsigned char>(occmaps[thread_idx], baseRegion.warped_interval, 1);
		hyp_vec[thread_idx](occmaps[thread_idx], baseRegion, pot_trunc, dispMin, range.first, range.second, optimization_vectors[i]);
		intervals::addRegionValue<unsigned char>(occmaps[thread_idx], baseRegion.warped_interval, 1);
	}
}





/*void gather_region_optimization_vector(float *dst_ptr, std::vector<float>& optimization_vectors_base, std::vector<std::vector<float>>& optimization_vectors_match, const DisparityRegion& baseRegion, const RegionContainer& match, int delta, const StereoSingleTask& task, const std::vector<float>& normalization_vector)
{
	const int vector_size = 5;
	const int crange = task.dispMax - task.dispMin + 1;
	auto range = getSubrange(baseRegion.base_disparity, delta, task);

	std::vector<float> other_optimization_vector(crange*vector_size);
	std::vector<float> disp_optimization_vector(vector_size);
	for(short d = range.first; d < range.second; ++d)
	{
		std::fill(disp_optimization_vector.begin(), disp_optimization_vector.end(), 0.0f);
		int corresponding_disp_idx = -d - match.task.dispMin;
		foreach_corresponding_region(baseRegion.other_regions[d-task.dispMin], [&](std::size_t idx, float percent) {
			const float* it = &(optimization_vectors_match[idx][corresponding_disp_idx*vector_size]);
			for(int i = 0; i < vector_size; ++i)
				disp_optimization_vector[i] += percent * *it++;
		});

		std::copy(disp_optimization_vector.begin(), disp_optimization_vector.end(), &(other_optimization_vector[(d-range.first)*vector_size]));
	}

	for(int i = 0; i < crange; ++i)
	{
		int offset = i*vector_size;
		for(int j = 0; j <  vector_size; ++j)
			*dst_ptr++ = baseRegion.optimization_vector[offset+j] * normalization_vector[j];
		for(int j = vector_size; j < vector_size*2; ++j)
			*dst_ptr++ = other_optimization_vector[offset+j-vector_size] * normalization_vector[j];
	}
}*/

void optimize_ml(RegionContainer& base, RegionContainer& match, const disparity_hypothesis_weight_vector& base_eval, std::function<float(const DisparityRegion&, const RegionContainer&, const RegionContainer&, int)> prop_eval, int delta)
{
	const int vector_size = 10;
	std::cout << "base" << std::endl;
	//refresh_optimization_vector(base, match, base_eval, delta);
	//refresh_optimization_vector(match, base, base_eval, delta);
	std::cout << "optimize" << std::endl;

	const int crange = base.task.dispMax - base.task.dispMin + 1;

	const std::size_t regions_count = base.regions.size();
	std::vector<float> normalization_vector(vector_size,1.0f);
	#pragma omp parallel for default(none) shared(base, match, delta, normalization_vector)
	for(std::size_t j = 0; j < regions_count; ++j)
	{
		std::vector<float> region_optimization_vector(crange*vector_size); //recycle taskwise in prediction mode
		//gather_region_optimization_vector(region_optimization_vector.data(), base.regions[j], match, delta, base.task, normalization_vector);

		//TODO: call predict function and save result
	}
}



void normalize_feature_vector(float *ptr, int n, const std::vector<float>& normalization_vector)
{
	int vector_size = normalization_vector.size();
	for(int j = 0; j < n; ++j)
	{
		for(int i = 0; i < vector_size; ++i)
			*ptr = *ptr * normalization_vector[i];
	}
}

/*void train_ml_optimizer(RegionContainer& base, RegionContainer& match, const disparity_hypothesis_weight_vector& base_eval, int delta)
{
	const int vector_size = 10;

	std::cout << "base" << std::endl;
	//refreshOptimizationBaseValues(base, match, base_eval, delta);
	//refreshOptimizationBaseValues(match, base, base_eval, delta);
	std::cout << "optimize" << std::endl;

	const int crange = base.task.dispMax - base.task.dispMin + 1;

	const std::size_t regions_count = base.regions.size();
	std::vector<float> normalization_vector(vector_size,1.0f);
	std::vector<float> featurevector(crange*vector_size*regions_count);

	std::vector<float> sums(vector_size, 0.0f); //per thread!!
	//#pragma omp parallel for default(none) shared(base, match, delta, normalization_vector)
	for(std::size_t j = 0; j < regions_count; ++j)
	{
		gather_region_optimization_vector(featurevector.data() + j*crange*vector_size, base.regions[j], match, delta, base.task, normalization_vector);

		const float *src_ptr = featurevector.data() + j*crange*vector_size;
		for(int k = 0; k < crange; ++k)
		{
			for(int i = 0; i < vector_size; ++i)
				sums[i] += *src_ptr++;
		}
	}

	//gather normalization
	float sum_normalizer = regions_count * crange;
	for(int i = 0; i < vector_size; ++i)
		sums[i] = sum_normalizer / sums[i];

	//apply normalization
	normalize_feature_vector(featurevector.data(), regions_count*crange, sums);

	//TODO: call training function
}*/


