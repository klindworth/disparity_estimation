#include "ml_region_optimizer.h"

#include <omp.h>
#include "region.h"
#include "segmentation/intervals.h"
#include "segmentation/intervals_algorithms.h"
#include "disparity_utils.h"
#include "genericfunctions.h"

void refresh_base_optimization_vector_internal(std::vector<std::vector<float>>& optimization_vectors, const RegionContainer& base, const RegionContainer& match, int delta)
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

	optimization_vectors.resize(regions_count);
	#pragma omp parallel for
	for(std::size_t i = 0; i < regions_count; ++i)
	{
		const DisparityRegion& baseRegion = base.regions[i];
		int thread_idx = omp_get_thread_num();
		auto range = getSubrange(baseRegion.base_disparity, delta, base.task);

		intervals::substractRegionValue<unsigned char>(occmaps[thread_idx], baseRegion.warped_interval, 1);
		hyp_vec[thread_idx](occmaps[thread_idx], baseRegion, pot_trunc, dispMin, range.first, range.second, optimization_vectors[i]);
		intervals::addRegionValue<unsigned char>(occmaps[thread_idx], baseRegion.warped_interval, 1);
	}
}

void ml_region_optimizer::refresh_base_optimization_vector(const RegionContainer& left, const RegionContainer& right, int delta)
{
	refresh_base_optimization_vector_internal(optimization_vectors_left, left, right, delta);
	refresh_base_optimization_vector_internal(optimization_vectors_right, right, left, delta);
}



void ml_region_optimizer::gather_region_optimization_vector(float *dst_ptr, const DisparityRegion& baseRegion, const std::vector<float>& optimization_vector_base, const std::vector<std::vector<float>>& optimization_vectors_match, const RegionContainer& match, int delta, const StereoSingleTask& task, const std::vector<float>& normalization_vector)
{
	const int vector_size = ml_region_optimizer::vector_size;
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
			*dst_ptr++ = optimization_vector_base[offset+j] * normalization_vector[j];
		for(int j = vector_size; j < vector_size*2; ++j)
			*dst_ptr++ = other_optimization_vector[offset+j-vector_size] * normalization_vector[j];
	}
}

void ml_region_optimizer::optimize_ml(RegionContainer& base, RegionContainer& match, std::vector<std::vector<float>>& optimization_vectors_base, std::vector<std::vector<float>>& optimization_vectors_match, int delta)
{
	const int vector_size = ml_region_optimizer::vector_size*2;
	std::cout << "base" << std::endl;
	refresh_base_optimization_vector(base, match, delta);
	//refresh_optimization_vector(base, match, base_eval, delta);
	//refresh_optimization_vector(match, base, base_eval, delta);
	std::cout << "optimize" << std::endl;

	const int crange = base.task.dispMax - base.task.dispMin + 1;

	const std::size_t regions_count = base.regions.size();
	std::vector<float> normalization_vector(vector_size*2,1.0f);
	#pragma omp parallel for
	for(std::size_t j = 0; j < regions_count; ++j)
	{
		std::vector<float> region_optimization_vector(crange*vector_size*2); //recycle taskwise in prediction mode
		gather_region_optimization_vector(region_optimization_vector.data(), base.regions[j], optimization_vectors_base[j], optimization_vectors_match, match, delta, base.task, normalization_vector);
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

void ml_region_optimizer::prepare_training(std::vector<float>& dst, const std::vector<std::vector<float>>& base_optimization_vectors, const std::vector<std::vector<float>>& match_optimization_vectors, const RegionContainer& base, const RegionContainer& match, int delta)
{
	const int vector_size = ml_region_optimizer::vector_size *2;
	//std::cout << "base" << std::endl;
	//refresh_base_optimization_vector(base, match, delta);
	std::cout << "optimize" << std::endl;

	const int crange = base.task.dispMax - base.task.dispMin + 1;

	const std::size_t regions_count = base.regions.size();
	std::vector<float> normalization_vector(vector_size,1.0f);
	dst.resize(crange*vector_size*regions_count);
	float* dst_ptr = dst.data();

	std::vector<float> sums(vector_size, 0.0f); //per thread!!
	//#pragma omp parallel for default(none) shared(base, match, delta, normalization_vector)
	for(std::size_t j = 0; j < regions_count; ++j)
	{
		gather_region_optimization_vector(dst_ptr + j*crange*vector_size, base.regions[j], base_optimization_vectors[j], match_optimization_vectors, match, delta, base.task, normalization_vector);

		const float *src_ptr = dst_ptr + j*crange*vector_size;
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
	normalize_feature_vector(dst_ptr, regions_count*crange, sums);
}

void ml_region_optimizer::run(RegionContainer& left, RegionContainer& right, const optimizer_settings& /*config*/, int refinement)
{
	refresh_base_optimization_vector(left, right, refinement);
	if(training_mode)
	{
		samples_left.emplace_back();
		prepare_training(samples_left.back(), optimization_vectors_left, optimization_vectors_right, left, right, refinement);
		samples_right.emplace_back();
		prepare_training(samples_right.back(), optimization_vectors_right, optimization_vectors_left, right, left, refinement);
	}
	else
	{
		optimize_ml(left, right, optimization_vectors_left, optimization_vectors_right, refinement);
		refresh_base_optimization_vector(left, right, refinement);
		optimize_ml(right, left, optimization_vectors_right, optimization_vectors_left, refinement);
	}
}

void ml_region_optimizer::reset(const RegionContainer& /*left*/, const RegionContainer& /*right*/)
{
	samples_left.clear();
	samples_right.clear();
}

void ml_region_optimizer::training()
{
	std::cout << "start actual training" << std::endl;
	//TODO: ground truth
	//cv::Mat_<float> samples
}
