/*
Copyright (c) 2014, Kai Klindworth
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

template<typename region_type, typename InsertIterator>
void region_ground_truth(const std::vector<region_type>& regions, cv::Mat_<unsigned char> gt, InsertIterator it)
{
	std::vector<unsigned char> averages(regions.size());
	for(std::size_t i = 0; i < regions.size(); ++i)
	{
		int sum = 0;
		int count = 0;

		intervals::foreach_region_point(regions[i].lineIntervals.begin(), regions[i].lineIntervals.end(), [&](cv::Point pt){
			unsigned char value = gt(pt);
			if(value != 0)
			{
				sum += value;
				++count;
			}
		});

		*it = count > 0 ? std::round(sum/count) : 0;
		++it;
	}
}

void ml_region_optimizer::refresh_base_optimization_vector(const RegionContainer& left, const RegionContainer& right, int delta)
{
	refresh_base_optimization_vector_internal(optimization_vectors_left, left, right, delta);
	refresh_base_optimization_vector_internal(optimization_vectors_right, right, left, delta);
}



void ml_region_optimizer::gather_region_optimization_vector(float *dst_ptr, const DisparityRegion& baseRegion, const std::vector<float>& optimization_vector_base, const std::vector<std::vector<float>>& optimization_vectors_match, const RegionContainer& match, int delta, const StereoSingleTask& task, const std::vector<float>& normalization_vector)
{
	const int crange = task.dispMax - task.dispMin + 1;
	auto range = getSubrange(baseRegion.base_disparity, delta, task);

	std::vector<float> other_optimization_vector(crange*vector_size_per_disp);
	std::vector<float> disp_optimization_vector(vector_size_per_disp);
	for(short d = range.first; d < range.second; ++d)
	{
		std::fill(disp_optimization_vector.begin(), disp_optimization_vector.end(), 0.0f);
		int corresponding_disp_idx = -d - match.task.dispMin;
		foreach_corresponding_region(baseRegion.other_regions[d-task.dispMin], [&](std::size_t idx, float percent) {
			const float* it = &(optimization_vectors_match[idx][corresponding_disp_idx*vector_size_per_disp]);
			for(int i = 0; i < vector_size_per_disp; ++i)
				disp_optimization_vector[i] += percent * *it++;
		});

		std::copy(disp_optimization_vector.begin(), disp_optimization_vector.end(), &(other_optimization_vector[(d-range.first)*vector_size_per_disp]));
	}

	for(int i = 0; i < crange; ++i)
	{
		int offset = i*vector_size_per_disp*2;
		for(int j = 0; j < vector_size_per_disp*2; ++j)
			*dst_ptr++ = optimization_vector_base[offset+j] * normalization_vector[j];
	}
	for(int i = 0; i < vector_size; ++i)
	{
		int idx = crange*vector_size_per_disp+i;
		*dst_ptr++ = optimization_vector_base[idx] * normalization_vector[vector_size_per_disp*2+i];
	}
}

void ml_region_optimizer::optimize_ml(RegionContainer& base, RegionContainer& match, std::vector<std::vector<float>>& optimization_vectors_base, std::vector<std::vector<float>>& optimization_vectors_match, int delta)
{
	std::cout << "base" << std::endl;
	refresh_base_optimization_vector(base, match, delta);
	//refresh_optimization_vector(base, match, base_eval, delta);
	//refresh_optimization_vector(match, base, base_eval, delta);
	std::cout << "optimize" << std::endl;

	const int crange = base.task.dispMax - base.task.dispMin + 1;

	const std::size_t regions_count = base.regions.size();
	std::vector<float> normalization_vector(normalizer_size,1.0f);
	#pragma omp parallel for
	for(std::size_t j = 0; j < regions_count; ++j)
	{
		std::vector<float> region_optimization_vector(crange*vector_size_per_disp*2); //recycle taskwise in prediction mode
		gather_region_optimization_vector(region_optimization_vector.data(), base.regions[j], optimization_vectors_base[j], optimization_vectors_match, match, delta, base.task, normalization_vector);
		//TODO: call predict function and save result
	}
}

void normalize_feature_vector(float *ptr, int n, const std::vector<float>& normalization_vector)
{
	int cmax = (n - ml_region_optimizer::vector_size) / ml_region_optimizer::vector_size_per_disp / 2;
	assert((n - ml_region_optimizer::vector_size) % (ml_region_optimizer::vector_size_per_disp * 2) == 0);
	for(int j = 0; j < cmax; ++j)
	{
		for(int i = 0; i < ml_region_optimizer::vector_size_per_disp*2; ++i)
			*ptr++ *= normalization_vector[i];
	}
}

void normalize_feature_vector(std::vector<float>& data, const std::vector<float>& normalization_vector)
{
	normalize_feature_vector(data.data(), data.size(), normalization_vector);
}

void ml_region_optimizer::prepare_training_sample(std::vector<float>& dst, const std::vector<std::vector<float>>& base_optimization_vectors, const std::vector<std::vector<float>>& match_optimization_vectors, const RegionContainer& base, const RegionContainer& match, int delta)
{
	const int vector_size = ml_region_optimizer::vector_size_per_disp *2;

	const int crange = base.task.dispMax - base.task.dispMin + 1;

	const std::size_t regions_count = base.regions.size();
	std::vector<float> normalization_vector(normalizer_size,1.0f);
	dst.resize(crange*vector_size*regions_count);
	float* dst_ptr = dst.data();

	std::vector<float> sums(normalizer_size, 0.0f); //per thread!!
	//#pragma omp parallel for default(none) shared(base, match, delta, normalization_vector)
	for(std::size_t j = 0; j < regions_count; ++j)
	{
		gather_region_optimization_vector(dst_ptr + j*crange*vector_size, base.regions[j], base_optimization_vectors[j], match_optimization_vectors, match, delta, base.task, normalization_vector);

		const float *src_ptr = dst_ptr + j*crange*vector_size;
		for(int k = 0; k < crange; ++k)
		{
			for(int i = 0; i < vector_size; ++i)
				sums[i] += *src_ptr++;
			++norm_count;
		}
	}
}

void ml_region_optimizer::run(RegionContainer& left, RegionContainer& right, const optimizer_settings& /*config*/, int refinement)
{
	refresh_base_optimization_vector(left, right, refinement);
	if(training_mode)
	{
		samples_left.emplace_back();
		prepare_training_sample(samples_left.back(), optimization_vectors_left, optimization_vectors_right, left, right, refinement);
		//samples_right.emplace_back();
		//prepare_training_sample(samples_right.back(), optimization_vectors_right, optimization_vectors_left, right, left, refinement);

		samples_gt.reserve(samples_gt.size() + left.regions.size());
		region_ground_truth(left.regions, left.task.groundTruth, std::back_inserter(samples_gt));
	}
	else
	{
		optimize_ml(left, right, optimization_vectors_left, optimization_vectors_right, refinement);
		refresh_base_optimization_vector(left, right, refinement);
		optimize_ml(right, left, optimization_vectors_right, optimization_vectors_left, refinement);
	}
}

void ml_region_optimizer::reset_internal()
{
	samples_left.clear();
	samples_right.clear();

	std::fill(norm_sums.begin(), norm_sums.end(), 0);
	norm_count = 0;
}

ml_region_optimizer::ml_region_optimizer()
{
	reset_internal();
}

void ml_region_optimizer::reset(const RegionContainer& /*left*/, const RegionContainer& /*right*/)
{
	reset_internal();
}

void ml_region_optimizer::training()
{
	int crange = 128;

	std::cout << "start actual training" << std::endl;

	//gather normalization
	float sum_normalizer = 1.0f / norm_count;
	for(int i = 0; i < vector_size_per_disp*2; ++i)
		norm_sums[i] *= sum_normalizer;
	for(int i = vector_size_per_disp*2; i < vector_size_per_disp*2+vector_size; ++i)
		norm_sums[i] /= samples_left.size();

	std::vector<float> normalization_vector(normalizer_size);
	std::copy(norm_sums.begin(), norm_sums.end(), normalization_vector.begin());


	//apply normalization
	for(auto& cvec : samples_left)
		normalize_feature_vector(cvec.data(), cvec.size(), normalization_vector);

	//clean invalid gt samples
	assert(samples_left.size() == samples_gt.size());

	std::size_t last_valid_idx = samples_left.size() -1;
	for(std::size_t i = 0; i <= last_valid_idx; ++i)
	{
		if(samples_gt[i] == 0)
		{
			std::swap(samples_left[i], samples_left[last_valid_idx]);
			samples_gt[i] = samples_gt[last_valid_idx];
			--last_valid_idx;
		}
	}
	samples_left.erase(samples_left.begin() + last_valid_idx + 1, samples_left.end());
	samples_gt.erase(samples_gt.begin() + last_valid_idx + 1, samples_gt.end());
	assert(samples_left.size() == samples_gt.size());

	//TODO: ground truth
	//cv::Mat_<float> samples
}
