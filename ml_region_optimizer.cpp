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

#include <opencv2/ml/ml.hpp>
#include "simple_nn.h"

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
	const int crange = task.range_size();
	auto range = getSubrange(baseRegion.base_disparity, delta, task);

	std::vector<float> other_optimization_vector(crange*vector_size_per_disp);
	std::vector<float> disp_optimization_vector(vector_size_per_disp);
	for(short d = range.first; d < range.second; ++d)
	{
		std::fill(disp_optimization_vector.begin(), disp_optimization_vector.end(), 0.0f);
		const int corresponding_disp_idx = -d - match.task.dispMin;
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
			*dst_ptr++ = optimization_vector_base[offset+j] - normalization_vector[j];
	}
	for(int i = 0; i < vector_size; ++i)
	{
		int idx = crange*vector_size_per_disp+i;
		*dst_ptr++ = optimization_vector_base[idx] - normalization_vector[vector_size_per_disp*2+i];
	}
}

void ml_region_optimizer::optimize_ml(RegionContainer& base, RegionContainer& match, std::vector<std::vector<float>>& optimization_vectors_base, std::vector<std::vector<float>>& optimization_vectors_match, int delta)
{
	std::cout << "base" << std::endl;
	refresh_base_optimization_vector(base, match, delta);
	//refresh_optimization_vector(base, match, base_eval, delta);
	//refresh_optimization_vector(match, base, base_eval, delta);
	std::cout << "optimize" << std::endl;

	const int crange = base.task.range_size();

	const std::size_t regions_count = base.regions.size();
	std::vector<float> normalization_vector(normalizer_size,0.0f);
	#pragma omp parallel for
	for(std::size_t j = 0; j < regions_count; ++j)
	{
		std::vector<float> region_optimization_vector(crange*vector_size_per_disp*2+vector_size); //recycle taskwise in prediction mode
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
			*ptr++ -= normalization_vector[i];
	}
}

void normalize_feature_vector(std::vector<float>& data, const std::vector<float>& normalization_vector)
{
	normalize_feature_vector(data.data(), data.size(), normalization_vector);
}

void ml_region_optimizer::prepare_training_sample(std::vector<std::vector<float>>& dst, const std::vector<std::vector<float>>& base_optimization_vectors, const std::vector<std::vector<float>>& match_optimization_vectors, const RegionContainer& base, const RegionContainer& match, int delta)
{
	samples_gt.reserve(samples_gt.size() + base.regions.size());
	std::vector<unsigned char> gt;
	gt.reserve(base.regions.size());
	region_ground_truth(base.regions, base.task.groundTruth, std::back_inserter(gt));

	const int crange = base.task.range_size();

	const std::size_t regions_count = base.regions.size();
	std::vector<float> normalization_vector(normalizer_size,0.0f);
	dst.reserve(dst.size() + base_optimization_vectors.size());

	assert(gt.size() == regions_count);
	std::vector<float> sums(normalizer_size, 0.0f); //per thread!!
	//#pragma omp parallel for default(none) shared(base, match, delta, normalization_vector)
	for(std::size_t j = 0; j < regions_count; ++j)
	{
		if(gt[j] != 0)
		{
			dst.emplace_back(vector_size_per_disp*2*crange+vector_size);
			float *dst_ptr = dst.back().data();
			gather_region_optimization_vector(dst_ptr, base.regions[j], base_optimization_vectors[j], match_optimization_vectors, match, delta, base.task, normalization_vector);

			for(int k = 0; k < crange; ++k)
			{
				for(int i = 0; i < vector_size_per_disp*2; ++i)
					sums[i] += *dst_ptr++;
				++mean_count;
			}

			samples_gt.push_back(gt[j]);
		}
	}
}

void ml_region_optimizer::run(RegionContainer& left, RegionContainer& right, const optimizer_settings& /*config*/, int refinement)
{
	refresh_base_optimization_vector(left, right, refinement);
	if(training_mode)
	{
		//samples_left.emplace_back();
		//std::size_t start_idx = samples_left.size();
		//samples_left.resize(samples_left.size() + left.regions.size() * (vector_size_per_disp*2+vector_size) * (left.task.dispMax - left.task.dispMin + 1));
		prepare_training_sample(samples_left, optimization_vectors_left, optimization_vectors_right, left, right, refinement);
		//samples_right.emplace_back();
		//prepare_training_sample(samples_right.back(), optimization_vectors_right, optimization_vectors_left, right, left, refinement);

		//samples_gt.reserve(samples_gt.size() + left.regions.size());
		//region_ground_truth(left.regions, left.task.groundTruth, std::back_inserter(samples_gt));
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

	std::fill(mean_sums.begin(), mean_sums.end(), 0);
	mean_count = 0;
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
	int crange = 256;

	std::cout << "start actual training" << std::endl;

	//gather normalization
	float sum_normalizer = 1.0f / mean_count;
	for(int i = 0; i < vector_size_per_disp*2; ++i)
		mean_sums[i] *= sum_normalizer;
	for(int i = vector_size_per_disp*2; i < vector_size_per_disp*2+vector_size; ++i)
		mean_sums[i] /= samples_left.size();

	std::vector<float> normalization_vector(normalizer_size);
	std::copy(mean_sums.begin(), mean_sums.end(), normalization_vector.begin());


	//apply normalization
	for(auto& cvec : samples_left)
		normalize_feature_vector(cvec.data(), cvec.size(), normalization_vector);

	assert(samples_left.size() == samples_gt.size());

	int dims = samples_left.front().size();
	std::cout << "copy" << std::endl;
	std::vector<std::vector<double>> data(samples_left.size());
	for(std::size_t i = 0; i < samples_left.size(); ++i)
	{
		std::vector<double> inner_data(samples_left[i].size());
		std::copy(samples_left[i].begin(), samples_left[i].end(), inner_data.begin());
		data[i] = std::move(inner_data);
	}

	std::vector<short> gt(samples_gt.size());
	std::copy(samples_gt.begin(), samples_gt.end(), gt.begin());

	std::cout << "ann" << std::endl;
	neural_network<double> net (dims, crange, {dims, dims});
	for(int i = 0; i < 17; ++i)
	{
		std::cout << "epoch: " << i << std::endl;
		net.training(data, gt, 32);
		if(i%4 == 0)
			net.test(data, gt);
	}

	std::cout << "fin" << std::endl;

	//TODO: ground truth
	/*std::cout << "copy" << std::endl;
	cv::Mat_<float> samples(samples_left.size(), samples_left.front().size());
	//cv::Mat_<float> gt(samples_gt.size(), 1);
	for(std::size_t i = 0; i < samples_left.size(); ++i)
		std::copy(samples_left[i].begin(), samples_left[i].end(), samples.ptr<float>(i, 0));

	cv::Mat_<float> gt(samples_left.size(), crange, 0.0f);
	for(std::size_t i = 0; i < samples_left.size(); ++i)
		gt(i,samples_gt[i]) = 1.0f;
	//std::copy(samples_gt.begin(), samples_gt.end(), gt.begin());

	std::cout << "ann" << std::endl;
	CvANN_MLP_TrainParams params;
	params.term_crit = cvTermCriteria( CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 100, 0.1 );

	CvANN_MLP ann;
	cv::Mat_<int> layers(4,1);
	int dims = samples_left.front().size();
	layers << dims, dims*3, dims*2, crange;
	std::cout << layers << std::endl;
	std::cout << samples.size() << std::endl;
	std::cout << gt.size() << std::endl;
	ann.create(layers);
	ann.train(samples, gt, cv::Mat(), cv::Mat(), params);

	std::cout << "fin" << std::endl;*/
}
