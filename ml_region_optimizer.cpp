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

#include <fstream>
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



void ml_region_optimizer::gather_region_optimization_vector(float *dst_ptr, const DisparityRegion& baseRegion, const std::vector<float>& optimization_vector_base, const std::vector<std::vector<float>>& optimization_vectors_match, const RegionContainer& match, int delta, const StereoSingleTask& task, const std::vector<float>& mean_normalization_vector, const std::vector<float>& stddev_normalization_vector)
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

	const float *base_src_ptr = optimization_vector_base.data();
	const float *other_src_ptr = other_optimization_vector.data();
	for(int i = 0; i < crange; ++i)
	{
		for(int j = 0; j < vector_size_per_disp; ++j)
		{
			float val = *base_src_ptr++ - mean_normalization_vector[j];
			val *= stddev_normalization_vector[j];
			*dst_ptr++ = val;
		}

		for(int j = 0; j< vector_size_per_disp; ++j)
		{
			float val = *other_src_ptr++ - mean_normalization_vector[j];
			val *= stddev_normalization_vector[j];
			*dst_ptr++ = val;
		}
	}
	for(int i = 0; i < vector_size; ++i)
	{
		float val = *base_src_ptr++ - mean_normalization_vector[vector_size_per_disp+i];
		val *= stddev_normalization_vector[vector_size_per_disp+i];
		*dst_ptr++ = val;
		//*dst_ptr++ = *other_src_ptr++ - normalization_vector[vector_size_per_disp+i];
	}
}

void ml_region_optimizer::optimize_ml(RegionContainer& base, RegionContainer& match, std::vector<std::vector<float>>& optimization_vectors_base, std::vector<std::vector<float>>& optimization_vectors_match, int delta)
{
	std::cout << "base" << std::endl;
	refresh_base_optimization_vector(base, match, delta);
	//refresh_optimization_vector(base, match, base_eval, delta);
	//refresh_optimization_vector(match, base, base_eval, delta);

	const int crange = base.task.range_size();
	int dims = crange * vector_size_per_disp*2+vector_size;

	std::cout << "ann" << std::endl;
	//neural_network<double> net (dims, crange, {dims, dims});
	neural_network<float> net(dims);
	//net.emplace_layer<vector_connected_layer>(vector_size_per_disp, vector_size_per_disp, vector_size);
	//net.emplace_layer<relu_layer>();
	net.emplace_layer<vector_connected_layer>(2, vector_size_per_disp*2, vector_size);
	net.emplace_layer<relu_layer>();
	net.emplace_layer<fully_connected_layer>(crange);
	net.emplace_layer<softmax_output_layer>();

	std::vector<float> mean_normalization_vector(normalizer_size,0.0f);
	std::vector<float> stddev_normalization_vector(normalizer_size, 0.0f);
	std::ifstream istream("weights.txt");

	for(auto& cval : mean_normalization_vector)
		istream >> cval;
	for(auto& cval : stddev_normalization_vector)
		istream >> cval;

	istream >> net;

	std::cout << "optimize" << std::endl;

	const std::size_t regions_count = base.regions.size();

	#pragma omp parallel for
	for(std::size_t j = 0; j < regions_count; ++j)
	{
		std::vector<float> region_optimization_vector(crange*vector_size_per_disp*2+vector_size); //recycle taskwise in prediction mode
		gather_region_optimization_vector(region_optimization_vector.data(), base.regions[j], optimization_vectors_base[j], optimization_vectors_match, match, delta, base.task, mean_normalization_vector, stddev_normalization_vector);
		//TODO: call predict function and save result

		//TODO: normalizer
		base.regions[j].disparity = net.predict(region_optimization_vector.data());
		std::cout << base.regions[j].disparity << std::endl;
	}
}

template<typename T>
void normalize_feature_vector(T *ptr, int n, const std::vector<T>& mean_normalization_vector, const std::vector<T>& stddev_normalization_vector)
{
	int cmax = (n - ml_region_optimizer::vector_size) / ml_region_optimizer::vector_size_per_disp;
	assert((n - ml_region_optimizer::vector_size) % (ml_region_optimizer::vector_size_per_disp) == 0);
	assert(mean_normalization_vector.size() == stddev_normalization_vector.size());
	for(int j = 0; j < cmax; ++j)
	{
		for(int i = 0; i < ml_region_optimizer::vector_size_per_disp; ++i)
		{
			*ptr -= mean_normalization_vector[i];
			*ptr++ *= stddev_normalization_vector[i];
		}
	}
	for(int j = 0; j < ml_region_optimizer::vector_size; ++j)
	{
		*ptr -= mean_normalization_vector[ml_region_optimizer::vector_size_per_disp+j];
		*ptr++ *= stddev_normalization_vector[ml_region_optimizer::vector_size_per_disp+j];
	}
}

template<typename T>
void normalize_feature_vector(std::vector<T>& data, const std::vector<T>& mean_normalization_vector, const std::vector<T>& stddev_normalization_vector)
{
	normalize_feature_vector(data.data(), data.size(), mean_normalization_vector, stddev_normalization_vector);
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
	std::vector<float> stddev_normalization_vector(normalizer_size, 1.0f);
	dst.reserve(dst.size() + base_optimization_vectors.size());

	assert(gt.size() == regions_count);
	//#pragma omp parallel for default(none) shared(base, match, delta, normalization_vector)
	int base_correct = 0, approx5 = 0, approx10 = 0, total = 0;
	for(std::size_t j = 0; j < regions_count; ++j)
	{
		if(gt[j] != 0)
		{
			dst.emplace_back(vector_size_per_disp*2*crange+vector_size);
			float *dst_ptr = dst.back().data();
			gather_region_optimization_vector(dst_ptr, base.regions[j], base_optimization_vectors[j], match_optimization_vectors, match, delta, base.task, normalization_vector, stddev_normalization_vector);

			samples_gt.push_back(gt[j]);

			int diff = std::abs(std::abs(base.regions[j].disparity) - gt[j]);
			if(diff == 0)
				++base_correct;
			if(diff < 5)
				++approx5;
			if(diff < 10)
				++approx10;
			++total;
		}
	}
	std::cout << "correct: " << (float)base_correct/total << ", approx5: " << (float)approx5/total << ", approx10: " << (float)approx10/total << std::endl;
}

void ml_region_optimizer::gather_mean(const std::vector<std::vector<float>>& data)
{
	std::fill(mean_sums.begin(), mean_sums.end(), 0);
	mean_count = 0;

	for(std::size_t j = 0; j < data.size(); ++j)
	{
		int crange = (data[j].size() - vector_size)/ vector_size_per_disp;
		const float* ptr = data[j].data();
		for(int k = 0; k < crange; ++k)
		{
			for(int i = 0; i < vector_size_per_disp; ++i)
				mean_sums[i] += *ptr++;
			++mean_count;
		}
		for(int k = 0; k < vector_size; ++k)
			mean_sums[vector_size_per_disp+k] += *ptr++;

		assert(std::distance(data[j].data(), ptr) == data[j].size());
	}
}

void ml_region_optimizer::gather_stddev(const std::vector<std::vector<float>>& data)
{
	std::fill(mean_sums.begin(), mean_sums.end(), 0);
	mean_count = 0;

	for(std::size_t j = 0; j < data.size(); ++j)
	{
		int crange = (data[j].size() - vector_size)/ vector_size_per_disp;
		const float* ptr = data[j].data();
		for(int k = 0; k < crange; ++k)
		{
			for(int i = 0; i < vector_size_per_disp; ++i)
			{
				mean_sums[i] += *ptr * *ptr;
				++ptr;
			}
			++mean_count;
		}
		for(int k = 0; k < vector_size; ++k)
		{
			mean_sums[vector_size_per_disp+k] += *ptr * *ptr;
			++ptr;
		}

		assert(std::distance(data[j].data(), ptr) == data[j].size());
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

	auto prepare_normalizer = [&](){
		//gather normalization
		for(int i = 0; i < vector_size_per_disp; ++i)
			mean_sums[i] /= mean_count;
		for(int i = vector_size_per_disp; i < vector_size_per_disp+vector_size; ++i)
			mean_sums[i] /= samples_left.size();
	};

	auto invert_normalizer = [&](){
		for(auto& val : mean_sums)
			val = 1.0 / std::sqrt(val);
	};

	gather_mean(samples_left);
	prepare_normalizer();
	std::vector<float> mean_normalization_vector(normalizer_size);
	std::copy(mean_sums.begin(), mean_sums.end(), mean_normalization_vector.begin());
	std::copy(mean_normalization_vector.begin(), mean_normalization_vector.end(), std::ostream_iterator<float>(std::cout, ", "));
	std::cout << std::endl;

	gather_stddev(samples_left);
	prepare_normalizer();
	std::copy(mean_sums.begin(), mean_sums.end(), std::ostream_iterator<float>(std::cout, ", "));
	invert_normalizer();
	std::vector<float> stddev_normalization_vector(mean_normalization_vector.size());
	std::copy(mean_sums.begin(), mean_sums.end(), stddev_normalization_vector.begin());
	std::cout << std::endl;

	//apply normalization
	for(auto& cvec : samples_left)
		normalize_feature_vector(cvec.data(), cvec.size(), mean_normalization_vector, stddev_normalization_vector);

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

	std::mt19937 rng;
	std::uniform_int_distribution<> dist(0, data.size() - 1);
	for(std::size_t i = 0; i < data.size(); ++i)
	{
		std::size_t exchange_idx = dist(rng);
		std::swap(data[i], data[exchange_idx]);
		std::swap(gt[i], gt[exchange_idx]);
	}

	//TODO: class statistics?

	std::cout << "ann" << std::endl;
	//neural_network<double> net (dims, crange, {dims, dims});
	neural_network<double> net(dims);
	//net.emplace_layer<vector_connected_layer>(vector_size_per_disp, vector_size_per_disp, vector_size);
	//net.emplace_layer<relu_layer>();
	net.emplace_layer<vector_connected_layer>(2, vector_size_per_disp*2, vector_size);
	net.emplace_layer<relu_layer>();
	net.emplace_layer<fully_connected_layer>(crange);
	net.emplace_layer<softmax_output_layer>();

	for(int i = 0; i < 161; ++i)
	{
		std::cout << "epoch: " << i << std::endl;
		net.training(data, gt, 64);
		if(i%4 == 0)
			net.test(data, gt);
	}

	std::ofstream ostream("weights.txt");
	ostream.precision(17);
	std::copy(mean_normalization_vector.begin(), mean_normalization_vector.end(), std::ostream_iterator<float>(ostream, " "));
	std::copy(stddev_normalization_vector.begin(), stddev_normalization_vector.end(), std::ostream_iterator<float>(ostream, " "));

	ostream << net;
	ostream.close();

	std::cout << "fin" << std::endl;
}
