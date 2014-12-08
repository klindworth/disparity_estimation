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

template<typename T, typename lambda_type>
void gather_statistic(const std::vector<T>& data, std::vector<T>& sums, int& count, lambda_type func)
{
	int crange = (data.size() - ml_region_optimizer::vector_size)/ ml_region_optimizer::vector_size_per_disp;
	const T* ptr = data.data();
	for(int k = 0; k < crange; ++k)
	{
		for(int i = 0; i < ml_region_optimizer::vector_size_per_disp; ++i)
			sums[i] += func(*ptr++);
		++count;
	}
	for(int k = 0; k < ml_region_optimizer::vector_size; ++k)
		sums[ml_region_optimizer::vector_size_per_disp+k] += func(*ptr++);

	assert(std::distance(data.data(), ptr) == data.size());
}

template<typename T, typename lambda_type>
void gather_statistic(const std::vector<std::vector<T>>& data, std::vector<T>& sums, int& count, lambda_type func)
{
	assert(sums.size() == ml_region_optimizer::normalizer_size);
	std::fill(sums.begin(), sums.end(), 0);
	count = 0;

	for(const std::vector<T>& cdata :data)
		gather_statistic(cdata, sums, count, func);
}

template<typename T>
void prepare_normalizer(std::vector<T>& sums, int count, std::size_t samples)
{
	for(int i = 0; i < ml_region_optimizer::vector_size_per_disp; ++i)
		sums[i] /= count;
	for(int i = ml_region_optimizer::vector_size_per_disp; i < ml_region_optimizer::vector_size_per_disp+ml_region_optimizer::vector_size; ++i)
		sums[i] /= samples;
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

void ml_region_optimizer::refresh_base_optimization_vector(const RegionContainer& left, const RegionContainer& right, int delta)
{
	refresh_base_optimization_vector_internal(optimization_vectors_left, left, right, delta);
	refresh_base_optimization_vector_internal(optimization_vectors_right, right, left, delta);
}

template<typename dst_type, typename src_type>
void gather_region_optimization_vector(dst_type *dst_ptr, const DisparityRegion& baseRegion, const std::vector<src_type>& optimization_vector_base, const std::vector<std::vector<src_type>>& optimization_vectors_match, const RegionContainer& match, int delta, const StereoSingleTask& task)
{
	const int crange = task.range_size();
	auto range = getSubrange(baseRegion.base_disparity, delta, task);

	std::vector<dst_type> disp_optimization_vector(ml_region_optimizer::vector_size_per_disp);
	for(short d = range.first; d <= range.second; ++d)
	{
		std::fill(disp_optimization_vector.begin(), disp_optimization_vector.end(), 0.0f);
		const int corresponding_disp_idx = -d - match.task.dispMin;
		foreach_corresponding_region(baseRegion.other_regions[d-task.dispMin], [&](std::size_t idx, float percent) {
			const src_type* it = &(optimization_vectors_match[idx][corresponding_disp_idx*ml_region_optimizer::vector_size_per_disp]);
			for(int i = 0; i < ml_region_optimizer::vector_size_per_disp; ++i)
				disp_optimization_vector[i] += percent * *it++;
		});

		const src_type *base_ptr = optimization_vector_base.data() + (d-range.first)*ml_region_optimizer::vector_size_per_disp;
		const dst_type *other_ptr = disp_optimization_vector.data();

		//float *ndst_ptr = dst_ptr + (d-range.first)*vector_size_per_disp*2;
		//float *ndst_ptr = dst_ptr + vector_size_per_disp*2*(int)std::abs(d);
		dst_type *ndst_ptr = dst_ptr + ml_region_optimizer::vector_size_per_disp*2*(crange - 1 - (int)std::abs(d));

		for(int j = 0; j < ml_region_optimizer::vector_size_per_disp; ++j)
			*ndst_ptr++ = *base_ptr++;

		for(int j = 0; j < ml_region_optimizer::vector_size_per_disp; ++j)
			*ndst_ptr++ = *other_ptr++;
	}

	const src_type *base_src_ptr = optimization_vector_base.data()+crange*ml_region_optimizer::vector_size_per_disp;

	dst_type *ndst_ptr = dst_ptr + crange*ml_region_optimizer::vector_size_per_disp*2;
	for(int i = 0; i < ml_region_optimizer::vector_size; ++i)
		*ndst_ptr++ = *base_src_ptr++;
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
	neural_network<double> net(dims);
	//net.emplace_layer<vector_connected_layer>(vector_size_per_disp, vector_size_per_disp, vector_size);
	//net.emplace_layer<relu_layer>();
	net.emplace_layer<vector_connected_layer>(2, vector_size_per_disp*2, vector_size);
	net.emplace_layer<relu_layer>();
	net.emplace_layer<fully_connected_layer>(crange);
	net.emplace_layer<softmax_output_layer>();

	std::vector<double> mean_normalization_vector(normalizer_size,0.0f);
	std::vector<double> stddev_normalization_vector(normalizer_size, 0.0f);
	std::ifstream istream("weights.txt");

	for(auto& cval : mean_normalization_vector)
		istream >> cval;
	for(auto& cval : stddev_normalization_vector)
		istream >> cval;

	istream >> net;

	std::cout << "optimize" << std::endl;

	const std::size_t regions_count = base.regions.size();

	short sign = (base.task.dispMin < 0) ? -1 : 1;

	#pragma omp parallel for
	for(std::size_t j = 0; j < regions_count; ++j)
	{
		std::vector<double> region_optimization_vector(crange*vector_size_per_disp*2+vector_size); //recycle taskwise in prediction mode
		gather_region_optimization_vector(region_optimization_vector.data(), base.regions[j], optimization_vectors_base[j], optimization_vectors_match, match, delta, base.task);
		normalize_feature_vector(region_optimization_vector, mean_normalization_vector, stddev_normalization_vector);
		base.regions[j].disparity = net.predict(region_optimization_vector.data()) * sign;
	}
}

void ml_region_optimizer::prepare_training_sample(std::vector<unsigned char>& dst_gt, std::vector<std::vector<double>>& dst_data, const std::vector<std::vector<float>>& base_optimization_vectors, const std::vector<std::vector<float>>& match_optimization_vectors, const RegionContainer& base, const RegionContainer& match, int delta)
{
	dst_gt.reserve(dst_gt.size() + base.regions.size());
	std::vector<unsigned char> gt;
	gt.reserve(base.regions.size());
	region_ground_truth(base.regions, base.task.groundTruth, std::back_inserter(gt));

	const int crange = base.task.range_size();

	const std::size_t regions_count = base.regions.size();
	dst_data.reserve(dst_data.size() + base_optimization_vectors.size());

	assert(gt.size() == regions_count);
	//#pragma omp parallel for default(none) shared(base, match, delta, normalization_vector)
	int base_correct = 0, approx5 = 0, approx10 = 0, total = 0;
	for(std::size_t j = 0; j < regions_count; ++j)
	{
		if(gt[j] != 0)
		{
			dst_data.emplace_back(vector_size_per_disp*2*crange+vector_size);
			double *dst_ptr = dst_data.back().data();
			gather_region_optimization_vector(dst_ptr, base.regions[j], base_optimization_vectors[j], match_optimization_vectors, match, delta, base.task);

			dst_gt.push_back(gt[j]);

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



void ml_region_optimizer::run(RegionContainer& left, RegionContainer& right, const optimizer_settings& /*config*/, int refinement)
{
	refresh_base_optimization_vector(left, right, refinement);
	if(training_mode)
	{
		prepare_training_sample(samples_gt_left, samples_left, optimization_vectors_left, optimization_vectors_right, left, right, refinement);
		prepare_training_sample(samples_gt_right, samples_right, optimization_vectors_right, optimization_vectors_left, right, left, refinement);
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
}

ml_region_optimizer::ml_region_optimizer()
{
	reset_internal();
}

void ml_region_optimizer::reset(const RegionContainer& /*left*/, const RegionContainer& /*right*/)
{
	reset_internal();
}

template<typename T>
void gather_normalizers(std::vector<std::vector<T>>& data, std::vector<T>& mean_normalizer, std::vector<T>& stddev_normalizer)
{
	mean_normalizer.resize(ml_region_optimizer::normalizer_size);
	stddev_normalizer.resize(ml_region_optimizer::normalizer_size);

	int mean_count = 0;
	gather_statistic(data, mean_normalizer, mean_count, [](T val) {return val;});
	prepare_normalizer(mean_normalizer, mean_count, data.size());

	int std_count = 0;
	gather_statistic(data, stddev_normalizer, std_count, [](T val) {return val*val;});
	prepare_normalizer(stddev_normalizer, std_count, data.size());

	for(auto& val : stddev_normalizer)
		val = 1.0 / std::sqrt(val);
}

void training_internal(std::vector<std::vector<double>>& samples, std::vector<unsigned char>& samples_gt, const std::string& filename)
{
	int crange = 256;

	std::cout << "start actual training" << std::endl;

	std::vector<double> mean_normalization_vector;
	std::vector<double> stddev_normalization_vector;
	gather_normalizers(samples, mean_normalization_vector, stddev_normalization_vector);

	//apply normalization
	for(auto& cvec : samples)
		normalize_feature_vector(cvec, mean_normalization_vector, stddev_normalization_vector);

	assert(samples.size() == samples_gt.size());

	int dims = samples.front().size();
	std::cout << "copy" << std::endl;

	std::vector<short> gt(samples_gt.size());
	std::copy(samples_gt.begin(), samples_gt.end(), gt.begin());

	std::mt19937 rng;
	std::uniform_int_distribution<> dist(0, samples.size() - 1);
	for(std::size_t i = 0; i < samples.size(); ++i)
	{
		std::size_t exchange_idx = dist(rng);
		std::swap(samples[i], samples[exchange_idx]);
		std::swap(gt[i], gt[exchange_idx]);
	}

	//TODO: class statistics?

	std::cout << "ann" << std::endl;
	//neural_network<double> net (dims, crange, {dims, dims});
	assert(dims == (ml_region_optimizer::vector_size_per_disp*2*crange)+ml_region_optimizer::vector_size);
	neural_network<double> net(dims);
	//net.emplace_layer<vector_connected_layer>(vector_size_per_disp, vector_size_per_disp, vector_size);
	//net.emplace_layer<relu_layer>();
	net.emplace_layer<vector_connected_layer>(2, ml_region_optimizer::vector_size_per_disp*2, ml_region_optimizer::vector_size);
	net.emplace_layer<relu_layer>();
	net.emplace_layer<fully_connected_layer>(crange);
	net.emplace_layer<softmax_output_layer>();

	for(int i = 0; i < 101; ++i)
	{
		std::cout << "epoch: " << i << std::endl;
		net.training(samples, gt, 64);
		if(i%4 == 0)
			net.test(samples, gt);
	}

	std::ofstream ostream(filename);
	ostream.precision(17);
	std::copy(mean_normalization_vector.begin(), mean_normalization_vector.end(), std::ostream_iterator<float>(ostream, " "));
	std::copy(stddev_normalization_vector.begin(), stddev_normalization_vector.end(), std::ostream_iterator<float>(ostream, " "));

	ostream << net;
	ostream.close();

	std::cout << "fin" << std::endl;
}

void ml_region_optimizer::training()
{
	training_internal(samples_left, samples_gt_left, "weights-left.txt");
	training_internal(samples_right, samples_gt_right, "weights-right.txt");
}
