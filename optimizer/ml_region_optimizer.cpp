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
#include "disparity_region.h"
#include "disparity_region_algorithms.h"
#include <segmentation/intervals.h>
#include <segmentation/intervals_algorithms.h>
#include "disparity_utils.h"
#include "genericfunctions.h"
#include "disparity_range.h"

#include <fstream>
#include <boost/lexical_cast.hpp>
#include <neural_network/network.h>

#include "debugmatstore.h"

using namespace neural_network;

void refresh_base_optimization_vector_internal(std::vector<std::vector<float>>& optimization_vectors, const region_container& base, const region_container& match, int delta)
{
	int pot_trunc = 15;

	std::vector<ml_feature_calculator> hyp_vec(omp_get_max_threads(), ml_feature_calculator(base, match));
	std::size_t regions_count = base.regions.size();

	optimization_vectors.resize(regions_count);
	#pragma omp parallel for
	for(std::size_t i = 0; i < regions_count; ++i)
	{
		const disparity_region& baseRegion = base.regions[i];
		int thread_idx = omp_get_thread_num();

		disparity_range drange = task_subrange(base.task, baseRegion.base_disparity, delta);
		hyp_vec[thread_idx](baseRegion, pot_trunc, drange, optimization_vectors[i]);
	}
}

template<typename region_type, typename InsertIterator>
void region_ground_truth(const std::vector<region_type>& regions, cv::Mat_<short> gt, InsertIterator it)
{
	for(std::size_t i = 0; i < regions.size(); ++i)
	{
		int sum = 0;
		int count = 0;

		intervals::foreach_region_point(regions[i].lineIntervals.begin(), regions[i].lineIntervals.end(), [&](cv::Point pt){
			short value = gt(pt);
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

	assert(std::distance(data.data(), ptr) == (int)data.size());
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

void ml_region_optimizer::refresh_base_optimization_vector(const region_container& left, const region_container& right, int delta)
{
	refresh_base_optimization_vector_internal(optimization_vectors_left, left, right, delta);
	refresh_base_optimization_vector_internal(optimization_vectors_right, right, left, delta);
}

template<typename dst_type, typename src_type>
void gather_region_optimization_vector(dst_type *dst_ptr, const disparity_region& baseRegion, const std::vector<src_type>& optimization_vector_base, const std::vector<std::vector<src_type>>& optimization_vectors_match, const region_container& match, int delta, const single_stereo_task& task)
{
	const int crange = task.range_size();
	disparity_range drange = task_subrange(task, baseRegion.base_disparity, delta);

	std::vector<dst_type> disp_optimization_vector(ml_region_optimizer::vector_size_per_disp);
	for(short d = drange.start(); d <= drange.end(); ++d)
	{
		std::fill(disp_optimization_vector.begin(), disp_optimization_vector.end(), 0.0f);
		const int corresponding_disp_idx = -d - match.task.dispMin;
		foreach_corresponding_region(baseRegion.corresponding_regions[d-task.dispMin], [&](std::size_t idx, float percent) {
			const src_type* it = &(optimization_vectors_match[idx][corresponding_disp_idx*ml_region_optimizer::vector_size_per_disp]);
			for(int i = 0; i < ml_region_optimizer::vector_size_per_disp; ++i)
				disp_optimization_vector[i] += percent * *it++;
		});

		const src_type *base_ptr = optimization_vector_base.data() + (d-drange.start())*ml_region_optimizer::vector_size_per_disp;
		const dst_type *other_ptr = disp_optimization_vector.data();

		dst_type *ndst_ptr = dst_ptr + (d-drange.start())*ml_region_optimizer::vector_size_per_disp*2;
		//float *ndst_ptr = dst_ptr + vector_size_per_disp*2*(int)std::abs(d);
		//dst_type *ndst_ptr = dst_ptr + ml_region_optimizer::vector_size_per_disp*2*(crange - 1 - (int)std::abs(d));

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

void ml_region_optimizer::optimize_ml(region_container& base, const region_container& match, std::vector<std::vector<float>>& optimization_vectors_base, std::vector<std::vector<float>>& optimization_vectors_match, int delta, const std::string& filename)
{
	std::cout << "base" << std::endl;

	const int crange = base.task.range_size();

	std::vector<double> mean_normalization_vector(normalizer_size,0.0f);
	std::vector<double> stddev_normalization_vector(normalizer_size, 0.0f);
	std::ifstream istream(filename);

	if(!istream.is_open())
		throw std::runtime_error("file not found: " + filename);

	for(auto& cval : mean_normalization_vector)
		istream >> cval;
	for(auto& cval : stddev_normalization_vector)
		istream >> cval;

	istream >> *nnet;

	std::cout << "optimize" << std::endl;

	const std::size_t regions_count = base.regions.size();

	short sign = (base.task.dispMin < 0) ? -1 : 1;

	#pragma omp parallel for
	for(std::size_t j = 0; j < regions_count; ++j)
	{
		std::vector<double> region_optimization_vector(crange*vector_size_per_disp*2+vector_size); //recycle taskwise in prediction mode
		gather_region_optimization_vector(region_optimization_vector.data(), base.regions[j], optimization_vectors_base[j], optimization_vectors_match, match, delta, base.task);
		normalize_feature_vector(region_optimization_vector, mean_normalization_vector, stddev_normalization_vector);
		base.regions[j].disparity = nnet->predict(region_optimization_vector.data()) * sign;
	}

	refresh_warped_regions(base);
	std::cout << "end optimize" << std::endl;
}

std::ostream& operator<<(std::ostream& stream, result_eps_calculator& res)
{
	res.print_to_stream(stream);
	return stream;
}

void ml_region_optimizer::prepare_training_sample(std::vector<short>& dst_gt, std::vector<std::vector<double>>& dst_data, const std::vector<std::vector<float>>& base_optimization_vectors, const std::vector<std::vector<float>>& match_optimization_vectors, const region_container& base, const region_container& match, int delta)
{
	dst_gt.reserve(dst_gt.size() + base.regions.size());
	std::vector<short> gt;
	gt.reserve(base.regions.size());
	region_ground_truth(base.regions, base.task.groundTruth, std::back_inserter(gt));

	const int crange = base.task.range_size();

	const std::size_t regions_count = base.regions.size();
	dst_data.reserve(dst_data.size() + base_optimization_vectors.size());

	assert(gt.size() == regions_count);
	//#pragma omp parallel for default(none) shared(base, match, delta, normalization_vector)
	result_eps_calculator diff_calc;
	for(std::size_t j = 0; j < regions_count; ++j)
	{
		if(gt[j] != 0)
		{
			dst_data.emplace_back(vector_size_per_disp*2*crange+vector_size);
			double *dst_ptr = dst_data.back().data();
			gather_region_optimization_vector(dst_ptr, base.regions[j], base_optimization_vectors[j], match_optimization_vectors, match, delta, base.task);

			dst_gt.push_back(gt[j]);

			//std::cout << base.regions[j].disparity << " vs " << gt[j] << std::endl;
			diff_calc(base.regions[j].disparity, gt[j]);
		}
	}
	std::cout << diff_calc << std::endl;
}



void ml_region_optimizer::run(region_container& left, region_container& right, const optimizer_settings& /*config*/, int refinement)
{
	refresh_base_optimization_vector(left, right, refinement);
	if(training_mode)
	{
		for(int i = 0; i < training_iteration; ++i)
		{
			optimize_ml(left, right, optimization_vectors_left, optimization_vectors_right, refinement, filename_left_prefix + std::to_string(i) + ".txt");
			optimize_ml(right, left, optimization_vectors_right, optimization_vectors_left, refinement, filename_right_prefix + std::to_string(i) + ".txt");
			refresh_base_optimization_vector(left, right, refinement);
		}

		prepare_training_sample(samples_gt_left, samples_left, optimization_vectors_left, optimization_vectors_right, left, right, refinement);
		prepare_training_sample(samples_gt_right, samples_right, optimization_vectors_right, optimization_vectors_left, right, left, refinement);
	}
	else
	{
		for(int i = 0; i <= training_iteration; ++i)
		{
			optimize_ml(left, right, optimization_vectors_left, optimization_vectors_right, refinement, filename_left_prefix + std::to_string(i) + ".txt");
			optimize_ml(right, left, optimization_vectors_right, optimization_vectors_left, refinement, filename_right_prefix + std::to_string(i) + ".txt");
			refresh_base_optimization_vector(left, right, refinement);
		}

		std::vector<short> gt;
		region_ground_truth(left.regions, left.task.groundTruth, std::back_inserter(gt));

		result_eps_calculator diff_calc;
		cv::Mat_<unsigned char> diff_image(left.image_size, 0);
		for(std::size_t i = 0; i < left.regions.size(); ++i)
		{
			if(gt[i] != 0)
			{
				diff_calc(gt[i], left.regions[i].disparity);
				total_diff_calc(gt[i], left.regions[i].disparity);

				intervals::set_region_value<unsigned char>(diff_image, left.regions[i].lineIntervals, std::abs(std::abs(gt[i]) - std::abs(left.regions[i].disparity)));
			}
		}

		matstore.add_mat(diff_image, "gt_diff");


		std::cout << diff_calc << std::endl;
		std::cout << total_diff_calc << std::endl;
	}
}

void ml_region_optimizer::reset_internal()
{
	samples_left.clear();
	samples_right.clear();

	const int crange = 164;
	int dims = crange * vector_size_per_disp*2+vector_size;
	//int nvector = ml_region_optimizer::vector_size_per_disp + ml_region_optimizer::vector_size;
	int nvector = ml_region_optimizer::vector_size_per_disp;
	int pass = ml_region_optimizer::vector_size;
	nnet = std::unique_ptr<network<double>>(new network<double>(dims));
	//nnet->emplace_layer<vector_extension_layer>(ml_region_optimizer::vector_size_per_disp, ml_region_optimizer::vector_size);
	//nnet->emplace_layer<vector_connected_layer>(nvector*2, nvector*2, pass);
	//nnet->emplace_layer<relu_layer>();
	nnet->emplace_layer<vector_connected_layer>(nvector*2, nvector*2, pass);
	nnet->emplace_layer<relu_layer>();
	nnet->emplace_layer<transpose_vector_connected_layer>(4, nvector*2, pass);
	nnet->emplace_layer<relu_layer>();
	//nnet->emplace_layer<row_connected_layer>(crange, crange, pass);
	//nnet->emplace_layer<relu_layer>();
	nnet->emplace_layer<fully_connected_layer>(crange);
	nnet->emplace_layer<relu_layer>();
	nnet->emplace_layer<fully_connected_layer>(crange);
	nnet->emplace_layer<softmax_output_layer>();

	/*nnet->emplace_layer<vector_connected_layer>(ml_region_optimizer::vector_size_per_disp, ml_region_optimizer::vector_size_per_disp, ml_region_optimizer::vector_size);
	nnet->emplace_layer<relu_layer>();
	nnet->emplace_layer<vector_connected_layer>(ml_region_optimizer::vector_size_per_disp*2, ml_region_optimizer::vector_size_per_disp*2, ml_region_optimizer::vector_size);
	nnet->emplace_layer<relu_layer>();
	nnet->emplace_layer<transpose_vector_connected_layer>(4, ml_region_optimizer::vector_size_per_disp*2, ml_region_optimizer::vector_size);
	nnet->emplace_layer<relu_layer>();
	nnet->emplace_layer<row_connected_layer>(crange, crange, ml_region_optimizer::vector_size);
	nnet->emplace_layer<relu_layer>();
	//nnet->emplace_layer<fully_connected_layer>(crange);
	//nnet->emplace_layer<relu_layer>();
	nnet->emplace_layer<fully_connected_layer>(crange);
	nnet->emplace_layer<softmax_output_layer>();*/
}

ml_region_optimizer::ml_region_optimizer()
{
	nnet = nullptr;

	reset_internal();
	training_iteration = 0;
	filename_left_prefix = "weights-left-";
	filename_right_prefix = "weights-right-";
}

ml_region_optimizer::~ml_region_optimizer()
{
}

void ml_region_optimizer::reset(const region_container& /*left*/, const region_container& /*right*/)
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

void training_internal(std::vector<std::vector<double>>& samples, std::vector<short>& samples_gt, const std::string& filename, int iteration)
{
	int crange = 164;

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

	std::mt19937 rng;
	std::uniform_int_distribution<> dist(0, samples.size() - 1);
	for(std::size_t i = 0; i < samples.size(); ++i)
	{
		std::size_t exchange_idx = dist(rng);
		std::swap(samples[i], samples[exchange_idx]);
		std::swap(samples_gt[i], samples_gt[exchange_idx]);
	}

	/*std::vector<unsigned int> stats(crange, 0);
	for(std::size_t i = 0; i < samples_gt.size(); ++i)
		++(stats[std::abs(samples_gt[i])]);
	for(std::size_t i = 0; i < stats.size(); ++i)
		std::cout << "[" << i << "] " << (float)stats[i]/samples_gt.size() << ", " << (float)stats[i]/samples_gt.size()/(1.0/crange) << "\n";
	std::cout << std::endl;*/

	//TODO: class statistics?

	std::cout << "ann" << std::endl;
	//neural_network<double> net (dims, crange, {dims, dims});
	assert(dims == (ml_region_optimizer::vector_size_per_disp*2*crange)+ml_region_optimizer::vector_size);
	network<double> net(dims);

	/*if(iteration > 0)
	{
		//load old weights as init?
		std::ifstream istream(filename);

		/*if(!istream.is_open())
			throw std::runtime_error("file not found: " + filename);

		for(auto& cval : mean_normalization_vector)
			istream >> cval;
		for(auto& cval : stddev_normalization_vector)
			istream >> cval;
	}*/

	//int nvector = ml_region_optimizer::vector_size_per_disp + ml_region_optimizer::vector_size;
	int nvector = ml_region_optimizer::vector_size_per_disp;
	int pass = ml_region_optimizer::vector_size;
	//net.emplace_layer<vector_extension_layer>(ml_region_optimizer::vector_size_per_disp, ml_region_optimizer::vector_size);
	//net.emplace_layer<vector_connected_layer>(nvector*2, nvector*2, pass);
	//net.emplace_layer<relu_layer>();
	net.emplace_layer<vector_connected_layer>(nvector*2, nvector*2, pass);
	net.emplace_layer<relu_layer>();
	net.emplace_layer<transpose_vector_connected_layer>(4, nvector*2, pass);
	net.emplace_layer<relu_layer>();
	//net.emplace_layer<row_connected_layer>(crange, crange, pass);
	//net.emplace_layer<relu_layer>();
	net.emplace_layer<fully_connected_layer>(crange);
	net.emplace_layer<relu_layer>();
	net.emplace_layer<fully_connected_layer>(crange);
	net.emplace_layer<softmax_output_layer>();

	net.training(samples, samples_gt, 64, 61, 4);

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
	training_internal(samples_left, samples_gt_left, filename_left_prefix + std::to_string(training_iteration) + ".txt", training_iteration);
	training_internal(samples_right, samples_gt_right, filename_right_prefix + std::to_string(training_iteration) + ".txt", training_iteration);
}

void ml_feature_calculator::update_result_vector(std::vector<float>& result_vector, const disparity_region& baseRegion, const disparity_range& drange)
{
	const int range = drange.size();
	const int dispMin = drange.offset();

	neighbor_values neigh = get_neighbor_values(baseRegion, drange);

	short left_neighbor_disp  = neigh.left.disparity;
	short right_neighbor_disp = neigh.right.disparity;
	float left_color_dev = neigh.left.color_dev;
	float right_color_dev = neigh.right.color_dev;

	//	float costs, occ_avg, neighbor_pot, lr_pot ,neighbor_color_pot;
	result_vector.resize(range*ml_region_optimizer::vector_size_per_disp+ml_region_optimizer::vector_size);
	float org_size = baseRegion.size();
	float *result_ptr = result_vector.data();
	for(int i = 0; i < range; ++i)
	{
		*result_ptr++ = cost_values[i];
		*result_ptr++ = occ_avg_values[i];
		*result_ptr++ = neighbor_pot_values[i];
		*result_ptr++ = lr_pot_values[i];
		*result_ptr++ = neighbor_color_pot_values[i];
		*result_ptr++ = (float)occ_temp[i].first / org_size;
		//*result_ptr++ = rel_cost_values[i];
		int hyp_disp = dispMin + i;
		*result_ptr++ = left_neighbor_disp - hyp_disp;
		*result_ptr++ = right_neighbor_disp - hyp_disp;
		//*result_ptr++ = top_neighbor_disp - hyp_disp;
		//*result_ptr++ = bottom_neighbor_disp - hyp_disp;
		*result_ptr++ = warp_costs_values[i];
	}
	//*result_ptr = baseRegion.disparity;
	*result_ptr++ = *std::min_element(cost_values.begin(), cost_values.end());
	*result_ptr++ = left_color_dev;
	*result_ptr++ = right_color_dev;
	//*result_ptr++ = top_color_dev;
	//*result_ptr++ = bottom_color_dev;
}

float create_min_version(std::vector<float>::iterator start, std::vector<float>::iterator end, std::vector<float>::iterator ins)
{
	float min_value = *(std::min_element(start, end));

	std::transform(start, end, ins, [min_value](float val){
		return val - min_value;
	});

	return min_value;
}

void ml_feature_calculator::operator()(const disparity_region& baseRegion, short pot_trunc, const disparity_range& drange, std::vector<float>& result_vector)
{
	const int range = drange.size();

	cost_values.resize(range);
	rel_cost_values.resize(range);

	update_warp_costs(baseRegion, drange);
	update_occ_avg(baseRegion, pot_trunc, drange);
	update_average_neighbor_values(baseRegion, pot_trunc, drange);
	update_lr_pot(baseRegion, pot_trunc, drange);

	for(int i = 0; i < range; ++i)
		cost_values[i] = baseRegion.disparity_costs((drange.start()+i)-baseRegion.disparity_offset);

	create_min_version(cost_values.begin(), cost_values.end(), rel_cost_values.begin());

	update_result_vector(result_vector, baseRegion, drange);
}
