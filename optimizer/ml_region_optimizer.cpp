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
#include <neural_network/data_normalizer.h>
#include "costmap_utils.h"
#include "debugmatstore.h"
#include "ml_region_optimizer_algorithms.h"

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

void ml_region_optimizer_base::refresh_base_optimization_vector(const region_container& left, const region_container& right, int delta)
{
	refresh_base_optimization_vector_internal(optimization_vectors_left, left, right, delta);
	refresh_base_optimization_vector_internal(optimization_vectors_right, right, left, delta);
}

void ml_region_optimizer::optimize_ml(region_container& base, const region_container& match, std::vector<std::vector<float>>& optimization_vectors_base, std::vector<std::vector<float>>& optimization_vectors_match, int delta, const std::string& filename)
{
	std::cout << "optimize" << std::endl;

	std::ifstream istream(filename);

	if(!istream.is_open())
		throw std::runtime_error("file not found: " + filename);

	data_normalizer<double> normalizer(istream);
	istream >> *nnet;

	const std::size_t regions_count = base.regions.size();
	const short sign = (base.task.dispMin < 0) ? -1 : 1;

	std::vector<std::vector<double>> region_optimization_vectors(omp_get_max_threads(), std::vector<double>(base.task.range_size()*vector_size_per_disp*2+vector_size));
	#pragma omp parallel for
	for(std::size_t j = 0; j < regions_count; ++j)
	{
		std::vector<double>& optimization_vector = region_optimization_vectors[omp_get_thread_num()];
		merge_with_corresponding_optimization_vector<ml_region_optimizer::vector_size, ml_region_optimizer::vector_size_per_disp>(optimization_vector.data(), base.regions[j], optimization_vectors_base[j], optimization_vectors_match, match, delta, base.task);
		normalizer.apply(optimization_vector);
		base.regions[j].disparity = nnet->predict(optimization_vector) * sign;
	}

	refresh_warped_regions(base);
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
			merge_with_corresponding_optimization_vector<ml_region_optimizer::vector_size, ml_region_optimizer::vector_size_per_disp>(dst_ptr, base.regions[j], base_optimization_vectors[j], match_optimization_vectors, match, delta, base.task);

			dst_gt.push_back(gt[j]);

			//std::cout << base.regions[j].disparity << " vs " << gt[j] << std::endl;
			diff_calc(base.regions[j].disparity, gt[j]);
		}
	}
	std::cout << diff_calc << std::endl;
}

void ml_region_optimizer_base::run(region_container& left, region_container& right, const optimizer_settings& /*config*/, int refinement)
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

		result_eps_calculator diff_calc = get_region_comparision(left.regions, gt);
		matstore.add_mat(get_region_gt_error_image(left, gt), "gt_diff");

		total_diff_calc += diff_calc;

		std::cout << diff_calc << std::endl;
		std::cout << "total: " << total_diff_calc << std::endl;
	}
}

void init_network(network<double>& net, int crange, int nvector, int pass)
{
	//net.emplace_layer<vector_extension_layer>(ml_region_optimizer::vector_size_per_disp, ml_region_optimizer::vector_size);
	//net.emplace_layer<vector_connected_layer>(nvector*2, nvector*2, pass);
	//net.emplace_layer<relu_layer>();
	net.emplace_layer<vector_connected_layer>(nvector*2, nvector*2, pass);
	net.emplace_layer<relu_layer>();
	net.emplace_layer<transpose_vector_connected_layer>(4, nvector*2, pass);
	net.emplace_layer<relu_layer>();
	net.emplace_layer<row_connected_layer>(crange, crange, pass);
	net.emplace_layer<relu_layer>();
	//net.emplace_layer<fully_connected_layer>(crange);
	//net.emplace_layer<relu_layer>();
	net.emplace_layer<fully_connected_layer>(crange);
	net.emplace_layer<softmax_output_layer>();
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

	init_network(*nnet, crange, nvector, pass);
}

ml_region_optimizer::ml_region_optimizer()
{
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

void training_internal(std::vector<std::vector<double>>& samples, std::vector<short>& samples_gt, const std::string& filename)
{
	int crange = 164;

	std::cout << "start actual training" << std::endl;

	data_normalizer<double> normalizer(ml_region_optimizer::vector_size_per_disp, ml_region_optimizer::vector_size);
	normalizer.gather(samples);
	normalizer.apply(samples);

	randomize_dataset(samples, samples_gt);

	//class statistics
	/*std::vector<unsigned int> stats(crange, 0);
	for(std::size_t i = 0; i < samples_gt.size(); ++i)
		++(stats[std::abs(samples_gt[i])]);
	for(std::size_t i = 0; i < stats.size(); ++i)
		std::cout << "[" << i << "] " << (float)stats[i]/samples_gt.size() << ", " << (float)stats[i]/samples_gt.size()/(1.0/crange) << "\n";
	std::cout << std::endl;*/

	int dims = samples.front().size();
	assert(dims == (ml_region_optimizer::vector_size_per_disp*2*crange)+ml_region_optimizer::vector_size);
	network<double> net(dims);

	//int nvector = ml_region_optimizer::vector_size_per_disp + ml_region_optimizer::vector_size;
	int nvector = ml_region_optimizer::vector_size_per_disp;
	int pass = ml_region_optimizer::vector_size;

	init_network(net, crange, nvector, pass);

	net.training(samples, samples_gt, 64, 61, 4);

	std::ofstream ostream(filename);
	ostream.precision(17);
	normalizer.write(ostream);

	ostream << net;
	ostream.close();

	std::cout << "fin" << std::endl;
}

void ml_region_optimizer::training()
{
	training_internal(samples_left, samples_gt_left, filename_left_prefix + std::to_string(training_iteration) + ".txt");
	training_internal(samples_right, samples_gt_right, filename_right_prefix + std::to_string(training_iteration) + ".txt");
}

void ml_feature_calculator::update_result_vector(std::vector<float>& result_vector, const disparity_region& baseRegion, const disparity_range& drange)
{
	const int range = drange.size();
	const int dispMin = drange.offset();

	neighbor_values neigh = get_neighbor_values(baseRegion, drange);

	short left_neighbor_disp  = neigh.left.disparity;
	short right_neighbor_disp = neigh.right.disparity;
	short top_neighbor_disp = neigh.top.disparity;
	short bottom_neighbor_disp = neigh.bottom.disparity;
	float left_color_dev = neigh.left.color_dev;
	float right_color_dev = neigh.right.color_dev;
	float top_color_dev = neigh.top.color_dev;
	float bottom_color_dev = neigh.bottom.color_dev;

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
		int hyp_disp = std::abs(dispMin + i);
		*result_ptr++ = left_neighbor_disp - hyp_disp;
		*result_ptr++ = right_neighbor_disp - hyp_disp;
		*result_ptr++ = top_neighbor_disp - hyp_disp;
		*result_ptr++ = bottom_neighbor_disp - hyp_disp;
		*result_ptr++ = warp_costs_values[i];
	//}
	//*result_ptr = baseRegion.disparity;
	*result_ptr++ = *std::min_element(cost_values.begin(), cost_values.end());
	*result_ptr++ = left_color_dev;
	*result_ptr++ = right_color_dev;
	*result_ptr++ = top_color_dev;
	*result_ptr++ = bottom_color_dev;
	*result_ptr++ = *std::min_element(warp_costs_values.begin(), warp_costs_values.end());
	*result_ptr++ = *std::min_element(lr_pot_values.begin(), lr_pot_values.end());
	*result_ptr++ = *std::min_element(neighbor_color_pot_values.begin(), neighbor_color_pot_values.end());

	*result_ptr++ = stats.mean;
	*result_ptr++ = stats.stddev;
}
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
	generate_stats(baseRegion, stats);

	for(int i = 0; i < range; ++i)
		cost_values[i] = baseRegion.disparity_costs((drange.start()+i)-baseRegion.disparity_offset);

	create_min_version(cost_values.begin(), cost_values.end(), rel_cost_values.begin());

	update_result_vector(result_vector, baseRegion, drange);
}
