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

#include "ml_region_optimizer_disp.h"
#include "ml_region_optimizer.h"

#include <omp.h>
#include "disparity_region.h"
#include "disparity_region_algorithms.h"
#include <segmentation/intervals.h>
#include <segmentation/intervals_algorithms.h>
#include "disparity_toolkit/disparity_utils.h"
#include "genericfunctions.h"
#include "disparity_toolkit/disparity_range.h"

#include <fstream>
#include <boost/lexical_cast.hpp>
#include <neural_network/network.h>
#include <neural_network/data_normalizer.h>
#include "costmap_utils.h"
#include "debugmatstore.h"
#include "ml_region_optimizer_algorithms.h"

using namespace neural_network;

void init_network_disp(network<double>& net, int /*crange*/, int nvector, int pass)
{
	//net.emplace_layer<vector_extension_layer>(ml_region_optimizer::vector_size_per_disp, ml_region_optimizer::vector_size);
	//net.emplace_layer<vector_connected_layer>(nvector*2, nvector*2, pass);
	//net.emplace_layer<relu_layer>();
	net.emplace_layer<vector_connected_layer>(nvector*2, nvector, pass);
	net.emplace_layer<relu_layer>();
	net.emplace_layer<vector_connected_layer>(nvector, nvector*2, pass);
	net.emplace_layer<relu_layer>();
	net.emplace_layer<fully_connected_layer>(nvector);
	net.emplace_layer<relu_layer>();
	net.emplace_layer<fully_connected_layer>(nvector);
	net.emplace_layer<relu_layer>();
	//net.emplace_layer<fully_connected_layer>(4);
	//net.emplace_layer<relu_layer>();
	net.emplace_layer<fully_connected_layer>(2);
	net.emplace_layer<softmax_output_layer>();
	//net.emplace_layer<tanh_output_layer>();
}

void ml_region_optimizer_disp::optimize_ml(region_container& base, const region_container& match, std::vector<std::vector<float>>& optimization_vectors_base, std::vector<std::vector<float>>& optimization_vectors_match, int delta, const std::string& filename)
{
	int crange = 164;
	std::cout << "optimize_dsip" << std::endl;

	std::ifstream istream(filename);

	if(!istream.is_open())
		throw std::runtime_error("file not found: " + filename);

	int sample_size = vector_size_per_disp*2+vector_size;

	network<double> net(sample_size);

	int nvector = ml_region_optimizer_disp::vector_size_per_disp;
	int pass = ml_region_optimizer_disp::vector_size;
	init_network_disp(net, sample_size, nvector, pass);

	data_normalizer<double> normalizer(istream);
	istream >> net;

	const std::size_t regions_count = base.regions.size();
	const short sign = (base.task.range.start() < 0) ? -1 : 1;

	std::vector<std::vector<double>> region_optimization_vectors(omp_get_max_threads(), std::vector<double>(base.task.range.size()*vector_size_per_disp*2+vector_size));
	//#pragma omp parallel for
	std::vector<double> temp(sample_size);
	for(std::size_t j = 0; j < regions_count; ++j)
	{
		std::vector<double>& optimization_vector = region_optimization_vectors[omp_get_thread_num()];
		merge_with_corresponding_optimization_vector<vector_size, vector_size_per_disp>(optimization_vector.data(), base.regions[j], optimization_vectors_base[j], optimization_vectors_match, match, delta, base.task);


		const double* temp_ptr = optimization_vector.data();

		double best_output = -1;
		int best_idx = -1;

		for(int i = 0; i < crange; ++i)
		{

			auto copy_sample = [&](int disp) {
				int per_disp = vector_size_per_disp*2;
				std::copy(temp_ptr + std::abs(disp)*per_disp, temp_ptr + (std::abs(disp)+1)*per_disp, temp.data());
				std::copy(temp_ptr + crange*per_disp, temp_ptr + crange*per_disp+vector_size, temp.data() + per_disp);
			};

			copy_sample(i);
			normalizer.apply(temp);
			double res = net.output(temp.data())[1];

			if(res > best_output)
			{
				best_idx = i;
				best_output = res;
			}
		}
		base.regions[j].disparity = best_idx * sign;
	}

	refresh_warped_regions(base);
}

void ml_region_optimizer_disp::prepare_training_sample(std::vector<short>& dst_gt, std::vector<std::vector<double>>& dst_data, const std::vector<std::vector<float>>& base_optimization_vectors, const std::vector<std::vector<float>>& match_optimization_vectors, const region_container& base, const region_container& match, int delta)
{
	dst_gt.reserve(dst_gt.size() + base.regions.size());
	std::vector<short> gt;
	gt.reserve(base.regions.size());
	region_ground_truth(base.regions, base.task.groundTruth, std::back_inserter(gt));

	const int crange = base.task.range.size();

	const std::size_t regions_count = base.regions.size();
	dst_data.reserve(dst_data.size() + base_optimization_vectors.size());

	int draw_eps = 4;
	std::mt19937 rng;
	std::uniform_int_distribution<> dist(0, crange - 2*draw_eps - 2);

	assert(gt.size() == regions_count);
	//#pragma omp parallel for default(none) shared(base, match, delta, normalization_vector)
	int hardcase_advance = 0;
	std::vector<double> temp_vector(vector_size_per_disp*2*crange+vector_size);
	result_eps_calculator diff_calc;
	for(std::size_t j = 0; j < regions_count; ++j)
	{
		if(gt[j] != 0 && gt[j] < crange)
		{
			int sample_size = vector_size_per_disp*2+vector_size;
			double *temp_ptr = temp_vector.data();
			merge_with_corresponding_optimization_vector<vector_size, vector_size_per_disp>(temp_ptr, base.regions[j], base_optimization_vectors[j], match_optimization_vectors, match, delta, base.task);

			auto copy_sample = [&](int disp, bool correct) {
				dst_data.emplace_back(sample_size);
				int per_disp = vector_size_per_disp*2;
				std::copy(temp_ptr + std::abs(disp)*per_disp, temp_ptr + (std::abs(disp)+1)*per_disp, dst_data.back().data());
				std::copy(temp_ptr + crange*per_disp, temp_ptr + crange*per_disp+vector_size, dst_data.back().data() + per_disp);

				dst_gt.push_back(correct ? 1 : 0);
			};

			auto draw_neg_sample = [&]() {
				int neg_disp_sample_idx = dist(rng);
				if(std::abs(gt[j]) - draw_eps < neg_disp_sample_idx)
					neg_disp_sample_idx += 2*draw_eps+1;

				assert(neg_disp_sample_idx >= 0 && neg_disp_sample_idx < crange);

				copy_sample(neg_disp_sample_idx, false);
			};

			//copy_sample(gt[j], true);

			//draw_neg_sample();

			if(std::abs(base.regions[j].disparity - gt[j]) > 5)
			{
				copy_sample(base.regions[j].disparity, false);
				copy_sample(gt[j], true);

				draw_neg_sample();
				draw_neg_sample();

				hardcase_advance++;
			}
			else if(hardcase_advance > 0)
			{
				draw_neg_sample();
				draw_neg_sample();
				draw_neg_sample();

				copy_sample(gt[j], true);

				hardcase_advance--;
			}
			//else
				//draw_neg_sample();


			//std::cout << base.regions[j].disparity << " vs " << gt[j] << std::endl;
			diff_calc(base.regions[j].disparity, gt[j]);
		}
	}
	std::cout << diff_calc << std::endl;
}

void ml_region_optimizer_disp::reset_internal()
{
	const int crange = 164;
	int dims = crange * vector_size_per_disp*2+vector_size;
	//int nvector = ml_region_optimizer::vector_size_per_disp + ml_region_optimizer::vector_size;
	int nvector = ml_region_optimizer_disp::vector_size_per_disp;
	int pass = ml_region_optimizer_disp::vector_size;
	nnet = std::unique_ptr<network<double>>(new network<double>(dims));
	init_network_disp(*nnet, crange, nvector, pass);
}

ml_region_optimizer_disp::ml_region_optimizer_disp()
{
	reset_internal();
	training_iteration = 0;
	filename_left_prefix = "weights-left-";
	filename_right_prefix = "weights-right-";
}

ml_region_optimizer_disp::~ml_region_optimizer_disp()
{
}

void ml_region_optimizer_disp::reset(const region_container& /*left*/, const region_container& /*right*/)
{
	reset_internal();
}

void training_internal_per_disp(std::vector<std::vector<double>>& samples, std::vector<short>& samples_gt, const std::string& filename)
{
	std::cout << "start disp actual training" << std::endl;
	std::cout << "samples: " << samples.size() << std::endl;

	data_normalizer<double> normalizer(ml_region_optimizer_disp::vector_size_per_disp*2+ml_region_optimizer_disp::vector_size, 0);
	normalizer.gather(samples);
	normalizer.apply(samples);

	randomize_dataset(samples, samples_gt);

	int dims = samples.front().size();
	network<double> net(dims);

	std::vector<unsigned int> stats(2, 0);
	for(std::size_t i = 0; i < samples_gt.size(); ++i)
		++(stats[std::abs(samples_gt[i])]);
	for(std::size_t i = 0; i < stats.size(); ++i)
		std::cout << "[" << i << "] " << (float)stats[i]/samples_gt.size() << "\n";
	std::cout << std::endl;

	//int nvector = ml_region_optimizer::vector_size_per_disp + ml_region_optimizer::vector_size;
	int nvector = ml_region_optimizer_disp::vector_size_per_disp;
	int pass = ml_region_optimizer_disp::vector_size;

	init_network_disp(net, dims, nvector, pass);
	net.multi_training(samples, samples_gt, 16, 61, 4);

	std::ofstream ostream(filename);
	ostream.precision(17);
	normalizer.write(ostream);

	ostream << net;
	ostream.close();

	std::cout << "disp_fin" << std::endl;
}

void ml_region_optimizer_disp::training()
{
	training_internal_per_disp(samples_left, samples_gt_left, filename_left_prefix + std::to_string(training_iteration) + "-disp.txt");
	std::cout << "\n\n ------------------------------------------- next side --------------------------------------------" << std::endl;
	training_internal_per_disp(samples_right, samples_gt_right, filename_right_prefix + std::to_string(training_iteration) + "-disp.txt");
}

