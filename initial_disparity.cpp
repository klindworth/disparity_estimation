/*
Copyright (c) 2013, Kai Klindworth
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

#include "initial_disparity.h"

#include <iterator>
#include <iostream>
#include <omp.h>

#include "debugmatstore.h"

#include "disparity_region.h"
#include "sliding_entropy.h"
#include "it_metrics.h"
#include "costmap_creators.h"
#include "disparity_utils.h"
#include <segmentation/intervals.h>
#include <segmentation/intervals_algorithms.h>
#include <segmentation/segmentation.h>
#include <segmentation/region_descriptor.h>
#include <segmentation/region_descriptor_algorithms.h>
#include "region_optimizer.h"
#include "configrun.h"
#include "misc.h"

#include "region_metrics.h"
#include "refinement.h"

#include "disparitywise_calculator.h"
#include "sncc_disparitywise_calculator.h"
#include "disparity_region_algorithms.h"

class sliding_sad_threaddata
{
public:
	cv::Mat m_base;
	int cwindowsizeX;
	int cwindowsizeY;
	int crow;
};

class sliding_sad
{
public:
	typedef float prob_table_type;
	typedef sliding_sad_threaddata thread_type;
private:

	cv::Mat m_match;

public:
	inline sliding_sad(const cv::Mat& match, unsigned int /*max_windowsize*/) : m_match(match)
	{
	}

	//prepares a row for calculation
	inline void prepare_row(thread_type& thread, const cv::Mat& /*match*/, int y)
	{
		thread.crow = y;
	}

	inline void prepare_window(thread_type& thread, const cv::Mat& base, int cwindowsizeX, int cwindowsizeY)
	{
		thread.cwindowsizeX = cwindowsizeX;
		thread.cwindowsizeY = cwindowsizeY;
		//copy the window for L1 Cache friendlieness
		thread.m_base = base.clone();
	}

	inline prob_table_type increm(thread_type& thread, int x)
	{
		cv::Mat match_window = subwindow(m_match, x, thread.crow, thread.cwindowsizeX, thread.cwindowsizeY).clone();

		return cv::norm(thread.m_base, match_window, cv::NORM_L1)/match_window.total()/256;
	}
};


typedef std::function<void(single_stereo_task&, const cv::Mat&, const cv::Mat&, std::vector<disparity_region>&, int)> disparity_region_func;

//for IT metrics (region wise)
template<typename cost_type>
void calculate_region_disparity_regionwise(single_stereo_task& task, const cv::Mat& base, const cv::Mat& match, std::vector<disparity_region>& regions, int delta)
{
	const std::size_t regions_count = regions.size();

	auto it = std::max_element(regions.begin(), regions.end(), [](const disparity_region& lhs, const disparity_region& rhs) {
		return lhs.m_size < rhs.m_size;
	});

	cost_type cost_agg(base, match, it->size() * 3);
	typename cost_type::thread_type cost_thread;

	#pragma omp parallel for default(none) shared(task, regions, base, match, delta, cost_agg) private(cost_thread)
	for(std::size_t i = 0; i < regions_count; ++i)
	{
		auto range = getSubrange(regions[i].base_disparity, delta, task);
		regions[i].disparity_offset = range.first;

		int dilate = regions[i].dilation;
		if(dilate == regions[i].old_dilation)
			continue;

		std::vector<region_interval> actual_pixel_idx = dilated_region(regions[i], dilate, base);

		region_disparity_internal(actual_pixel_idx, cost_agg, cost_thread, regions[i], base, match, range.first, range.second);

		regions[i].old_dilation = dilate;
	}
}

class sad_disparitywise_calculator
{
public:
	typedef unsigned char result_type;

	sad_disparitywise_calculator(const cv::Mat& pbase, const cv::Mat& pmatch) : base(pbase), match(pmatch)
	{
	}

	cv::Mat operator ()(int d)
	{
		cv::Mat pbase = prepare_base(base, d);
		cv::Mat pmatch = prepare_match(match, d);

		cv::Mat diff;
		cv::absdiff(pbase, pmatch, diff);

		return diff; //TODO: result enlargement?, channel summation
	}

private:
	cv::Mat base, match;
};

template<typename calculator>
void calculate_region_generic(single_stereo_task& task, const cv::Mat& base, const cv::Mat& match, std::vector<disparity_region>& regions, int delta)
{
	std::cout << "delta: " << delta << std::endl;
	const std::size_t regions_count = regions.size();

	int crange = task.range_size();
	if(delta != 0)
		crange = 2*delta+1;

	for(std::size_t i = 0; i < regions_count; ++i)
	{
		auto range = getSubrange(regions[i].base_disparity, delta, task);
		regions[i].disparity_offset = range.first;
		regions[i].disparity_costs = cv::Mat(crange, 1, CV_32FC1, cv::Scalar(500));
	}

	calculator calc(base,match);

	#pragma omp parallel for
	for(int d = task.dispMin; d <= task.dispMax; ++d)
	{
		cv::Mat diff = calc(d);

		for(std::size_t i = 0; i < regions_count; ++i)
		{
			auto range = getSubrange(regions[i].base_disparity, delta, task);
			if(d>= range.first && d <= range.second)
			{
				std::vector<region_interval> filtered = filtered_region(base.size[1], regions[i].lineIntervals, d);
				cv::Mat diff_region = region_as_mat(diff, filtered, std::min(0, d));
				float sum = cv::norm(diff_region, cv::NORM_L1);

				if(diff_region.total() > 0)
					regions[i].disparity_costs(d-regions[i].disparity_offset) = sum/diff_region.total()/diff_region.channels();
				else
					regions[i].disparity_costs(d-regions[i].disparity_offset) = 1.0f;
			}
		}
	}

	for(std::size_t i=0; i < regions_count; ++i)
	{
		auto it = std::min_element(regions[i].disparity_costs.begin(), regions[i].disparity_costs.end());
		regions[i].disparity = std::distance(regions[i].disparity_costs.begin(), it) + regions[i].disparity_offset;
		/*EstimationStep step;
		step.costs = *it;
		step.disparity = regions[i].disparity;
		auto range = getSubrange(regions[i].base_disparity, delta, task);
		step.searchrange_start = range.first;
		step.searchrange_end = range.second;
		step.base_disparity = regions[i].base_disparity;
		regions[i].results.push_back(step);*/
	}
}

template<typename calculator>
void calculate_relaxed_region_generic(single_stereo_task& task, const cv::Mat& base, const cv::Mat& match, std::vector<disparity_region>& regions, int delta)
{
	std::cout << "delta: " << delta << std::endl;
	const std::size_t regions_count = regions.size();

	int crange = task.range_size();
	if(delta != 0)
		crange = 2*delta+1;

	for(std::size_t i = 0; i < regions_count; ++i)
		regions[i].disparity_costs = cv::Mat(crange, 1, CV_32FC1, cv::Scalar(500));

	//region:disp:y_interval
	//region:y_interval*disparity_range
	std::vector<std::vector<float> > row_costs;
	std::vector<std::vector<int> > row_sizes;

	calculator calc(base,match);

	//std::cout << "allocate" << std::endl;
	//allocate in a loop
	for(std::size_t i = 0; i < regions_count; ++i)
	{
		auto range = getSubrange(regions[i].base_disparity, delta, task);
		regions[i].disparity_offset = range.first;
		row_costs.emplace_back((range.second - range.first + 1) * regions[i].lineIntervals.size(), 1.0f);
		row_sizes.emplace_back((range.second - range.first + 1) * regions[i].lineIntervals.size(), 0);
	}

	std::cout << "rowwise" << std::endl;
	//rowwise costs
	int width = base.cols;
	#pragma omp parallel for
	for(int d = task.dispMin; d <= task.dispMax; ++d)
	{
		cv::Mat diff = calc(d);
		int base_offset = std::min(0,d);

		for(std::size_t i = 0; i < regions_count; ++i)
		{
			auto range = getSubrange(regions[i].base_disparity, delta, task);
			std::size_t idx = d - range.first;
			std::size_t count = regions[i].lineIntervals.size();

			if(d>= range.first && d <= range.second)
			{
				for(std::size_t j = 0; j < count; ++j)
				{
					const region_interval& cinterval = regions[i].lineIntervals[j];
					int lower = std::max(cinterval.lower+d, 0)-d + base_offset;
					int upper = std::min(cinterval.upper+d, width)-d + base_offset;
					int y = cinterval.y;

					float sum = 0.0;
					const float *diff_ptr = diff.ptr<float>(y,lower);
					for(int x = lower; x < upper; ++x)
						sum += *diff_ptr++;

					row_costs[i][idx*count+j] = sum;
					row_sizes[i][idx*count+j] = std::max(upper-lower+1,0);
				}
			}
		}
	}

	const float p = 0.9;
	std::array<float,3> penalties{0.0f, 1/p, 1/(p*p)};

	std::cout << "region" << std::endl;
	//calculate regioncost
	#pragma omp parallel for
	for(int d = task.dispMin; d <= task.dispMax; ++d)
	{
		for(std::size_t i = 0; i < regions_count; ++i)
		{
			auto range = getSubrange(regions[i].base_disparity, delta, task);
			std::size_t idx = d - range.first;
			std::size_t count = regions[i].lineIntervals.size();

			if(d>= range.first && d <= range.second)
			{
				int delta_neg = std::max((int)range.first, d - 2) - d;
				int delta_pos = std::min((int)range.second, d + 2) - d;

				float sum_costs = 0.0f;
				int sum_size = 0;
				for(std::size_t j = 0; j < count; ++j)
				{
					float rcost = std::numeric_limits<float>::max();
					int rsize = 0;
					for(int delta = delta_neg; delta <= delta_pos; ++delta)
					{
						float cpenanlty = penalties[std::abs(delta)];
						int delta_idx = delta*count;
						int csize = row_sizes[i][count*idx+j+delta_idx];
						float ccost = row_costs[i][count*idx+j+delta_idx] / csize * cpenanlty;

						if(ccost != 0 && ccost < rcost && csize > 0)
						{
							rcost = ccost;
							rsize = csize;
						}
					}
					if(rsize > 0)
					{
						sum_costs += rcost;
						sum_size += rsize;
					}
				}
				if(sum_size > 0)
					regions[i].disparity_costs(d-regions[i].disparity_offset) = sum_costs/sum_size;
				else
					regions[i].disparity_costs(d-regions[i].disparity_offset) = 2.0f;
			}
		}
	}

	for(std::size_t i=0; i < regions_count; ++i)
	{
		auto it = std::min_element(regions[i].disparity_costs.begin(), regions[i].disparity_costs.end());
		regions[i].disparity = std::distance(regions[i].disparity_costs.begin(), it) + regions[i].disparity_offset;
		/*EstimationStep step;
		step.costs = *it;
		step.disparity = regions[i].disparity;
		auto range = getSubrange(regions[i].base_disparity, delta, task);
		step.searchrange_start = range.first;
		step.searchrange_end = range.second;
		step.base_disparity = regions[i].base_disparity;
		regions[i].results.push_back(step);*/
	}
}

//untested
cv::Mat convertDisparityFromPartialCostmap(const cv::Mat& disparity, const cv::Mat& rangeCenters, int subsampling = 1)
{
	return disparity + rangeCenters * subsampling;
}

cv::Mat convertToFullDisparityCostMap(const cv::Mat& cost_map, const cv::Mat& rangeCenters, int dispMin, int dispMax, float initValue)
{
	assert(cost_map.dims == 3 && cost_map.size[0] > 0 && cost_map.size[1] > 0 && cost_map.size[2] > 0);
	int disparityRange = dispMax - dispMin + 1;
	int sz[] = {cost_map.size[0], cost_map.size[1], disparityRange};
	cv::Mat result(3, sz, CV_32FC1, cv::Scalar(initValue));

	for(int y = 0; y < cost_map.size[0]; ++y)
	{
		for(int x = 0; x < cost_map.size[1]; ++x)
		{
			const float* src_ptr = cost_map.ptr<float>(y,x,0);
			int dispSubStart = std::max(rangeCenters.at<short>(y,x)- cost_map.size[2]/2+1, dispMin);
			int dispSubEnd = std::min(rangeCenters.at<short>(y,x) + cost_map.size[2]/2, dispMax);
			//target constraints
			assert(dispSubStart - dispMin >= 0);
			assert(dispSubEnd - dispSubStart < disparityRange);
			assert(dispSubEnd - dispMin < disparityRange);
			//source constraints
			assert(dispSubEnd-dispSubStart+1 >= 0);
			assert(dispSubEnd-dispSubStart+1 <= cost_map.size[2]);
			float* result_ptr = result.ptr<float>(y, x, dispSubStart - dispMin);
			memcpy(result_ptr, src_ptr, (dispSubEnd-dispSubStart+1)*sizeof(float));
		}
	}

	return result;
}

void generate_region_information(region_container& left, region_container& right)
{
	std::cout << "warped_idx" << std::endl;
	refresh_warped_regions(left);
	refresh_warped_regions(right);
}

std::vector<region_interval> exposureVector(const cv::Mat& occlusionMap)
{
	std::vector<region_interval> exposure;

	const cv::Mat_<unsigned char> occ = occlusionMap;
	intervals::turn_value_into_intervals(occ, std::back_inserter(exposure), (unsigned char)0);

	cv::Mat_<unsigned char> temp = cv::Mat(occlusionMap.size(), CV_8UC1, cv::Scalar(0));
	intervals::set_region_value<unsigned char>(temp, exposure, (unsigned char)255);
	exposure.clear();

	cv::erode(temp, temp, cv::Mat(3,3, CV_8UC1, cv::Scalar(1)));
	matstore.add_mat(temp, "exposure");

	intervals::turn_value_into_intervals(temp, std::back_inserter(exposure), (unsigned char)255);

	return exposure;
}

void dilateLR(single_stereo_task& task, std::vector<disparity_region>& regions_base, int dilate_step, int delta)
{
	parallel_region(regions_base, [&](disparity_region& cregion) {
		generate_stats(cregion, task, delta);

		if(cregion.stats.confidence_range == 0 || cregion.stats.confidence_range > 2 || cregion.stats.confidence_variance > 0.2)
			cregion.dilation += dilate_step;
	});
}

void single_pass_region_disparity(stereo_task& task, region_container& left, region_container& right, const initial_disparity_config& config, bool b_refinement, disparity_region_func disparity_calculator, region_optimizer& optimizer)
{
	int refinement = 0;
	if(b_refinement)
		refinement = config.region_refinement_delta;

	region_descriptors::calculate_all_average_colors(task.forward.base, left.regions.begin(), left.regions.end());
	region_descriptors::calculate_all_average_colors(task.backward.base, right.regions.begin(), right.regions.end());

	std::cout << "corresponding regions" << std::endl;
	determine_corresponding_regions(left, right, 0);
	determine_corresponding_regions(right, left, 0);

	std::cout << "init disp" << std::endl;

	for(disparity_region& cregion : left.regions)
	{
		cregion.dilation = 0;
		cregion.old_dilation = -1;
		cregion.disparity_offset = task.forward.dispMin;
	}
	for(disparity_region& cregion : right.regions)
	{
		cregion.dilation = 0;
		cregion.old_dilation = -1;
		cregion.disparity_offset = task.backward.dispMin;
	}
	for(unsigned int i = 0; i <= config.dilate; i+= config.dilate_step)
	{
		std::cout << i << std::endl;
		disparity_calculator(task.forward,  task.algoLeft,  task.algoRight, left.regions, refinement);
		disparity_calculator(task.backward, task.algoRight, task.algoLeft, right.regions, refinement);

		std::cout << "dilation" << std::endl;
		dilateLR(task.forward,  left.regions,  config.dilate_step, refinement);
		dilateLR(task.backward, right.regions, config.dilate_step, refinement);
	}
	std::cout << "fin" << std::endl;

	if(config.verbose)
	{
		cv::Mat initial_disp_left  = disparity_by_segments(left);
		cv::Mat initial_disp_right = disparity_by_segments(right);

		matstore.add_mat(disparity::create_image(initial_disp_left), "init_left");
		matstore.add_mat(disparity::create_image(initial_disp_right), "right_left");
	}

	generate_region_information(left, right);
	generate_stats(left.regions, task.forward, refinement);
	generate_stats(right.regions, task.backward, refinement);

	optimizer.run(left, right, config.optimizer, b_refinement ? config.region_refinement_delta : 0);
	//run_optimization(left, right, config.optimizer, b_refinement ? config.region_refinement_delta : 0);

	assert(region_descriptors::invariants::check_labels_Intervals(left.regions.begin(), left.regions.end(), left.labels));
	assert(region_descriptors::invariants::check_labels_Intervals(right.regions.begin(), right.regions.end(), right.labels));

	cv::Mat opt_disp_left  = disparity_by_segments(left);
	cv::Mat opt_disp_right = disparity_by_segments(right);

	if(config.verbose)
	{
		matstore.add_mat(regionWiseImage<float>(left, [](const disparity_region& region){return region.stats.stddev;}), "stddev-left");
		matstore.add_mat(regionWiseImage<float>(right, [](const disparity_region& region){return region.stats.stddev;}), "stddev-right");
		matstore.add_mat(regionWiseImage<float>(left, [](const disparity_region& region){return region.stats.mean;}), "mean-left");
		matstore.add_mat(regionWiseImage<float>(left ,[](const disparity_region& region){return region.stats.confidence2;}), "confidence2-left");
		matstore.add_mat(regionWiseImage<float>(right ,[](const disparity_region& region){return region.stats.confidence2;}), "confidence2-right");
		matstore.add_mat(regionWiseImage<float>(left, [](const disparity_region& region){return region.stats.stddev/region.stats.mean;}), "stddev-norm");
		//matstore.addMat(regionWiseImage<float>(task.forward, left.regions, [&](const SegRegion& region){return region.disparity_costs(region.disparity-task.forward.dispMin);}), "opt-left");
		//matstore.addMat(regionWiseImage<float>(task.backward, right.regions, [&](const SegRegion& region){return region.disparity_costs(region.disparity-task.backward.dispMin);}), "opt-right");
		//matstore.addMat(regionWiseImage<float>(task.forward, left.regions, [&](const SegRegion& region){return region.confidence(region.disparity-task.forward.dispMin);}), "mi-conf-left");
		//matstore.addMat(regionWiseImage<float>(task.backward, right.regions, [&](const SegRegion& region){return region.confidence(region.disparity-task.backward.dispMin);}), "mi-conf-right");

		cv::Mat warped = disparity::warp_disparity<short>(opt_disp_left);
		cv::Mat occ_mat = disparity::occlusion_map<short>(opt_disp_left, warped);
		matstore.add_mat(value_scaled_image<unsigned char, unsigned char>(occ_mat), "occ");
	}
}

cv::Mat getNormalDisparity(cv::Mat& initial_disparity, const cv::Mat& costmap, const refinement_config& refconfig, int subsampling = 1)
{
	return convertDisparityFromPartialCostmap(disparity::create_from_costmap(costmap, -refconfig.deltaDisp/2+1, subsampling), initial_disparity, subsampling);
}

std::pair<cv::Mat, cv::Mat> segment_based_disparity_it(stereo_task& task, const initial_disparity_config& config , const refinement_config& refconfig, int subsampling, region_optimizer& optimizer)
{
	const int quantizer = 4;

	disparity_region_func disparity_function;
	refinement_func_type ref_func;
	if(config.metric_type == "sad")
	{
		//SAD
		typedef sliding_sad refinement_metric;
		disparity_function = calculate_region_generic<sad_disparitywise_calculator>;
		//disparity_function = calculate_relaxed_region_generic<sad_disparitywise_calculator>;
		task.algoLeft = task.left;
		task.algoRight = task.right;
		ref_func = refine_initial_disparity<refinement_metric, quantizer>;
	}
	else if(config.metric_type == "sncc")
	{
		typedef sliding_sad refinement_metric;
		//disparity_function = calculate_region_generic<sncc_disparitywise_calculator>;
		disparity_function = calculate_relaxed_region_generic<sncc_disparitywise_calculator>;
		task.algoLeft = task.leftGray;
		task.algoRight = task.rightGray;
		ref_func = refine_initial_disparity<refinement_metric, quantizer>;
	}
	else
	{
		typedef normalized_information_distance_calc<float> it_metric;
		//typedef normalized_variation_of_information_calc<float> it_metric;
		//typedef variation_of_information_calc<float> it_metric;

		typedef region_info_disparity<it_metric, quantizer> disparity_metric;

		//IT
		typedef costmap_creators::entropy::flexible_windowsize<it_metric, quantizer> refinement_metric;
		disparity_function = calculate_region_disparity_regionwise<disparity_metric>;
		task.algoLeft  = quantize_image(task.leftGray, quantizer);
		task.algoRight = quantize_image(task.rightGray, quantizer);
		ref_func = refine_initial_disparity<refinement_metric, quantizer>;
	}

	std::shared_ptr<region_container> left  = std::make_shared<region_container>();
	std::shared_ptr<region_container> right = std::make_shared<region_container>();

	//segment_based_disparity_internal(task, left, right, config, disparity_function);

	auto segmentationLeft  = create_segmentation_instance(config.segmentation);
	auto segmentationRight = create_segmentation_instance(config.segmentation);

	fill_region_container(left, task.forward, segmentationLeft);
	fill_region_container(right, task.backward, segmentationRight);

	for(disparity_region& cregion : left->regions)
		cregion.base_disparity = 0;
	for(disparity_region& cregion : right->regions)
		cregion.base_disparity = 0;

	single_pass_region_disparity(task, *left, *right, config, false, disparity_function, optimizer);

	//matstore.addMat(createDisparityImage(getDisparityBySegments(left)), "disp_fused_left");
	//matstore.addMat(createDisparityImage(getDisparityBySegments(right)), "disp_fused_right");

	initial_disparity_config config2 = config;
	for(int i = 0; i < config2.region_refinement_rounds; ++i)
	{
		//regionwise refinement
		if(config2.region_refinement_delta != 0)
		{
			std::cout << "refine: " << config2.region_refinement_delta << std::endl;
			//segmentationLeft->refine(*left);
			//segmentationRight->refine(*right);
			segmentationLeft->refinement(*left);
			segmentationRight->refinement(*right);

			//matstore.addMat(createDisparityImage(getDisparityBySegments(left)), "disp_unfused_left");
			//matstore.addMat(createDisparityImage(getDisparityBySegments(right)), "disp_unfused_right");

			single_pass_region_disparity(task, *left, *right, config2, true, disparity_function, optimizer);

			config2.region_refinement_delta /= 2;

			for(disparity_region cregion : left->regions)
				cregion.base_disparity = cregion.disparity;
			for(disparity_region cregion : right->regions)
				cregion.base_disparity = cregion.disparity;
		}
	}

	cv::Mat disparity_left  = disparity_by_segments(*left);
	cv::Mat disparity_right = disparity_by_segments(*right);

	//pixelwise refinement
	if(config.enable_refinement)
	{
		cv::Mat costmap_left  = ref_func(disparity_left, task.forward,  task.algoLeft,  task.algoRight, *left,  refconfig);
		cv::Mat costmap_right = ref_func(disparity_right, task.backward, task.algoRight, task.algoLeft,  *right, refconfig);

		disparity_left  = getNormalDisparity(disparity_left,  costmap_left, refconfig, subsampling);
		disparity_right = getNormalDisparity(disparity_right, costmap_right, refconfig, subsampling);
	}

	if(config.verbose)
	{
		matstore.set_region_container(left, right);
		matstore.add_mat(disparity_left, "disp_left");
		matstore.add_mat(disparity_right, "disp_right");
	}

	return std::make_pair(disparity_left, disparity_right);
}



initial_disparity_algo::initial_disparity_algo(initial_disparity_config &config, refinement_config &refconfig, std::shared_ptr<region_optimizer>& optimizer) :  m_optimizer(optimizer), m_config(config), m_refconfig(refconfig)
{
}

std::pair<cv::Mat, cv::Mat> initial_disparity_algo::operator ()(stereo_task& task)
{
	std::cout << "task: " << task.name << std::endl;
	int subsampling = 1; //TODO avoid this
	matstore.start_new_task(task.name, task);
	return segment_based_disparity_it(task, m_config, m_refconfig, subsampling, *m_optimizer);
}

void initial_disparity_algo::train(std::vector<stereo_task>& tasks)
{
	m_optimizer->set_training_mode(true);
	for(stereo_task& ctask : tasks)
		operator ()(ctask);
	m_optimizer->training();
}

void initial_disparity_algo::writeConfig(cv::FileStorage &fs)
{
	fs << m_config;
	fs << m_refconfig;
}

