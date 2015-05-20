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

#include "region_metrics.h"
#include "refinement.h"

#include "disparitywise_calculator.h"
#include "sncc_disparitywise_calculator.h"
#include "disparity_region_algorithms.h"
#include "converter_to_region.h"
#include "sad_disparitywise.h"

class sliding_sad_threaddata
{
public:
	cv::Mat m_base;
	int cwindowsizeX;
	int cwindowsizeY;
	int crow;
	int cbase_x;
};

class sliding_sad
{
public:
	typedef float prob_table_type;
	typedef sliding_sad_threaddata thread_type;
private:

	cv::Mat m_base, m_match;

public:
	inline sliding_sad(const cv::Mat& base, const cv::Mat& match, const disparity_range&, unsigned int /*max_windowsize*/) : m_base(base), m_match(match)
	{
	}

	//prepares a row for calculation
	inline void prepare_row(thread_type& thread, int y)
	{
		thread.crow = y;
	}

	inline void prepare_window(thread_type& thread, int x, int cwindowsizeX, int cwindowsizeY)
	{
		thread.cwindowsizeX = cwindowsizeX;
		thread.cwindowsizeY = cwindowsizeY;
		//copy the window for L1 Cache friendlieness
		thread.cbase_x = x;
		thread.m_base = subwindow(m_base, x, thread.crow, cwindowsizeX, cwindowsizeY).clone();
	}

	inline prob_table_type increm(thread_type& thread, int x, int d)
	{
		cv::Mat match_window = subwindow(m_match, x+d, thread.crow, thread.cwindowsizeX, thread.cwindowsizeY).clone();

		return cv::norm(thread.m_base, match_window, cv::NORM_L1)/match_window.total()/256;
	}
};

class sliding_sncc
{
public:
	typedef float prob_table_type;
	typedef sliding_sad_threaddata thread_type;

private:
	cv::Mat m_base, m_match;
	std::vector<cv::Mat_<float> > results;

public:
	inline sliding_sncc(const cv::Mat& base, const cv::Mat& match, const disparity_range& range, unsigned int) : m_base(base), m_match(match)
	{
		sncc_disparitywise_calculator sncc(base, match);
		results.resize(range.size());
		for(int d = range.start(); d <= range.end(); ++d)
			results[std::abs(d)] = sncc(d);
	}

	//prepares a row for calculation
	inline void prepare_row(thread_type& thread, int y)
	{
		thread.crow = y;
	}

	inline void prepare_window(thread_type& thread, int x, int cwindowsizeX, int cwindowsizeY)
	{
		thread.cwindowsizeX = cwindowsizeX;
		thread.cwindowsizeY = cwindowsizeY;
		thread.cbase_x = x;
	}

	inline prob_table_type increm(thread_type& thread, int x, int d)
	{
		int d_offset = d < 0 ? d : 0;
		return cv::norm(subwindow(results[std::abs(d)], x+d_offset, thread.crow, thread.cwindowsizeX, thread.cwindowsizeY))/thread.cwindowsizeX/thread.cwindowsizeY;
	}
};


typedef std::function<void(single_stereo_task&, const cv::Mat&, const cv::Mat&, std::vector<disparity_region>&, int)> disparity_region_func;

//untested
disparity_map convertDisparityFromPartialCostmap(const disparity_map& disparity_delta, const disparity_map& initial_disparity, int subsampling = 1)
{
	return disparity_map(cv::Mat_<short>(disparity_delta + initial_disparity * subsampling), subsampling);
}

template<typename T, typename reg_type, typename lambda_type>
cv::Mat region_wise_image(const segmentation_image<reg_type>& image, lambda_type func)
{
	return value_scaled_image<T, unsigned char>(region_descriptors::set_regionwise<T>(image.image_size, image.regions, func));
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
		cregion.disparity_offset = task.forward.range.start();
	for(disparity_region& cregion : right.regions)
		cregion.disparity_offset = task.backward.range.start();

	disparity_calculator(task.forward,  task.algoLeft,  task.algoRight, left.regions, refinement);
	disparity_calculator(task.backward, task.algoRight, task.algoLeft, right.regions, refinement);

	std::cout << "fin" << std::endl;

	/*if(config.verbose)
	{
		cv::Mat initial_disp_left  = disparity_by_segments(left);
		cv::Mat initial_disp_right = disparity_by_segments(right);

		matstore.add_mat(disparity::create_image(initial_disp_left), "init_left");
		matstore.add_mat(disparity::create_image(initial_disp_right), "right_left");
	}*/

	generate_region_information(left, right);

	optimizer.run(left, right, config.optimizer, b_refinement ? config.region_refinement_delta : 0);
	//run_optimization(left, right, config.optimizer, b_refinement ? config.region_refinement_delta : 0);

	assert(region_descriptors::invariants::check_labels_Intervals(left.regions.begin(), left.regions.end(), left.labels));
	assert(region_descriptors::invariants::check_labels_Intervals(right.regions.begin(), right.regions.end(), right.labels));

	cv::Mat opt_disp_left  = disparity_by_segments(left);
	cv::Mat opt_disp_right = disparity_by_segments(right);

	if(config.verbose)
	{
		/*matstore.add_mat(region_wise_image<float>(left, [](const disparity_region& region){return region.stats.stddev;}), "stddev-left");
		matstore.add_mat(region_wise_image<float>(right, [](const disparity_region& region){return region.stats.stddev;}), "stddev-right");
		matstore.add_mat(region_wise_image<float>(left, [](const disparity_region& region){return region.stats.mean;}), "mean-left");
		matstore.add_mat(region_wise_image<float>(left ,[](const disparity_region& region){return region.stats.confidence2;}), "confidence2-left");
		matstore.add_mat(region_wise_image<float>(right ,[](const disparity_region& region){return region.stats.confidence2;}), "confidence2-right");
		matstore.add_mat(region_wise_image<float>(left, [](const disparity_region& region){return region.stats.stddev/region.stats.mean;}), "stddev-norm");*/
		//matstore.addMat(regionWiseImage<float>(task.forward, left.regions, [&](const SegRegion& region){return region.disparity_costs(region.disparity-task.forward.dispMin);}), "opt-left");
		//matstore.addMat(regionWiseImage<float>(task.backward, right.regions, [&](const SegRegion& region){return region.disparity_costs(region.disparity-task.backward.dispMin);}), "opt-right");
		//matstore.addMat(regionWiseImage<float>(task.forward, left.regions, [&](const SegRegion& region){return region.confidence(region.disparity-task.forward.dispMin);}), "mi-conf-left");
		//matstore.addMat(regionWiseImage<float>(task.backward, right.regions, [&](const SegRegion& region){return region.confidence(region.disparity-task.backward.dispMin);}), "mi-conf-right");

		cv::Mat warped = disparity::warp_disparity<short>(opt_disp_left);
		cv::Mat occ_mat = disparity::occlusion_map<short>(opt_disp_left, warped);
		//matstore.add_mat(value_scaled_image<unsigned char, unsigned char>(occ_mat), "occ");
	}
}

disparity_map getNormalDisparity(const disparity_map& initial_disparity, const cv::Mat& costmap, const refinement_config& refconfig, int subsampling = 1)
{
	return convertDisparityFromPartialCostmap(disparity::create_from_costmap(costmap, -refconfig.deltaDisp, subsampling), initial_disparity, subsampling);
}

std::pair<disparity_map, disparity_map> segment_based_disparity_it(stereo_task& task, const initial_disparity_config& config , const refinement_config& refconfig, int subsampling, region_optimizer& optimizer)
{
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
		ref_func = refine_initial_disparity<refinement_metric>;
	}
	else if(config.metric_type == "sncc")
	{
		//typedef sliding_sad refinement_metric;
		typedef sliding_sncc refinement_metric;
		//disparity_function = calculate_region_generic<sncc_disparitywise_calculator>;
		disparity_function = calculate_relaxed_region_generic<sncc_disparitywise_calculator>;
		task.algoLeft = task.leftGray;
		task.algoRight = task.rightGray;
		ref_func = refine_initial_disparity<refinement_metric>;
	}
	else
	{
		const int quantizer = 8;
		typedef normalized_information_distance_calc<float> it_metric;
		//typedef normalized_variation_of_information_calc<float> it_metric;
		//typedef variation_of_information_calc<float> it_metric;

		typedef region_info_disparity<it_metric, quantizer> disparity_metric;

		//IT
		typedef costmap_creators::entropy::flexible_windowsize<float, it_metric, quantizer> refinement_metric;
		disparity_function = calculate_region_disparity_regionwise<disparity_metric>;
		task.algoLeft  = quantize_image(task.leftGray, quantizer);
		task.algoRight = quantize_image(task.rightGray, quantizer);
		ref_func = refine_initial_disparity<refinement_metric>;
	}

	auto segmentationLeft  = create_segmentation_instance(config.segmentation);
	auto segmentationRight = create_segmentation_instance(config.segmentation);

	std::shared_ptr<region_container> left  =  segmentationLeft->segmentation_image<region_container>(task.forward.base,  task.forward);
	std::shared_ptr<region_container> right = segmentationRight->segmentation_image<region_container>(task.backward.base, task.backward);

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

	disparity_map disparity_left  = disparity_by_segments(*left);
	disparity_map disparity_right = disparity_by_segments(*right);

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

std::pair<disparity_map, disparity_map> initial_disparity_algo::operator ()(stereo_task& task)
{
	std::cout << "task: " << task.name << std::endl;
	//int subsampling = 1; //TODO avoid this
	int subsampling = task.groundTruthSubsampling;
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

