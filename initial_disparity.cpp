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

#include "debugmatstore.h"

#include "region.h"
#include "fast_array.h"
#include "slidingEntropy.h"
#include "it_metrics.h"
#include "sparse_counter.h"
#include "costmap_creators.h"
#include "disparity_utils.h"
#include "intervals.h"
#include "intervals_algorithms.h"
#include "region_optimizer.h"
#include "configrun.h"
#include "misc.h"
#include "segmentation.h"
#include "region_descriptor.h"
#include "region_metrics.h"
#include "refinement.h"
#include "region_descriptor_algorithms.h"

#include <iterator>
#include <cstdlib>
#include <iostream>
#include <memory>

//for IT metrics (region wise)
template<typename cost_type>
void calculateRegionDisparity(StereoSingleTask& task, const cv::Mat& base, const cv::Mat& match, std::vector<DisparityRegion>& regions, unsigned int dilate, const std::vector<RegionInterval>& occ, int delta)
{
	const std::size_t regions_count = regions.size();

	auto it = std::max_element(regions.begin(), regions.end(), [](const DisparityRegion& lhs, const DisparityRegion& rhs) {
		return lhs.m_size < rhs.m_size;
	});

	cost_type cost_agg(base, match, it->size() * 10);
	typename cost_type::thread_type cost_thread;
	#pragma omp parallel for default(none) shared(task, regions, dilate, base, match, occ, delta, cost_agg) private(cost_thread)
	for(std::size_t i = 0; i < regions_count; ++i)
	{
		auto range = getSubrange(regions[i].base_disparity, delta, task);
		regions[i].disparity_offset = range.first;
		getRegionDisparity<cost_type >(cost_agg, cost_thread, regions[i], base, match, range.first, range.second, dilate, occ);
	}
}

//for SAD (disparity wise)
template<typename cost_type>
void calculate_region_disparity(StereoSingleTask& task, const cv::Mat& base, const cv::Mat& match, std::vector<DisparityRegion>& regions, unsigned int dilate, const std::vector<RegionInterval>& occ, int delta)
{
	std::cout << "delta: " << delta << std::endl;
	const std::size_t regions_count = regions.size();

	int crange = task.dispMax - task.dispMin + 1;
	if(delta != 0)
		crange = 2*delta+1;

	for(std::size_t i = 0; i < regions_count; ++i)
		regions[i].disparity_costs = cv::Mat(crange, 1, CV_32FC1, cv::Scalar(500));

	for(int d = task.dispMin; d <= task.dispMax; ++d)
	{
		cv::Mat pbase = prepare_base(base, d);
		cv::Mat pmatch = prepare_match(match, d);

		cv::Mat diff;
		cv::absdiff(pbase, pmatch, diff);

		for(std::size_t i = 0; i < regions_count; ++i)
		{
			auto range = getSubrange(regions[i].base_disparity, delta, task);
			regions[i].disparity_offset = range.first;
			if(d>= range.first && d <= range.second)
			{
				std::vector<RegionInterval> filtered = filter_region(regions[i].lineIntervals, d, occ, base.size[1]);

				cv::Mat diff_region = getRegionAsMat(diff, filtered, std::min(0, d));
				float sum = cv::norm(diff_region, cv::NORM_L1);

				if(diff_region.total() > 0)
					regions[i].disparity_costs(d-regions[i].disparity_offset) = sum/diff_region.total()/256/diff_region.channels();
				else
					regions[i].disparity_costs(d-regions[i].disparity_offset) = 2.0f;
			}
		}
	}

	for(std::size_t i=0; i < regions_count; ++i)
	{
		auto it = std::min_element(regions[i].disparity_costs.begin(), regions[i].disparity_costs.end());
		regions[i].disparity = std::distance(regions[i].disparity_costs.begin(), it) + regions[i].disparity_offset;
		EstimationStep step;
		step.costs = *it;
		step.disparity = regions[i].disparity;
		auto range = getSubrange(regions[i].base_disparity, delta, task);
		step.searchrange_start = range.first;
		step.searchrange_end = range.second;
		step.base_disparity = regions[i].base_disparity;
		regions[i].results.push_back(step);
		//TODO: needed?
		regions[i].old_dilation = regions[i].dilation;
	}
}

void fillRegionContainer(RegionContainer& result, StereoSingleTask& task, std::shared_ptr<segmentation_algorithm>& algorithm)
{
	result.task = task;

	int regions_count = cachedSegmentation(task, result.labels, algorithm);

	result.regions = std::vector<DisparityRegion>(regions_count);//getRegionVector(result.labels, regions_count);
	fillRegionDescriptors(result.regions.begin(), result.regions.end(), result.labels);
	//cv::Mat test = getWrongColorSegmentationImage(result.labels, regions_count);
	cv::Mat test = getWrongColorSegmentationImage(result);
	matstore.addMat(test, "segtest");

	std::cout << "regions count: " << regions_count << std::endl;


	//getAllRegionEntropies(task.baseGray, result.regions);

	generate_neighborhood(result.labels, result.regions);
}

//untested
cv::Mat convertDisparityFromPartialCostmap(const cv::Mat& disparity, const cv::Mat& rangeCenters, int subsampling = 1)
{
	cv::Mat result(rangeCenters.size(), CV_16SC1);

	short *dst_ptr = result.ptr<short>(0);
	const short *src_ptr = disparity.ptr<short>(0);
	const short *range_ptr = rangeCenters.ptr<short>(0);

	for(std::size_t i = 0; i < disparity.total(); ++i)
	{
		*dst_ptr++ = *src_ptr++ + *range_ptr++ * subsampling;
	}

	return result;
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

void generateOccStat(RegionContainer& container)
{
	/*cv::Mat occ_stat(container.task.base.size(), CV_8UC1, cv::Scalar(0));

	for(SegRegion& cregion : container.regions)
		intervals::addRegionValue<unsigned char>(occ_stat, cregion.warped_interval, 1);

	const std::size_t regions_count = container.regions.size();
	#pragma omp parallel for default(none) shared(container, occ_stat)
	for(std::size_t j = 0; j < regions_count; ++j)
	{
		SegRegion& cregion = container.regions[j];
		cregion.occlusion.fill(0);
		for(RegionInterval& cinterval : cregion.warped_interval)
		{
			for(int x = cinterval.lower; x < cinterval.upper; ++x)
			{
				int cocclusion = std::min((int)occ_stat.at<unsigned char>(cinterval.y, x), (int)cregion.occlusion.size()-1);
				cregion.occlusion[cocclusion] += 1;
			}
		}
		cregion.occlusion[0] = cregion.out_of_image;
		int occ_sum = 0;
		for(std::size_t i = 0; i < cregion.occlusion.size(); ++i)
			occ_sum += i*cregion.occlusion[i];
		cregion.occ_value = (float)occ_sum/cregion.size;
		cregion.stats.occ_val = cregion.occ_value;
	}*/
}

void generateRegionInformation(RegionContainer& left, RegionContainer& right)
{
	std::cout << "warped_idx" << std::endl;
	refreshWarpedIdx(left);
	refreshWarpedIdx(right);

	/*std::cout << "occ stat" << std::endl;
	generateOccStat(left);
	generateOccStat(right);*/
}

void generateFundamentalRegionInformation(StereoTask& task, RegionContainer& left, RegionContainer& right, int delta)
{
	std::cout << "stats" << std::endl;

	generateStats(left.regions, task.forward, delta);
	generateStats(right.regions, task.backward, delta);

	generateRegionInformation(left, right);
}

std::vector<ValueRegionInterval<short> > getIntervalDisparityBySegments(const RegionContainer& container, std::size_t exclude)
{
	std::vector<ValueRegionInterval<short> > result;

	for(std::size_t i = 0; i < container.regions.size(); ++i)
	{
		if(i != exclude)
		{
			result.reserve(result.size() + container.regions[i].lineIntervals.size());
			for(const RegionInterval& cinterval : container.regions[i].lineIntervals)
				result.push_back(ValueRegionInterval<short>(cinterval, container.regions[i].disparity));
		}
	}

	std::sort(result.begin(), result.end());

	return result;
}

std::vector<RegionInterval> exposureVector(const cv::Mat& occlusionMap)
{
	std::vector<RegionInterval> exposure;

	const cv::Mat_<unsigned char> occ = occlusionMap;
	intervals::turnValueIntoIntervals(occ, std::back_inserter(exposure), (unsigned char)0);

	cv::Mat_<unsigned char> temp = cv::Mat(occlusionMap.size(), CV_8UC1, cv::Scalar(0));
	intervals::setRegionValue<unsigned char>(temp, exposure, (unsigned char)255);
	exposure.clear();

	cv::erode(temp, temp, cv::Mat(3,3, CV_8UC1, cv::Scalar(1)));
	matstore.addMat(temp, "exposure");

	intervals::turnValueIntoIntervals(temp, std::back_inserter(exposure), (unsigned char)255);

	return exposure;
}

/*void dilateLR(StereoSingleTask& task, std::vector<SegRegion>& regions_base, std::vector<SegRegion>& regions_match, int dilate_step, int delta)
{
	const std::size_t regions_count = regions_base.size();
	#pragma omp parallel for default(none) shared(regions_base, regions_match, task, dilate_step, delta)
	for(std::size_t i = 0; i < regions_count; ++i)
	{
		/*float pot_lr = getOtherRegionsAverage(regions_match, regions_base[i].other_regions[regions_base[i].disparity-task.dispMin], [&](const SegRegion& cregion){return (float)std::min(std::abs(cregion.disparity+regions_base[i].disparity), 15);});
		if(pot_lr >= 5.0f)
			regions_base[i].dilation += dilate_step;*/

		/*generateStats(regions_base[i], task, delta);

		if(regions_base[i].stats.confidence_range == 0 || regions_base[i].stats.confidence_range > 2 || regions_base[i].stats.confidence_variance > 0.2)
			regions_base[i].dilation += dilate_step;

		//if(!((regions_base[i].stats.minima.size() < 2) && !(regions_base[i].stats.minima.size() == 0 && regions_base[i].stats.bad_minima.size() > 2)))
			//regions_base[i].dilation += dilate_step;
	}
}*/

void dilateLR(StereoSingleTask& task, std::vector<DisparityRegion>& regions_base, std::vector<DisparityRegion>& /*regions_match*/, int dilate_step, int delta)
{
	parallel_region(regions_base, [&](DisparityRegion& cregion) {
		generateStats(cregion, task, delta);

		if(cregion.stats.confidence_range == 0 || cregion.stats.confidence_range > 2 || cregion.stats.confidence_variance > 0.2)
			cregion.dilation += dilate_step;
	});
}

//template<typename disparity_metric>
void single_pass_region_disparity(StereoTask& task, RegionContainer& left, RegionContainer& right, const InitialDisparityConfig& config, bool b_refinement, std::function<void(StereoSingleTask&, const cv::Mat&, const cv::Mat&, std::vector<DisparityRegion>&, unsigned int, const std::vector<RegionInterval>&, int)> disparity_calculator)
{
	int refinement = 0;
	if(b_refinement)
		refinement = config.region_refinement_delta;

	calculate_all_average_colors(task.forward.base, left.regions);
	calculate_all_average_colors(task.backward.base, right.regions);

	std::cout << "lr-check" << std::endl;
	labelLRCheck(left.labels, right.labels, left.regions, task.forward, 0);
	labelLRCheck(right.labels, left.labels, right.regions, task.backward, 0);

	std::cout << "init disp" << std::endl;

	std::vector<RegionInterval> occ_left, occ_right;

	assert(config.occ_rounds >= 1);
	for(int j = 0; j < config.occ_rounds;++j)
	{
		for(DisparityRegion& cregion : left.regions)
		{
			cregion.dilation = 0;
			cregion.old_dilation = -1;
			cregion.disparity_offset = task.forward.dispMin;
		}
		for(DisparityRegion& cregion : right.regions)
		{
			cregion.dilation = 0;
			cregion.old_dilation = -1;
			cregion.disparity_offset = task.backward.dispMin;
		}
		for(unsigned int i = 0; i <= config.dilate; i+= config.dilate_step)
		{
			std::cout << i << std::endl;
			disparity_calculator(task.forward,  task.algoLeft,  task.algoRight, left.regions, config.dilate, occ_left, refinement);
			disparity_calculator(task.backward, task.algoRight, task.algoLeft, right.regions, config.dilate, occ_right, refinement);

			std::cout << "dilateLR" << std::endl;
			dilateLR(task.forward, left.regions, right.regions, config.dilate_step, refinement);
			dilateLR(task.backward, right.regions, left.regions, config.dilate_step, refinement);
		}
		std::cout << "fin" << std::endl;

		cv::Mat initial_disp_left  = getDisparityBySegments(left);
		cv::Mat initial_disp_right = getDisparityBySegments(right);

		if(config.verbose)
		{
			matstore.addMat(createDisparityImage(initial_disp_left), "init_left");
			matstore.addMat(createDisparityImage(initial_disp_right), "right_left");
		}

		generateFundamentalRegionInformation(task, left, right, refinement);

		run_optimization(task, left, right, config.optimizer, b_refinement ? config.region_refinement_delta : 0);

		assert(checkLabelsIntervalsInvariant(left.regions, left.labels, left.regions.size()));
		assert(checkLabelsIntervalsInvariant(right.regions, right.labels, right.regions.size()));

		cv::Mat opt_disp_left  = getDisparityBySegments(left);
		cv::Mat opt_disp_right = getDisparityBySegments(right);

		cv::Mat exp_left  = occlusionStat<short>(opt_disp_left, 1.0f);
		cv::Mat exp_right = occlusionStat<short>(opt_disp_right, 1.0f);

		occ_right = exposureVector(exp_left);
		occ_left  = exposureVector(exp_right);


		if(config.verbose)
		{
			matstore.addMat(regionWiseImage<unsigned char>(task.forward, left.regions, [](const DisparityRegion& region){return (unsigned char)region.stats.minima.size();}), "minima-left");
			matstore.addMat(regionWiseImage<float>(task.forward, left.regions, [](const DisparityRegion& region){return region.stats.stddev;}), "stddev-left");
			matstore.addMat(regionWiseImage<float>(task.backward, right.regions, [](const DisparityRegion& region){return region.stats.stddev;}), "stddev-right");
			matstore.addMat(regionWiseImage<float>(task.forward, left.regions, [](const DisparityRegion& region){return region.stats.mean;}), "mean-left");
			matstore.addMat(regionWiseImage<float>(task.forward, left.regions ,[](const DisparityRegion& region){return region.stats.confidence2;}), "confidence2-left");
			matstore.addMat(regionWiseImage<float>(task.backward, right.regions ,[](const DisparityRegion& region){return region.stats.confidence2;}), "confidence2-right");
			matstore.addMat(regionWiseImage<float>(task.forward, left.regions, [](const DisparityRegion& region){return region.stats.stddev/region.stats.mean;}), "stddev-norm");
			//matstore.addMat(regionWiseImage<float>(task.forward, left.regions, [&](const SegRegion& region){return region.disparity_costs(region.disparity-task.forward.dispMin);}), "opt-left");
			//matstore.addMat(regionWiseImage<float>(task.backward, right.regions, [&](const SegRegion& region){return region.disparity_costs(region.disparity-task.backward.dispMin);}), "opt-right");
			//matstore.addMat(regionWiseImage<float>(task.forward, left.regions, [&](const SegRegion& region){return region.confidence(region.disparity-task.forward.dispMin);}), "mi-conf-left");
			//matstore.addMat(regionWiseImage<float>(task.backward, right.regions, [&](const SegRegion& region){return region.confidence(region.disparity-task.backward.dispMin);}), "mi-conf-right");
			matstore.addMat(getValueScaledImage<unsigned char, unsigned char>(exp_left), "exp-left");
			matstore.addMat(getValueScaledImage<unsigned char, unsigned char>(exp_right), "exp-right");

			cv::Mat warped = warpDisparity<short>(opt_disp_left);
			cv::Mat occ_mat = occlusionMap<short>(opt_disp_left, warped);
			matstore.addMat(getValueScaledImage<unsigned char, unsigned char>(occ_mat), "occ");
		}
	}

	//exposure
	/*std::vector<RegionInterval> exposure = exposureVector(exp_left);

	calculateRegionDisparity<disparity_metric>(task.backward, right_quant, left_quant, right.regions, config.dilate, exposure);
	cv::Mat disp_occ = createDisparityImage(getDisparityBySegments(right));
	matstore.addMat(disp_occ, "disp_occ-right");

	run_optimization(task, left, right, config);

	cv::Mat disp_occ2 = createDisparityImage(getDisparityBySegments(right));
	matstore.addMat(disp_occ2, "disp_occ-right-opt");*/

	//cv::Mat cost_left  = getRegionCostmap(left.regions,  task.left.rows, task.left.cols, task.dispRange);
	//cv::Mat cost_right = getRegionCostmap(right.regions, task.left.rows, task.left.cols, task.dispRange);

	//return std::make_pair(cost_left, cost_right);
}

template<typename disparity_function>
void segment_based_disparity_internal(disparity_function func, StereoTask& task, RegionContainer& left, RegionContainer& right, const InitialDisparityConfig& config, std::shared_ptr<segmentation_algorithm>& algorithm)
{
	auto segmentationLeft  = getSegmentationClass(config.segmentation);
	auto segmentationRight = getSegmentationClass(config.segmentation);

	fillRegionContainer(left, task.forward, segmentationLeft);
	fillRegionContainer(right, task.backward, segmentationRight);

	for(DisparityRegion& cregion : left.regions)
		cregion.base_disparity = 0;
	for(DisparityRegion& cregion : right.regions)
		cregion.base_disparity = 0;

	//for SAD
	//single_pass_region_disparity(task, left, right, config, false, calculate_region_disparity<disparity_metric>);
	//for IT metrics
	single_pass_region_disparity(task, left, right, config, false, func);

	//matstore.addMat(createDisparityImage(getDisparityBySegments(left)), "disp_fused_left");
	//matstore.addMat(createDisparityImage(getDisparityBySegments(right)), "disp_fused_right");

	InitialDisparityConfig config2 = config;
	for(int i = 0; i < 2; ++i)
	{
		//regionwise refinement
		if(algorithm->refinementPossible() && config2.region_refinement_delta != 0)
		{
			std::cout << "refine: " << config2.region_refinement_delta << std::endl;
			segmentationLeft->refine(left);
			segmentationRight->refine(right);

			//matstore.addMat(createDisparityImage(getDisparityBySegments(left)), "disp_unfused_left");
			//matstore.addMat(createDisparityImage(getDisparityBySegments(right)), "disp_unfused_right");

			//for it metrics
			single_pass_region_disparity(task, left, right, config2, true, func);
			//for SAD
			//single_pass_region_disparity(task, left, right, config, true, calculate_region_disparity<disparity_metric>);

			config2.region_refinement_delta /= 2;

			for(DisparityRegion cregion : left.regions)
				cregion.base_disparity = cregion.disparity;
			for(DisparityRegion cregion : right.regions)
				cregion.base_disparity = cregion.disparity;
		}
	}
}

cv::Mat getNormalDisparity(cv::Mat& initial_disparity, const cv::Mat& costmap, const RefinementConfig& refconfig, int subsampling = 1)
{
	return convertDisparityFromPartialCostmap(createDisparity(costmap, -refconfig.deltaDisp/2+1, subsampling), initial_disparity, subsampling);
}

std::pair<cv::Mat, cv::Mat> segment_based_disparity_it(StereoTask& task, const InitialDisparityConfig& config , const RefinementConfig& refconfig, std::shared_ptr<segmentation_algorithm>& algorithm, int subsampling)
{
	const int quantizer = 4;

	typedef normalized_information_distance_calc<float> it_metric;
	//typedef normalized_variation_of_information_calc<float> it_metric;
	//typedef variation_of_information_calc<float> it_metric;
	typedef RegionInfoDisparityConf<it_metric, quantizer> disparity_metric;
	//typedef slidingEntropyFlex<it_metric, quantizer> refinement_metric;
	typedef slidingSAD refinement_metric;

	std::shared_ptr<RegionContainer> left  = std::make_shared<RegionContainer>();
	std::shared_ptr<RegionContainer> right = std::make_shared<RegionContainer>();

	//for IT
	/*task.algoLeft  = quantizeImage(task.leftGray, quantizer);
	task.algoRight = quantizeImage(task.rightGray, quantizer);
	segment_based_disparity_internal(calculateRegionDisparity<disparity_metric>, task, *left, *right, config, algorithm);*/
	//for SAD
	task.algoLeft = task.left;
	task.algoRight = task.right;
	segment_based_disparity_internal(calculate_region_disparity<disparity_metric>, task, *left, *right, config, algorithm);

	cv::Mat disparity_left  = getDisparityBySegments(*left);
	cv::Mat disparity_right = getDisparityBySegments(*right);

	//pixelwise refinement
	if(config.enable_refinement)
	{
		cv::Mat costmap_left  = refineInitialDisparity<refinement_metric, quantizer>(disparity_left, task.forward,  task.algoLeft,  task.algoRight, *left,  refconfig);
		cv::Mat costmap_right = refineInitialDisparity<refinement_metric, quantizer>(disparity_right, task.backward, task.algoRight, task.algoLeft,  *right, refconfig);

		disparity_left  = getNormalDisparity(disparity_left,  costmap_left, refconfig, subsampling);
		disparity_right = getNormalDisparity(disparity_right, costmap_right, refconfig, subsampling);
	}

	if(config.verbose)
		matstore.setRegionContainer(left, right);

	return std::make_pair(disparity_left, disparity_right);
}



