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
#include <fstream>
#include <memory>

typedef std::function<void(StereoSingleTask&, const cv::Mat&, const cv::Mat&, std::vector<DisparityRegion>&, const std::vector<RegionInterval>&, int)> disparity_region_func;

//for IT metrics (region wise)
template<typename cost_type>
void calculate_region_disparity_regionwise(StereoSingleTask& task, const cv::Mat& base, const cv::Mat& match, std::vector<DisparityRegion>& regions, const std::vector<RegionInterval>& occ, int delta)
{
	const std::size_t regions_count = regions.size();

	auto it = std::max_element(regions.begin(), regions.end(), [](const DisparityRegion& lhs, const DisparityRegion& rhs) {
		return lhs.m_size < rhs.m_size;
	});

	cost_type cost_agg(base, match, it->size() * 10);
	typename cost_type::thread_type cost_thread;
	#pragma omp parallel for default(none) shared(task, regions, base, match, occ, delta, cost_agg) private(cost_thread)
	for(std::size_t i = 0; i < regions_count; ++i)
	{
		auto range = getSubrange(regions[i].base_disparity, delta, task);
		regions[i].disparity_offset = range.first;
		getRegionDisparity<cost_type >(cost_agg, cost_thread, regions[i], base, match, range.first, range.second, occ);
	}
}

//for SAD (disparity wise)
void calculate_region_sad(StereoSingleTask& task, const cv::Mat& base, const cv::Mat& match, std::vector<DisparityRegion>& regions, const std::vector<RegionInterval>& occ, int delta)
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

int cachedSegmentation(StereoSingleTask& task, cv::Mat_<int>& labels, std::shared_ptr<segmentation_algorithm>& algorithm)
{
	int regions_count = 0;
	if(algorithm->cacheAllowed())
	{
		std::string filename = "cache/" + task.fullname + "_" + algorithm->cacheName() + ".cache.cvmat";
		std::ifstream istream(filename, std::ifstream::binary);
		if(istream.is_open())
		{
			std::cout << "use cachefile: " << filename << std::endl;
			istream.read((char*)&regions_count, sizeof(int));
			labels = streamToMat(istream);
			istream.close();
		}
		else
		{
			std::cout << "create cachefile: " << filename << std::endl;
			regions_count = (*algorithm)(task.base, labels);

			std::ofstream ostream(filename, std::ofstream::binary);
			ostream.write((char*)&regions_count, sizeof(int));
			matToStream(labels, ostream);
			ostream.close();
		}
	}
	else
		regions_count = (*algorithm)(task.base, labels);
	return regions_count;
}

void fillRegionContainer(std::shared_ptr<RegionContainer>& result, StereoSingleTask& task, std::shared_ptr<segmentation_algorithm>& algorithm)
{
	result = algorithm->getSegmentationImage<RegionContainer>(task.base);
	result->task = task;

	matstore.addMat(getWrongColorSegmentationImage(result->task.base.size(), result->regions), "segtest");
	std::cout << "regions count: " << result->segment_count << std::endl;
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

void generateRegionInformation(RegionContainer& left, RegionContainer& right)
{
	std::cout << "warped_idx" << std::endl;
	refreshWarpedIdx(left);
	refreshWarpedIdx(right);
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

void dilateLR(StereoSingleTask& task, std::vector<DisparityRegion>& regions_base, std::vector<DisparityRegion>& /*regions_match*/, int dilate_step, int delta)
{
	parallel_region(regions_base, [&](DisparityRegion& cregion) {
		generateStats(cregion, task, delta);

		if(cregion.stats.confidence_range == 0 || cregion.stats.confidence_range > 2 || cregion.stats.confidence_variance > 0.2)
			cregion.dilation += dilate_step;
	});
}

void single_pass_region_disparity(StereoTask& task, RegionContainer& left, RegionContainer& right, const InitialDisparityConfig& config, bool b_refinement, disparity_region_func disparity_calculator)
{
	int refinement = 0;
	if(b_refinement)
		refinement = config.region_refinement_delta;

	calculate_all_average_colors(task.forward.base, left.regions);
	calculate_all_average_colors(task.backward.base, right.regions);

	std::cout << "lr-check" << std::endl;
	//labelLRCheck(left.labels, right.labels, left.regions, task.forward, 0);
	//labelLRCheck(right.labels, left.labels, right.regions, task.backward, 0);
	labelLRCheck(left, right, 0);
	labelLRCheck(right, left, 0);

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
			disparity_calculator(task.forward,  task.algoLeft,  task.algoRight, left.regions, occ_left, refinement);
			disparity_calculator(task.backward, task.algoRight, task.algoLeft, right.regions, occ_right, refinement);

			std::cout << "dilateLR" << std::endl;
			dilateLR(task.forward,  left.regions,  right.regions, config.dilate_step, refinement);
			dilateLR(task.backward, right.regions, left.regions,  config.dilate_step, refinement);
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
			matstore.addMat(regionWiseImage<unsigned char>(left, [](const DisparityRegion& region){return (unsigned char)region.stats.minima.size();}), "minima-left");
			matstore.addMat(regionWiseImage<float>(left, [](const DisparityRegion& region){return region.stats.stddev;}), "stddev-left");
			matstore.addMat(regionWiseImage<float>(right, [](const DisparityRegion& region){return region.stats.stddev;}), "stddev-right");
			matstore.addMat(regionWiseImage<float>(left, [](const DisparityRegion& region){return region.stats.mean;}), "mean-left");
			matstore.addMat(regionWiseImage<float>(left ,[](const DisparityRegion& region){return region.stats.confidence2;}), "confidence2-left");
			matstore.addMat(regionWiseImage<float>(right ,[](const DisparityRegion& region){return region.stats.confidence2;}), "confidence2-right");
			matstore.addMat(regionWiseImage<float>(left, [](const DisparityRegion& region){return region.stats.stddev/region.stats.mean;}), "stddev-norm");
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
}

void segment_based_disparity_internal(StereoTask& task, std::shared_ptr<RegionContainer>& left, std::shared_ptr<RegionContainer>& right, const InitialDisparityConfig& config, std::shared_ptr<segmentation_algorithm>& algorithm, disparity_region_func disparity_func)
{
	auto segmentationLeft  = getSegmentationClass(config.segmentation);
	auto segmentationRight = getSegmentationClass(config.segmentation);

	fillRegionContainer(left, task.forward, segmentationLeft);
	fillRegionContainer(right, task.backward, segmentationRight);

	for(DisparityRegion& cregion : left->regions)
		cregion.base_disparity = 0;
	for(DisparityRegion& cregion : right->regions)
		cregion.base_disparity = 0;

	single_pass_region_disparity(task, *left, *right, config, false, disparity_func);

	//matstore.addMat(createDisparityImage(getDisparityBySegments(left)), "disp_fused_left");
	//matstore.addMat(createDisparityImage(getDisparityBySegments(right)), "disp_fused_right");

	InitialDisparityConfig config2 = config;
	for(int i = 0; i < 2; ++i)
	{
		//regionwise refinement
		if(algorithm->refinementPossible() && config2.region_refinement_delta != 0)
		{
			std::cout << "refine: " << config2.region_refinement_delta << std::endl;
			segmentationLeft->refine(*left);
			segmentationRight->refine(*right);

			//matstore.addMat(createDisparityImage(getDisparityBySegments(left)), "disp_unfused_left");
			//matstore.addMat(createDisparityImage(getDisparityBySegments(right)), "disp_unfused_right");

			single_pass_region_disparity(task, *left, *right, config2, true, disparity_func);

			config2.region_refinement_delta /= 2;

			for(DisparityRegion cregion : left->regions)
				cregion.base_disparity = cregion.disparity;
			for(DisparityRegion cregion : right->regions)
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
	typedef slidingEntropyFlex<it_metric, quantizer> refinement_metric;
	//typedef slidingSAD refinement_metric;

	std::shared_ptr<RegionContainer> left  = std::make_shared<RegionContainer>();
	std::shared_ptr<RegionContainer> right = std::make_shared<RegionContainer>();

	//SAD
	/*auto disparity_function = calculate_region_sad;
	task.algoLeft = task.left;
	task.algoRight = task.right;*/
	//IT
	auto disparity_function = calculate_region_disparity_regionwise<disparity_metric>;
	task.algoLeft  = quantizeImage(task.leftGray, quantizer);
	task.algoRight = quantizeImage(task.rightGray, quantizer);

	segment_based_disparity_internal(task, left, right, config, algorithm, disparity_function);

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
