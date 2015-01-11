#ifndef CONVERTER_TO_REGION_H_
#define CONVERTER_TO_REGION_H_

#include <opencv2/core/core.hpp>
#include "disparity_range.h"

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
		//auto range = getSubrange(regions[i].base_disparity, delta, task);
		disparity_range drange = task_subrange(task, regions[i].base_disparity, delta);
		regions[i].disparity_offset = drange.start();
		regions[i].disparity_costs = cv::Mat(crange, 1, CV_32FC1, cv::Scalar(500));
	}

	calculator calc(base,match);

	#pragma omp parallel for
	for(int d = task.dispMin; d <= task.dispMax; ++d)
	{
		cv::Mat diff = calc(d);

		for(std::size_t i = 0; i < regions_count; ++i)
		{
			disparity_range drange = task_subrange(task, regions[i].base_disparity, delta);
			if(drange.valid(d))
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
		disparity_range drange = task_subrange(task, regions[i].base_disparity, delta);
		regions[i].disparity_offset = drange.start();
		row_costs.emplace_back(drange.size() * regions[i].lineIntervals.size(), 1.0f);
		row_sizes.emplace_back(drange.size() * regions[i].lineIntervals.size(), 0);
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
			disparity_range drange = task_subrange(task, regions[i].base_disparity, delta);
			std::size_t idx = d - drange.start();
			std::size_t count = regions[i].lineIntervals.size();

			if(drange.valid(d))
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
			disparity_range drange = task_subrange(task, regions[i].base_disparity, delta);
			std::size_t idx = d - drange.start();
			std::size_t count = regions[i].lineIntervals.size();

			if(drange.valid(d))
			{
				int delta_neg = std::max(drange.start(), d - 2) - d;
				int delta_pos = std::min(drange.end(), d + 2) - d;

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

#endif
