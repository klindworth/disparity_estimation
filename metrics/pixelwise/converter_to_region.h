#ifndef CONVERTER_TO_REGION_H_
#define CONVERTER_TO_REGION_H_

#include <opencv2/core/core.hpp>
#include "disparity_range.h"

template<typename calculator>
void calculate_region_generic(single_stereo_task& task, const cv::Mat& base, const cv::Mat& match, std::vector<disparity_region>& regions, int delta)
{
	std::cout << "delta: " << delta << std::endl;
	const std::size_t regions_count = regions.size();

	int crange = task.range.size();
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
	for(int d = task.range.start(); d <= task.range.end(); ++d)
	{
		cv::Mat diff = calc(d);

		for(std::size_t i = 0; i < regions_count; ++i)
		{
			disparity_range drange = task_subrange(task, regions[i].base_disparity, delta);
			if(drange.valid_disparity(d))
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

	int crange = task.range.size();
	if(delta != 0)
		crange = 2*delta+1;

	for(std::size_t i = 0; i < regions_count; ++i)
		regions[i].disparity_costs = cv::Mat(crange, 1, CV_32FC1, cv::Scalar(500));

	//memory layout
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
	const int width = base.cols;
	#pragma omp parallel for
	for(int d = task.range.start(); d <= task.range.end(); ++d)
	{
		cv::Mat diff = calc(d);
		const int base_offset = std::min(0,d);

		for(std::size_t i = 0; i < regions_count; ++i)
		{
			const disparity_range drange = task_subrange(task, regions[i].base_disparity, delta);

			if(drange.valid_disparity(d))
			{
				const std::size_t row_count = regions[i].lineIntervals.size();
				const std::size_t idx = drange.index(d)*row_count;

				for(std::size_t j = 0; j < row_count; ++j)
				{
					const region_interval& cinterval = regions[i].lineIntervals[j];
					const int lower = std::max(cinterval.lower+d, 0)-d + base_offset;
					const int upper = std::min(cinterval.upper+d, width)-d + base_offset;

					float sum = 0.0;
					const float *diff_ptr = diff.ptr<float>(cinterval.y,lower);
					for(int x = lower; x < upper; ++x)
						sum += *diff_ptr++;

					row_costs[i][idx+j] = sum;
					row_sizes[i][idx+j] = std::max(upper-lower+1,0);
				}
			}
		}
	}

	const float p = 0.9;
	const std::array<float,3> penalties{1.0f, 1/p, 1/(p*p)};

	std::cout << "region" << std::endl;
	//calculate regioncost
	//pragma omp parallel for
	for(int d = task.range.start(); d <= task.range.end(); ++d)
	{
		for(std::size_t i = 0; i < regions_count; ++i)
		{
			const disparity_range drange = task_subrange(task, regions[i].base_disparity, delta);

			if(drange.valid_disparity(d))
			{
				//std::size_t idx = d - drange.start();
				const std::size_t row_count = regions[i].lineIntervals.size();

				const int delta_neg = std::max(drange.start(), d - 2) - d;
				const int delta_pos = std::min(drange.end(), d + 2) - d;

				float region_costs = 0.0f;
				int region_size = 0;
				int actual_row_count = 0;
				for(std::size_t j = 0; j < row_count; ++j)
				{
					float row_cost = std::numeric_limits<float>::max();
					int row_size = 0;
					std::size_t idx = row_count*drange.index(d)+j;

					for(int delta = delta_neg; delta <= delta_pos; ++delta)
					{
						float cpenanlty = penalties[std::abs(delta)];
						int delta_idx = delta*row_count;
						int csize = row_sizes[i][idx+delta_idx];
						float ccost = row_costs[i][idx+delta_idx] / csize * cpenanlty;

						if(ccost != 0 && ccost < row_cost && csize > 0)
						{
							row_cost = ccost;
							row_size = csize;
						}
					}
					if(row_size > 0)
					{
						region_costs += row_cost;
						region_size += row_size;
						++actual_row_count;
					}
				}
				if(region_size > 0)
					regions[i].disparity_costs(d-regions[i].disparity_offset) = region_costs / actual_row_count;// /sum_size;
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
