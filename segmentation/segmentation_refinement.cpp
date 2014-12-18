#include "segmentation_refinement.h"

#include "intervals.h"
#include "region_descriptor.h"

void hsplit_region(const region_descriptor& descriptor, region_descriptor& first, region_descriptor& second, int split_threshold)
{
	first.lineIntervals.clear();
	first.bounding_box.height = split_threshold - descriptor.bounding_box.y;

	second.lineIntervals.clear();
	second.bounding_box.y = split_threshold;
	second.bounding_box.height = descriptor.bounding_box.y + descriptor.bounding_box.height - split_threshold;

	for(const region_interval& cinterval : descriptor.lineIntervals)
	{
		if(cinterval.y < split_threshold)
			first.lineIntervals.push_back(cinterval);
		else
			second.lineIntervals.push_back(cinterval);
	}

	first.m_size = size_of_region(first.lineIntervals);
	second.m_size = size_of_region(second.lineIntervals);
}

void vsplit_region(const region_descriptor& descriptor, region_descriptor& first, region_descriptor& second, int split_threshold)
{
	first.lineIntervals.clear();
	first.bounding_box.width = split_threshold - descriptor.bounding_box.x;

	second.lineIntervals.clear();
	second.bounding_box.x = split_threshold;
	second.bounding_box.width = descriptor.bounding_box.x + descriptor.bounding_box.width - split_threshold;

	for(const region_interval& cinterval : descriptor.lineIntervals)
	{
		if(cinterval.upper > split_threshold && cinterval.lower <= split_threshold)
		{
			first.lineIntervals.push_back(region_interval(cinterval.y, cinterval.lower, split_threshold));
			second.lineIntervals.push_back(region_interval(cinterval.y, split_threshold, cinterval.upper));
		}
		else if(cinterval.upper <= split_threshold)
			first.lineIntervals.push_back(cinterval);
		else
			second.lineIntervals.push_back(cinterval);
	}

	first.m_size = size_of_region(first.lineIntervals);
	second.m_size = size_of_region(second.lineIntervals);
}

cv::Point region_avg_point(const region_descriptor& descriptor)
{
	long long x_avg = 0;
	long long y_avg = 0;

	for(const region_interval& cinterval : descriptor.lineIntervals)
	{
		x_avg += cinterval.upper - (cinterval.upper - cinterval.lower)/2 - 1;
		y_avg += cinterval.y;
	}
	x_avg /= descriptor.lineIntervals.size();
	y_avg /= descriptor.lineIntervals.size();

	return cv::Point(x_avg+1,y_avg+1);//because we want open intervalls +1
}

