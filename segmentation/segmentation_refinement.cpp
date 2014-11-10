#include "segmentation_refinement.h"

#include "intervals.h"
#include "region_descriptor.h"

void hsplit_region(const RegionDescriptor& descriptor, RegionDescriptor& first, RegionDescriptor& second, int split_threshold)
{
	first.lineIntervals.clear();
	first.bounding_box.height = split_threshold - descriptor.bounding_box.y;

	second.lineIntervals.clear();
	second.bounding_box.y = split_threshold;
	second.bounding_box.height = descriptor.bounding_box.y + descriptor.bounding_box.height - split_threshold;

	for(const RegionInterval& cinterval : descriptor.lineIntervals)
	{
		if(cinterval.y < split_threshold)
			first.lineIntervals.push_back(cinterval);
		else
			second.lineIntervals.push_back(cinterval);
	}

	first.m_size = getSizeOfRegion(first.lineIntervals);
	second.m_size = getSizeOfRegion(second.lineIntervals);
}

void vsplit_region(const RegionDescriptor& descriptor, RegionDescriptor& first, RegionDescriptor& second, int split_threshold)
{
	first.lineIntervals.clear();
	first.bounding_box.width = split_threshold - descriptor.bounding_box.x;

	second.lineIntervals.clear();
	second.bounding_box.x = split_threshold;
	second.bounding_box.width = descriptor.bounding_box.x + descriptor.bounding_box.width - split_threshold;

	for(const RegionInterval& cinterval : descriptor.lineIntervals)
	{
		if(cinterval.upper > split_threshold && cinterval.lower <= split_threshold)
		{
			first.lineIntervals.push_back(RegionInterval(cinterval.y, cinterval.lower, split_threshold));
			second.lineIntervals.push_back(RegionInterval(cinterval.y, split_threshold, cinterval.upper));
		}
		else if(cinterval.upper <= split_threshold)
			first.lineIntervals.push_back(cinterval);
		else
			second.lineIntervals.push_back(cinterval);
	}

	first.m_size = getSizeOfRegion(first.lineIntervals);
	second.m_size = getSizeOfRegion(second.lineIntervals);
}

cv::Point region_avg_point(const RegionDescriptor& descriptor)
{
	long long x_avg = 0;
	long long y_avg = 0;

	for(const RegionInterval& cinterval : descriptor.lineIntervals)
	{
		x_avg += cinterval.upper - (cinterval.upper - cinterval.lower)/2 - 1;
		y_avg += cinterval.y;
	}
	x_avg /= descriptor.lineIntervals.size();
	y_avg /= descriptor.lineIntervals.size();

	return cv::Point(x_avg+1,y_avg+1);//because we want open intervalls +1
}

RegionDescriptor create_rectangle(cv::Rect rect)
{
	RegionDescriptor result;
	for(int i = 0; i < rect.height; ++i)
		result.lineIntervals.push_back(RegionInterval(i+rect.y, rect.x, rect.x + rect.width));

	result.bounding_box = rect;
	result.m_size = rect.area();

	return result;
}

void split_region_test()
{
	//quadratic case
	RegionDescriptor test = create_rectangle(cv::Rect(5,5,10,10));
	std::vector<RegionDescriptor> res1;
	int ret1 = split_region(test, 5, std::back_inserter(res1));

	assert(res1.size() == 4);
	assert(res1[0].size() == 25);
	assert(res1[1].size() == 25);
	assert(res1[2].size() == 25);
	assert(res1[3].size() == 25);
	assert(ret1 == 4);

	//too small case (both directions)
	std::vector<RegionDescriptor> res2;
	int ret2 = split_region(test, 6, std::back_inserter(res2));
	assert(ret2 == 1);
	assert(res2.size() == 1);
	assert(res2[0].size() == 100);

	//rectangular case
	RegionDescriptor test3 = create_rectangle(cv::Rect(10,15, 10,20));
	std::vector<RegionDescriptor> res3;
	int ret3 = split_region(test3, 4, std::back_inserter(res3));
	assert(ret3 == 4);
	assert(res3.size() == 4);
	assert(res3[0].size() == 50);
	assert(res3[1].size() == 50);
	assert(res3[2].size() == 50);
	assert(res3[3].size() == 50);

	//to small case (one direction)
	std::vector<RegionDescriptor> res4;
	int ret4 = split_region(test3, 10, std::back_inserter(res4));
	assert(ret4 == 2);
	assert(res4.size() == 2);
	assert(res4[0].size() == 100);
	assert(res4[1].size() == 100);

	//irregular shaped case
	RegionDescriptor test5 = test3;
	test5.lineIntervals.push_back(RegionInterval(30,15,40));
	test5.bounding_box.width = 25;
	test5.bounding_box.height += 1;
	std::vector<RegionDescriptor> res5;
	int ret5 = split_region(test5, 5, std::back_inserter(res5));
	assert(ret5 == 4);
	assert(res5.size() == 4);

	return;
}
