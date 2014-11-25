#include "region_descriptor.h"
#include "region_descriptor_algorithms.h"
#include "segmentation_refinement.h"

#include <gtest/gtest.h>

RegionDescriptor create_rectangle(cv::Rect rect)
{
	RegionDescriptor result;
	for(int i = 0; i < rect.height; ++i)
		result.lineIntervals.push_back(RegionInterval(i+rect.y, rect.x, rect.x + rect.width));

	result.bounding_box = rect;
	result.m_size = rect.area();

	return result;
}

TEST(SplitRegionTest, Basic)
{
	//quadratic case
	RegionDescriptor test = create_rectangle(cv::Rect(5,5,10,10));
	std::vector<RegionDescriptor> res1;
	int ret1 = split_region(test, 5, std::back_inserter(res1));

	EXPECT_EQ(res1.size(), 4);
	EXPECT_EQ(res1[0].size(), 25);
	EXPECT_EQ(res1[1].size(), 25);
	EXPECT_EQ(res1[2].size(), 25);
	EXPECT_EQ(res1[3].size(), 25);
	EXPECT_EQ(ret1, 4);

	//too small case (both directions)
	std::vector<RegionDescriptor> res2;
	int ret2 = split_region(test, 6, std::back_inserter(res2));
	EXPECT_EQ(ret2, 1);
	EXPECT_EQ(res2.size(), 1);
	EXPECT_EQ(res2[0].size(), 100);
}

TEST(SplitRegionTest, Basic2)
{
	//rectangular case
	RegionDescriptor test3 = create_rectangle(cv::Rect(10,15, 10,20));
	std::vector<RegionDescriptor> res3;
	int ret3 = split_region(test3, 4, std::back_inserter(res3));
	EXPECT_EQ(ret3, 4);
	EXPECT_EQ(res3.size(), 4);
	EXPECT_EQ(res3[0].size(), 50);
	EXPECT_EQ(res3[1].size(), 50);
	EXPECT_EQ(res3[2].size(), 50);
	EXPECT_EQ(res3[3].size(), 50);

	//to small case (one direction)
	std::vector<RegionDescriptor> res4;
	int ret4 = split_region(test3, 10, std::back_inserter(res4));
	EXPECT_EQ(ret4, 2);
	EXPECT_EQ(res4.size(), 2);
	EXPECT_EQ(res4[0].size(), 100);
	EXPECT_EQ(res4[1].size(), 100);

	//irregular shaped case
	RegionDescriptor test5 = test3;
	test5.lineIntervals.push_back(RegionInterval(30,15,40));
	test5.bounding_box.width = 25;
	test5.bounding_box.height += 1;
	std::vector<RegionDescriptor> res5;
	int ret5 = split_region(test5, 5, std::back_inserter(res5));
	EXPECT_EQ(ret5, 4);
	EXPECT_EQ(res5.size(), 4);
}

