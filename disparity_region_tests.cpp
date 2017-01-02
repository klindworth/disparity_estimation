#include <gtest/gtest.h>
#include <algorithm>

#include "disparity_region.h"
#include "disparity_region_algorithms.h"


TEST(FilteredRegionPositiveDisp, FilteredRegion)
{
	const int width = 55;
	const int disparity = 10;
	std::vector<region_interval> input {region_interval(1,10, 50), region_interval(2,10, 51), region_interval(3,45,60), region_interval(4,10, 35)};
	std::vector<region_interval> expected {region_interval(1,10,45), region_interval(2,10,45), region_interval(4,10,35)};

	std::vector<region_interval> actual = filtered_region(width, input, disparity);

	EXPECT_EQ(expected.size(), actual.size());

	EXPECT_TRUE(std::equal(expected.begin(), expected.end(), actual.begin(), actual.end()));
}

TEST(FilteredRegionNegativeDisp, FilteredRegion)
{
	const int width = 55;
	const int disparity = -20;
	std::vector<region_interval> input {region_interval(1,10, 50), region_interval(2,15, 51), region_interval(3,2,15), region_interval(4,10, 35)};
	std::vector<region_interval> expected {region_interval(1,20,50), region_interval(2,20,51), region_interval(4,20,35)};

	std::vector<region_interval> actual = filtered_region(width, input, disparity);

	EXPECT_EQ(expected.size(), actual.size());

	EXPECT_TRUE(std::equal(expected.begin(), expected.end(), actual.begin(), actual.end()));
}

TEST(ForeachWarpedIntervalPoint, ForeachAlgos)
{
	region_interval intv(5, 10, 15);
	std::vector<int> expected {12,13,14};
	std::vector<int> result;
	foreach_warped_interval_point(intv, 15, 2, [&](cv::Point pt) {
		EXPECT_EQ(5, pt.y);
		result.push_back(pt.x);
	});

	EXPECT_TRUE(std::equal(expected.begin(), expected.end(), result.begin(), result.end()));
}

TEST(ForeachWarpedIntervalPointNeg, ForeachAlgos)
{
	region_interval intv(5, 3, 7);
	std::vector<int> expected {0,1,2};
	std::vector<int> result;
	foreach_warped_interval_point(intv, 15, -4, [&](cv::Point pt) {
		EXPECT_EQ(5, pt.y);
		result.push_back(pt.x);
	});

	EXPECT_TRUE(std::equal(expected.begin(), expected.end(), result.begin(), result.end()));
}

TEST(ForeachFilteredIntervalPointNeg, ForeachAlgos)
{
	region_interval intv(5, 3, 7);
	std::vector<int> expected {4,5,6};
	std::vector<int> result;
	foreach_filtered_interval_point(intv, 15, -4, [&](cv::Point pt) {
		EXPECT_EQ(5, pt.y);
		result.push_back(pt.x);
	});

	EXPECT_TRUE(std::equal(expected.begin(), expected.end(), result.begin(), result.end()));
}

