#include "region_descriptor.h"
#include "region_descriptor_algorithms.h"
#include "segmentation_refinement.h"
#include "intervals.h"
#include "intervals_algorithms.h"
#include <iostream>
#include <assert.h>
#include <vector>
#include <algorithm>
#include <iterator>

#include <gtest/gtest.h>

region_descriptor create_rectangle(cv::Rect rect)
{
	region_descriptor result;
	for(int i = 0; i < rect.height; ++i)
		result.lineIntervals.push_back(region_interval(i+rect.y, rect.x, rect.x + rect.width));

	result.bounding_box = rect;
	result.m_size = rect.area();

	return result;
}

TEST(SplitRegionTest, Basic)
{
	//quadratic case
	region_descriptor test = create_rectangle(cv::Rect(5,5,10,10));
	std::vector<region_descriptor> res1;
	int ret1 = split_region(test, 5, std::back_inserter(res1));

	EXPECT_EQ(res1.size(), 4);
	EXPECT_EQ(res1[0].size(), 25);
	EXPECT_EQ(res1[1].size(), 25);
	EXPECT_EQ(res1[2].size(), 25);
	EXPECT_EQ(res1[3].size(), 25);
	EXPECT_EQ(ret1, 4);

	//too small case (both directions)
	std::vector<region_descriptor> res2;
	int ret2 = split_region(test, 6, std::back_inserter(res2));
	EXPECT_EQ(ret2, 1);
	EXPECT_EQ(res2.size(), 1);
	EXPECT_EQ(res2[0].size(), 100);
}

TEST(SplitRegionTest, Basic2)
{
	//rectangular case
	region_descriptor test3 = create_rectangle(cv::Rect(10,15, 10,20));
	std::vector<region_descriptor> res3;
	int ret3 = split_region(test3, 4, std::back_inserter(res3));
	EXPECT_EQ(ret3, 4);
	EXPECT_EQ(res3.size(), 4);
	EXPECT_EQ(res3[0].size(), 50);
	EXPECT_EQ(res3[1].size(), 50);
	EXPECT_EQ(res3[2].size(), 50);
	EXPECT_EQ(res3[3].size(), 50);

	//to small case (one direction)
	std::vector<region_descriptor> res4;
	int ret4 = split_region(test3, 10, std::back_inserter(res4));
	EXPECT_EQ(ret4, 2);
	EXPECT_EQ(res4.size(), 2);
	EXPECT_EQ(res4[0].size(), 100);
	EXPECT_EQ(res4[1].size(), 100);

	//irregular shaped case
	region_descriptor test5 = test3;
	test5.lineIntervals.push_back(region_interval(30,15,40));
	test5.bounding_box.width = 25;
	test5.bounding_box.height += 1;
	std::vector<region_descriptor> res5;
	int ret5 = split_region(test5, 5, std::back_inserter(res5));
	EXPECT_EQ(ret5, 4);
	EXPECT_EQ(res5.size(), 4);
}

template<typename Iterator>
bool mismatch_output(Iterator it1begin, Iterator it1end, Iterator it2begin)
{
	auto it = std::mismatch(it1begin, it1end, it2begin);
	if(it.first != it1end)
	{
		std::cout << "first mismatch: " << *(it.first) << " vs " << *(it.second) << std::endl;
		return false;
	}
	return true;
}

TEST(Interval, Simple1)
{
	std::vector<region_interval> base {region_interval(0, 2, 6), region_interval(0, 10, 19), region_interval(0,22,29), region_interval(0,30,36), region_interval(1,2,6), region_interval(2,4,9)};
	std::vector<region_interval> match {region_interval(0, 3,5), region_interval(0,9,13), region_interval(0,15,21), region_interval(0,23,26), region_interval(0,36, 41), region_interval(2,5,7)};
	std::vector<region_interval> difference;
	//std::vector<RegionInterval> intersection;

	std::vector<region_interval> difference_desired {region_interval(0,2,3), region_interval(0,5,6), region_interval(0, 13, 15), region_interval(0,22,23), region_interval(0,26,29), region_interval(0,30,36), region_interval(1,2,6), region_interval(2,4,5),region_interval(2, 7, 9)};
	//std::vector<RegionInterval> intersection_desired {RegionInterval(0,3,5), RegionInterval(0,10,13), RegionInterval(0,15,19), RegionInterval(0,23,26), RegionInterval(2,5,7)};

	//intervals::intersection(base.begin(), base.end(), match.begin(), match.end(), 0, std::back_inserter(intersection), std::back_inserter(difference));

	//std::cout << std::endl << "--intersections--" << std::endl;
	//std::copy(intersection.begin(), intersection.end(), std::ostream_iterator<RegionInterval>(std::cout, " "));

	/*std::cout << std::endl << "--difference--" << std::endl;
	std::copy(difference.begin(), difference.end(), std::ostream_iterator<RegionInterval>(std::cout, " "));
	std::cout << std::endl;

	std::cout << "difference" << std::endl;*/
	EXPECT_TRUE(mismatch_output(difference.begin(), difference.end(), difference_desired.begin()));

	/*std::cout << "intersection" << std::endl;
	if(!mismatch_output(intersection.begin(), intersection.end(), intersection_desired.begin()))
		return false;*/
}

TEST(Interval, Simple2)
{
	std::vector<region_interval> base {region_interval(0,1,2), region_interval(0, 2, 6), region_interval(0, 10, 19), region_interval(0,22,29), region_interval(0,30,36), region_interval(1,2,6), region_interval(2,4,9), region_interval(5,1,9), region_interval(6,2,6)};
	std::vector<region_interval> match {region_interval(0,1,2), region_interval(0, 3,5), region_interval(0,9,13), region_interval(0,15,21), region_interval(0,23,26), region_interval(0,36, 41), region_interval(2,5,7), region_interval(3,5,9), region_interval(5,3,6), region_interval(6,2,6)};
	std::vector<region_interval> difference;
	//std::vector<RegionInterval> intersection;

	std::vector<region_interval> difference_desired {region_interval(0,2,3), region_interval(0,5,6), region_interval(0, 13, 15), region_interval(0,22,23), region_interval(0,26,29), region_interval(0,30,36), region_interval(1,2,6), region_interval(2,4,5),region_interval(2, 7, 9), region_interval(5,1,3), region_interval(5,6,9)};
	/*std::vector<RegionInterval> intersection_desired {RegionInterval(0,1,2), RegionInterval(0,3,5), RegionInterval(0,10,13), RegionInterval(0,15,19), RegionInterval(0,23,26), RegionInterval(2,5,7), RegionInterval(5,3,6), RegionInterval(6,2,6)};

	intervals::intersection(base.begin(), base.end(), match.begin(), match.end(), 0, std::back_inserter(intersection), std::back_inserter(difference));

	std::cout << std::endl << "--intersections--" << std::endl;
	std::copy(intersection.begin(), intersection.end(), std::ostream_iterator<RegionInterval>(std::cout, " "));*/

	/*std::cout << std::endl << "--difference--" << std::endl;
	std::copy(difference.begin(), difference.end(), std::ostream_iterator<RegionInterval>(std::cout, " "));
	std::cout << std::endl;*/

	//std::cout << "difference" << std::endl;
	EXPECT_TRUE(mismatch_output(difference.begin(), difference.end(), difference_desired.begin()));

	/*std::cout << "intersection" << std::endl;
	if(!mismatch_output(intersection.begin(), intersection.end(), intersection_desired.begin()))
		return false;*/
}

TEST(Interval, Simple3)
{
	std::vector<region_interval> base  {region_interval(0, 319, 329), region_interval(1, 318, 329), region_interval(2, 318, 329), region_interval(4, 318,329), region_interval(5, 318,329), region_interval(11, 317,318), region_interval(12, 317,318)};
	std::vector<region_interval> match {region_interval(0, 327, 328), region_interval(1, 326, 328), region_interval(2, 322, 328), region_interval(4, 321,322), region_interval(4, 323,328)};
	std::vector<region_interval> difference;
	//std::vector<RegionInterval> intersection;

	std::vector<region_interval> difference_desired {region_interval(0,319,327), region_interval(0,328,329), region_interval(1, 318, 326), region_interval(1,328,329), region_interval(2,318,322), region_interval(2,328,329), region_interval(4, 318,321), region_interval(4, 322,323), region_interval(4, 328,329), region_interval(5, 318,329), region_interval(11, 317,318), region_interval(12, 317,318)};
	std::vector<region_interval> intersection_desired {region_interval(0, 327, 328), region_interval(1, 326, 328), region_interval(2, 322, 328), region_interval(4, 321,322), region_interval(4, 323,328)};

	/*intervals::intersection(base.begin(), base.end(), match.begin(), match.end(), 0, std::back_inserter(intersection), std::back_inserter(difference));

	std::cout << std::endl << "--intersections--" << std::endl;
	std::copy(intersection.begin(), intersection.end(), std::ostream_iterator<RegionInterval>(std::cout, " "));*/

	/*std::cout << std::endl << "--difference--" << std::endl;
	std::copy(difference.begin(), difference.end(), std::ostream_iterator<RegionInterval>(std::cout, " "));
	std::cout << std::endl;*/

	//std::cout << "difference" << std::endl;
	EXPECT_TRUE(mismatch_output(difference.begin(), difference.end(), difference_desired.begin()));

	/*std::cout << "intersection" << std::endl;
	if(!mismatch_output(intersection.begin(), intersection.end(), intersection_desired.begin()))
		return false;

	std::vector<RegionInterval> diff_test;
	intervals::difference(base.begin(), base.end(), match.begin(), match.end(), 0, std::back_inserter(diff_test), true);


	std::cout << "difference2" << std::endl;
	if(!mismatch_output(diff_test.begin(), diff_test.end(), difference_desired.begin()))
		return false;*/
}

TEST(Interval, SimpleDisparity)
{
	std::vector<region_interval> base {region_interval(0, 2, 6), region_interval(0, 10, 19), region_interval(0,22,29), region_interval(0,30,36), region_interval(1,2,6), region_interval(2,4,9)};
	std::vector<region_interval> match {region_interval(0, 3,5), region_interval(0,9,13), region_interval(0,15,21), region_interval(0,23,26), region_interval(0,36, 41), region_interval(2,5,7)};
	std::vector<region_interval> difference;
	//std::vector<RegionInterval> intersection;

	std::vector<region_interval> difference_desired {region_interval(0,2,4), region_interval(0, 14, 16), region_interval(0,22,24), region_interval(0,27,29), region_interval(0,30,36), region_interval(1,2,6), region_interval(2,4,6),region_interval(2, 8, 9)};
	/*std::vector<RegionInterval> intersection_desired {RegionInterval(0,4,6), RegionInterval(0,10,14), RegionInterval(0,16,19), RegionInterval(0,24,27), RegionInterval(2,6,8)};

	intervals::intersection(base.begin(), base.end(), match.begin(), match.end(), -1, std::back_inserter(intersection), std::back_inserter(difference), true);

	std::cout << std::endl << "--intersections--" << std::endl;
	std::copy(intersection.begin(), intersection.end(), std::ostream_iterator<RegionInterval>(std::cout, " "));*/

	/*std::cout << std::endl << "--difference--" << std::endl;
	std::copy(difference.begin(), difference.end(), std::ostream_iterator<RegionInterval>(std::cout, " "));
	std::cout << std::endl;*/

	//std::cout << "difference" << std::endl;
	EXPECT_TRUE(mismatch_output(difference.begin(), difference.end(), difference_desired.begin()));

	/*std::cout << "intersection" << std::endl;
	if(!mismatch_output(intersection.begin(), intersection.end(), intersection_desired.begin()))
		return false;*/
}

TEST(Interval, ValueRegion)
{
	cv::Mat mat(1, 20, CV_8UC1, cv::Scalar(0));
	for(int i = 3; i <= 5; ++i)
		mat.at<unsigned char>(0,i) = 2;
	for(int i = 9; i <= 14; ++i)
		mat.at<unsigned char>(0,i) = 5;

	std::vector<value_region_interval<unsigned char>> desired = {value_region_interval<unsigned char>(0, 0, 3, 0), value_region_interval<unsigned char>(0, 3, 6, 2), value_region_interval<unsigned char>(0, 6, 9, 0), value_region_interval<unsigned char>(0, 9, 15, 5), value_region_interval<unsigned char>(0, 15, 20, 0)};

	std::vector<value_region_interval<unsigned char>> calc;

	cv::Mat_<unsigned char> mat1 = mat;

	intervals::convert_mat_to_value(mat1, std::back_inserter(calc));

	/*std::cout << "---- value intervals ----" << std::endl;
	std::copy(calc.begin(), calc.end(), std::ostream_iterator<RegionInterval>(std::cout, ", "));
	std::cout << std::endl;*/

	EXPECT_TRUE(mismatch_output(calc.begin(), calc.end(), desired.begin()));
}

//move up
/*TEST(Interval, ThresholdConvert)
{
	cv::Mat_<unsigned char> test(1, 20, 15);

	for(int i = 0; i < 5; ++i)
	{
		test.at<unsigned char>(i+5) = 5;
		test.at<unsigned char>(i+12) = 5;
	}
	std::vector<region_interval> result;
	intervals::convert_minima_ranges(test, std::back_inserter(result), (unsigned char)10);

	std::vector<region_interval> expected {region_interval(0, 5, 10), region_interval(0,12,17)};

	EXPECT_TRUE(mismatch_output(result.begin(), result.end(), expected.begin()));
}*/

//ported
TEST(Interval, Intersecting)
{
	region_interval test1(0,1,4);
	region_interval test2(0,2,5);
	region_interval test3(0,4,9);
	EXPECT_TRUE(test1.intersecting(test2));
	EXPECT_FALSE(test1.intersecting(test3));
	EXPECT_TRUE(test2.intersecting(test3));

	EXPECT_TRUE(test1.intersectingOrdered(test2));
	EXPECT_FALSE(test1.intersectingOrdered(test3));
	EXPECT_TRUE(test2.intersectingOrdered(test3));

	EXPECT_TRUE(test2.intersecting(test1));
	EXPECT_FALSE(test3.intersecting(test1));
	EXPECT_TRUE(test3.intersecting(test2));
}

/*void value_intersecting_test()
{
	std::cout << "---- value intersecting test ----" << std::endl;
	std::vector<ValueRegionInterval<short> > vec {ValueRegionInterval<short>(0,1,5,1), ValueRegionInterval<short>(0,3,8,2), ValueRegionInterval<short>(0,5,8,2)};
	std::vector<ValueRegionInterval<short> > desired {ValueRegionInterval<short>(0,1,2,1), ValueRegionInterval<short>(0,3,8,2)};

	auto value_compare = [](const ValueRegionInterval<short>& i1, const ValueRegionInterval<short>& i2) {
		return i1.value > i2.value;
	};

	intervals::cleanup<ValueRegionInterval<short>>(vec.begin(), vec.end(), value_compare);
	vec.erase(std::remove_if(vec.begin(), vec.end(), [](const ValueRegionInterval<short>& interval) {return interval.length() <= 0;}), vec.end());

	mismatch_output(vec.begin(), vec.end(), desired.begin());


}*/


