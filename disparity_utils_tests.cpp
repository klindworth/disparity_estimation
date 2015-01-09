#include "disparity_utils.h"
#include <gtest/gtest.h>

#include <iostream>
#include <algorithm>

TEST(DisparityUtils, MinTrivial)
{
	//std::size_t minimal_cost_disparity(const T* cost_ptr, int range, int dispMin)
	std::vector<int> input {-1,-2,-4,0,1};
	std::size_t idx = disparity::minimal_cost_disparity(input.data(), input.size(), -input.size());
	ASSERT_EQ(idx, 2);
}

TEST(DisparityUtils, MinSided)
{
	std::vector<int> input {0,0,-1,-1,2,2};
	std::size_t idx1 = disparity::minimal_cost_disparity(input.data(), input.size(), -input.size());
	std::size_t idx2 = disparity::minimal_cost_disparity(input.data(), input.size(), 0);
	//Prefere smaller disparity values
	ASSERT_EQ(idx1, 3); //with dispMin=-6 -> d=-3; the other -1 with idx=2 would be d=-4
	ASSERT_EQ(idx2, 2); //with dispMin=0 -> d=2; the other -1 with idx=3 would be d=3
}

TEST(DisparityUtils, WarpImageSingleRow)
{
	//WarpImage preferes rightmost value

	std::vector<int> input {1, 2,3,4, 5};
	std::vector<short> disp1  {0,-1,0,0,-2};
	std::vector<short> disp2  {2,0,0,1,0};
	std::vector<int> expected1 {2,0,5,4,0};
	//std::vector<int> expected1b {1,0,3,4,0};

	std::vector<int> expected2 {0,2,3,0,5};
	//std::vector<int> expected2b {0,2,1,0,4};

	cv::Mat_<int> input_mat = (cv::Mat_<int>(input)).reshape(1,1);
	cv::Mat_<short> disp1_mat = (cv::Mat_<short>(disp1)).reshape(1,1);
	cv::Mat_<short> disp2_mat = (cv::Mat_<short>(disp2)).reshape(1,1);

	cv::Mat_<int> res1 = disparity::warp_image(input_mat, disp1_mat);
	cv::Mat_<int> res2 = disparity::warp_image(input_mat, disp2_mat);

	ASSERT_TRUE(std::equal(res1.begin(), res1.end(), expected1.begin()));
	ASSERT_TRUE(std::equal(res2.begin(), res2.end(), expected2.begin()));
}

TEST(DisparityUtils, WarpImageMultiRow)
{
	//WarpImage preferes rightmost value

	std::vector<int> input {1, 2,3,4, 5,  6,7,8,9,10};
	std::vector<short> disp1  {0,-1,0,0,-2,  0,0,-1,0,0};
	std::vector<short> disp2  {2,0,0,1,0, 0,0,1,0,0};

	std::vector<int> expected1 {2,0,5,4,0,  6,8,0,9,10};
	std::vector<int> expected2 {0,2,3,0,5,  6,7,0,9,10};

	cv::Mat_<int> input_mat = (cv::Mat_<int>(input)).reshape(1,2);
	cv::Mat_<short> disp1_mat = (cv::Mat_<short>(disp1)).reshape(1,2);
	cv::Mat_<short> disp2_mat = (cv::Mat_<short>(disp2)).reshape(1,2);

	cv::Mat_<int> res1 = disparity::warp_image(input_mat, disp1_mat);
	cv::Mat_<int> res2 = disparity::warp_image(input_mat, disp2_mat);

	ASSERT_TRUE(std::equal(res1.begin(), res1.end(), expected1.begin()));
	ASSERT_TRUE(std::equal(res2.begin(), res2.end(), expected2.begin()));
}

TEST(DisparityUtils, OcclusionStat)
{
	//WarpImage preferes rightmost value

	std::vector<short> disp1  {0,-1,0,0,-2};
	std::vector<short> disp2  {2,0,0,1,0};
	std::vector<unsigned char> expected1 {2,0,2,1,0};

	std::vector<unsigned char> expected2 {0,1,2,0,2};

	cv::Mat_<short> disp1_mat = (cv::Mat_<short>(disp1)).reshape(1,1);
	cv::Mat_<short> disp2_mat = (cv::Mat_<short>(disp2)).reshape(1,1);

	cv::Mat_<unsigned char> res1 = disparity::occlusion_stat(disp1_mat);
	cv::Mat_<unsigned char> res2 = disparity::occlusion_stat(disp2_mat);

	ASSERT_TRUE(std::equal(res1.begin(), res1.end(), expected1.begin()));
	ASSERT_TRUE(std::equal(res2.begin(), res2.end(), expected2.begin()));
}

TEST(DisparityUtils, WarpDisparity)
{
	std::vector<short> disp1  {0,-1,0,0,-2};
	std::vector<short> disp2  {2,0,0,1,0};

	std::vector<short> expected1 {1,0,2,0,0};
	std::vector<short> expected2 {0,0,-2,0,-1};

	cv::Mat_<short> disp1_mat = (cv::Mat_<short>(disp1)).reshape(1,1);
	cv::Mat_<short> disp2_mat = (cv::Mat_<short>(disp2)).reshape(1,1);

	cv::Mat_<int> res1 = disparity::warp_disparity(disp1_mat);
	cv::Mat_<int> res2 = disparity::warp_disparity(disp2_mat);

	ASSERT_TRUE(std::equal(res1.begin(), res1.end(), expected1.begin()));
	ASSERT_TRUE(std::equal(res2.begin(), res2.end(), expected2.begin()));
}

