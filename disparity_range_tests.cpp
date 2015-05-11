#include "disparity_range.h"

#include <gtest/gtest.h>

TEST(DispRange, SubPos)
{
	disparity_range test1(0,128);
	disparity_range test2 = test1.subrange(64, 10);

	EXPECT_EQ(54, test2.start());
	EXPECT_EQ(74, test2.end());
	EXPECT_EQ(54, test2.index(54));
}

TEST(DispRange, SubNeg)
{
	disparity_range test1(-128,0);
	disparity_range test2 = test1.subrange(-64, 10);

	EXPECT_EQ(-74, test2.start());
	EXPECT_EQ(-54, test2.end());
	EXPECT_EQ(74, test2.index(-54));
}

TEST(DispRange, SubSpacePos)
{
	disparity_range test1(0,128);
	disparity_range test2 = test1.subrange_with_subspace(64, 10);

	EXPECT_EQ(54, test2.start());
	EXPECT_EQ(74, test2.end());
	EXPECT_EQ( 0, test2.index(54));
	EXPECT_EQ(20, test2.index(74));
}

TEST(DispRange, SubSpacePosBorder)
{
	disparity_range test1(0,128);
	disparity_range test2 = test1.subrange_with_subspace(5, 10);

	EXPECT_EQ( 0, test2.start());
	EXPECT_EQ(15, test2.end());
	EXPECT_EQ( 5, test2.index(0));
	EXPECT_EQ(20, test2.index(15));
}

TEST(DispRange, SubSpaceNegBorder)
{
	disparity_range test1(-128,0);
	disparity_range test2 = test1.subrange_with_subspace(-5, 10);

	EXPECT_EQ(-15, test2.start());
	EXPECT_EQ( 0, test2.end());
	EXPECT_EQ(15, test2.index(0));
	EXPECT_EQ( 0, test2.index(-15));
}
