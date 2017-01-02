#include "disparity_range.h"

#include <gtest/gtest.h>

TEST(DispRange, SubPos)
{
	disparity_range test1(0,128);
	EXPECT_EQ(129, test1.size());

	disparity_range test2 = test1.subrange(64, 10);

	EXPECT_EQ(54, test2.start());
	EXPECT_EQ(74, test2.end());
	EXPECT_EQ(54, test2.index(54));
	EXPECT_EQ(21, test2.size());
}

TEST(DispRange, SubNeg)
{
	disparity_range test1(-128,0);
	disparity_range test2 = test1.subrange(-64, 10);

	EXPECT_EQ(-74, test2.start());
	EXPECT_EQ(-54, test2.end());
	EXPECT_EQ(74, test2.index(-54));
	EXPECT_EQ(129, test1.size());
	EXPECT_EQ(21, test2.size());
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

TEST(DispRange, ImRestrictPos)
{
	disparity_range test(0,128);
	int width = 400;
	for(int i = 0; i < width; ++i)
	{
		disparity_range rtest = test.restrict_to_image(i, width);
		ASSERT_EQ(test.offset(), rtest.offset());
		ASSERT_EQ(0, rtest.start());
		if(i < width - test.end())
			ASSERT_EQ(rtest.size(), test.size());
		else
			ASSERT_EQ(width - 1, rtest.end() + i);
		ASSERT_TRUE(rtest.end() + i < width);
	}
}

TEST(DispRange, ImRestrictNeg)
{
	disparity_range test(-128,0);
	int width = 400;
	for(int i = 0; i < width; ++i)
	{
		disparity_range rtest = test.restrict_to_image(i, width);
		ASSERT_EQ(test.offset(), rtest.offset());
		ASSERT_EQ(0, rtest.end());
		if(i + test.start() < 0)
			ASSERT_EQ(0, rtest.start()+i);
		else
			ASSERT_EQ(rtest.size(), test.size());
		ASSERT_TRUE(rtest.start() + i >= 0);
	}
}
